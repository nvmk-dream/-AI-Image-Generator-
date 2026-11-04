package com.example.ai

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import android.content.Context
import android.graphics.Bitmap
import android.net.Uri
import com.example.data.ActiveDiffusionEngine
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import java.io.File
import java.nio.FloatBuffer
import java.nio.IntBuffer
import java.nio.LongBuffer
import java.nio.ShortBuffer
import java.util.EnumSet
import java.util.Locale
import java.util.concurrent.atomic.AtomicBoolean

data class GenerationProgress(
    val stageTitle: String,
    val currentStep: Int,
    val totalSteps: Int,
    val percent: Int,
    val elapsedMs: Long
)

data class TextToImageResult(
    val bitmap: Bitmap,
    val finalGenerationInput: String,
    val normalizedClipPrompt: String,
    val seed: Long,
    val steps: Int,
    val resolutionLabel: String,
    val durationMs: Long
)

data class ImageToImageResult(
    val bitmap: Bitmap,
    val croppedSourceFile: File,
    val finalGenerationInput: String,
    val normalizedClipPrompt: String,
    val strength: Float,
    val seed: Long,
    val activeSteps: Int,
    val resolutionLabel: String,
    val durationMs: Long
)

data class AvatarCustomizationResult(
    val bitmap: Bitmap,
    val croppedSourceFile: File,
    val tab: AvatarCustomTab,
    val usedSlot7AgingModel: Boolean,
    val usedMediaPipeVision: Boolean,
    val positivePrompt: String,
    val effectiveStrength: Float,
    val seed: Long,
    val activeSteps: Int,
    val resolutionLabel: String,
    val durationMs: Long
)

class OnnxDiffusionEngine(
    private val context: Context,
    private val modelManager: OnnxModelManager
) {
    private val tokenizer = ClipTokenizer(context)
    private val cancelRequested = AtomicBoolean(false)
    private val generationMutex = Mutex()
    private val runLock = Any()

    @Volatile
    private var activeRunOptions: OrtSession.RunOptions? = null

    fun requestCancel() {
        cancelRequested.set(true)
        synchronized(runLock) {
            runCatching { activeRunOptions?.setTerminate(true) }
        }
    }

    private fun runInterruptible(
        session: OrtSession,
        inputs: Map<String, OnnxTensor>
    ): OrtSession.Result {
        val opts = OrtSession.RunOptions()
        synchronized(runLock) {
            if (cancelRequested.get()) {
                runCatching { opts.close() }
                throw CancellationException("Generation cancelled by user.")
            }
            activeRunOptions = opts
        }
        return try {
            session.run(inputs, opts)
        } catch (t: Throwable) {
            if (cancelRequested.get()) {
                throw CancellationException("Generation cancelled by user.").apply { initCause(t) }
            }
            throw t
        } finally {
            synchronized(runLock) {
                if (activeRunOptions === opts) {
                    activeRunOptions = null
                }
            }
            runCatching { opts.close() }
        }
    }

    /**
     * Phase 3 Verification: Loads each installed ONNX slot one-by-one, inspects real
     * tensor names, types (FP16/FP32/INT64), and shapes, and releases RAM immediately.
     */
    suspend fun verifyInstalledModels(inferenceMode: String): Result<String> = runCatching {
        val env = OrtEnvironment.getEnvironment()
        val preferNnapi = inferenceMode.contains("NNAPI", ignoreCase = true)
        val lines = mutableListOf<String>()
        lines.add("ONNX Runtime v${env.version} (${if (preferNnapi) "NNAPI + CPU Fallback" else "CPU Multi-Thread"})")

        var verifiedCount = 0
        var incompleteCount = 0

        fun verifyOnnxTarget(label: String, file: File?) {
            if (file == null || !file.exists()) {
                lines.add("• $label: Not uploaded yet")
                return
            }
            val missingCompanions = OnnxModelManager.findMissingCompanionFiles(file)
            if (missingCompanions.isNotEmpty()) {
                lines.add(
                    "⚠️ $label (${OnnxModelManager.formatBytes(file.length())}) -> Incomplete: missing ${missingCompanions.joinToString()}"
                )
                incompleteCount++
                return
            }
            val t0 = System.currentTimeMillis()
            var repairTag = ""
            val session = createSafeSession(env, file, inferenceMode) { repairTag = " [Auto-Optimized INT8]" }
            session.use { s ->
                val dt = System.currentTimeMillis() - t0
                val inputsDesc = s.inputInfo.entries.joinToString { (name, node) ->
                    val ti = node.info as? TensorInfo
                    "$name:${ti?.type ?: "?"}${ti?.shape?.contentToString() ?: ""}"
                }
                lines.add("✅ $label$repairTag (${dt}ms) -> $inputsDesc")
                verifiedCount++
            }
        }

        currentCoroutineContext().ensureActive()
        verifyOnnxTarget("Slot 1A+1B Fast LCM UNet", modelManager.resolveEngineUnetFile(ActiveDiffusionEngine.FAST_LCM))
        currentCoroutineContext().ensureActive()
        verifyOnnxTarget("Slot 2A+2B Standard SD UNet", modelManager.resolveEngineUnetFile(ActiveDiffusionEngine.STANDARD_SD))
        currentCoroutineContext().ensureActive()
        verifyOnnxTarget("Slot 3 Text Encoder (CLIP)", modelManager.resolveExistingSlotFile(DiffusionModelSlot.TEXT_ENCODER))
        currentCoroutineContext().ensureActive()
        verifyOnnxTarget("Slot 4 VAE Decoder", modelManager.resolveExistingSlotFile(DiffusionModelSlot.VAE_DECODER))
        currentCoroutineContext().ensureActive()
        verifyOnnxTarget("Slot 5 VAE Encoder", modelManager.resolveExistingSlotFile(DiffusionModelSlot.VAE_ENCODER))

        val tokDir = modelManager.dedicatedFolder(DedicatedModelSlot.TOKENIZER)
        val hasCustomTok = File(tokDir, "vocab.json").exists() && File(tokDir, "merges.txt").exists()
        val tokIds = tokenizer.encode("photorealistic portrait")
        lines.add(
            "✅ Slot 6 Tokenizer -> ${if (hasCustomTok) "Custom vocab.json & merges.txt" else "Built-in CLIP vocab.json & merges.txt"} (${tokIds.size} tokens)"
        )

        val slot7AgingFile = modelManager.resolveExistingSlotFile(DiffusionModelSlot.FACE_AGING)
        if (slot7AgingFile != null) {
            currentCoroutineContext().ensureActive()
            verifyOnnxTarget("Slot 7 Optional Aging Engine", slot7AgingFile)
        }

        if (verifiedCount == 0 && incompleteCount == 0) {
            error("No .onnx files installed yet. Open Models tab to upload individual slots or select your IG folder.")
        }
        lines.joinToString("\n")
    }

    /**
     * Phase 4 Full Offline Text-to-Image Pipeline:
     * Final Generation Input = User Image Prompt + Saved Content Safety Prompt
     * Executes Text Encoder -> UNet (LCM or Euler) -> VAE Decoder sequentially on-device.
     */
    suspend fun generateTextToImage(
        userPrompt: String,
        finalGenerationInput: String,
        negativePrompt: String,
        resolutionLabel: String,
        inferenceMode: String,
        samplerType: SamplerType,
        steps: Int,
        guidanceScale: Float,
        seed: Long,
        activeEngine: ActiveDiffusionEngine = ActiveDiffusionEngine.FAST_LCM,
        onProgress: (GenerationProgress) -> Unit
    ): Result<TextToImageResult> = runCatching {
        currentCoroutineContext().ensureActive()
        generationMutex.withLock {
            currentCoroutineContext().ensureActive()
            cancelRequested.set(false)
            val startMs = System.currentTimeMillis()

        // 1. Mandatory safety check
        val safetyCheck = MandatorySafetyGuard.validateRequest(
            userPrompt = userPrompt,
            hasRealPersonReferencePhoto = false
        )
        if (!safetyCheck.allowed) {
            error(safetyCheck.messageEn ?: "Prompt blocked by safety check.")
        }

        // 2. Verify required Text-to-Image ONNX models & selected Active Engine pair (1A+1B or 2A+2B)
        val textEncoderFile = requireCompleteSlotFile(DiffusionModelSlot.TEXT_ENCODER, "Slot 3 (Text Encoder CLIP)")
        val unetFile = requireCompleteActiveEngineFile(activeEngine)
        val vaeDecoderFile = requireCompleteSlotFile(DiffusionModelSlot.VAE_DECODER, "Slot 4 (VAE Decoder)")

        val (width, height) = parseResolution(resolutionLabel)
        val latentW = width / 8
        val latentH = height / 8
        val latentChannels = 4
        val latentSize = latentChannels * latentH * latentW
        val env = OrtEnvironment.getEnvironment()

        val precheckLcm = detectLikelyLcmModel(unetFile = unetFile, unetSession = null)
        val plannedSampler = if (samplerType == SamplerType.LCM_FAST && !precheckLcm) {
            SamplerType.EULER_STANDARD
        } else {
            samplerType
        }
        val plannedSteps = if (plannedSampler == SamplerType.EULER_STANDARD && samplerType == SamplerType.LCM_FAST) {
            steps.coerceAtLeast(12)
        } else {
            steps
        }

        // 3. Tokenize Final Generation Input (User Prompt + Saved Content Safety Prompt)
        onProgress(
            GenerationProgress(
                stageTitle = "Encoding Prompt (CLIP Text Encoder)...",
                currentStep = 0,
                totalSteps = plannedSteps,
                percent = 6,
                elapsedMs = System.currentTimeMillis() - startMs
            )
        )
        checkNotCancelled()

        val normalizedClipPrompt = OfflineTamilPromptTranslator.normalizeForClip(finalGenerationInput)
        val posTokenIds = tokenizer.encode(normalizedClipPrompt)
        // Always encode negative prompt (~236KB) so Classifier-Free Guidance is ready if UNet is non-LCM
        val negTokenIds = tokenizer.encode(
            negativePrompt.ifBlank { "blurry, deformed, bad anatomy, low quality" }
        )

        // 4. Stage 1: Run Text Encoder & immediately release session to save RAM
        val (posEmbeddings, negEmbeddings, hiddenDim) = createSafeSession(env, textEncoderFile, inferenceMode).use { teSession ->
            val posOut = runTextEncoder(env, teSession, posTokenIds)
            checkNotCancelled()
            val negOut = runTextEncoder(env, teSession, negTokenIds).first
            Triple(posOut.first, negOut, posOut.second)
        }
        System.gc()

        onProgress(
            GenerationProgress(
                stageTitle = "Loading UNet Diffusion Engine...",
                currentStep = 0,
                totalSteps = plannedSteps,
                percent = 14,
                elapsedMs = System.currentTimeMillis() - startMs
            )
        )
        checkNotCancelled()

        var executedTotalSteps = plannedSteps

        // 5. Stage 2: Run UNet Denoising Loop (auto-adapts Euler CFG if standard SD 1.5 UNet is installed)
        val denoisedLatents = createSafeSession(
            env = env,
            modelFile = unetFile,
            inferenceMode = inferenceMode,
            onRepairStatus = { repairMsg ->
                onProgress(
                    GenerationProgress(
                        stageTitle = repairMsg,
                        currentStep = 0,
                        totalSteps = plannedSteps,
                        percent = 15,
                        elapsedMs = System.currentTimeMillis() - startMs
                    )
                )
            }
        ).use { unetSession ->
            val isLcmModel = detectLikelyLcmModel(unetFile = unetFile, unetSession = unetSession)
            val effectiveSampler = if (samplerType == SamplerType.LCM_FAST && !isLcmModel) {
                SamplerType.EULER_STANDARD
            } else {
                samplerType
            }
            val effectiveSteps = if (effectiveSampler == SamplerType.EULER_STANDARD && samplerType == SamplerType.LCM_FAST) {
                steps.coerceAtLeast(12)
            } else {
                steps
            }
            val effectiveGuidance = if (effectiveSampler == SamplerType.EULER_STANDARD &&
                samplerType == SamplerType.LCM_FAST &&
                guidanceScale < 4.0f
            ) {
                7.0f
            } else {
                guidanceScale
            }
            executedTotalSteps = effectiveSteps

            if (effectiveSampler == SamplerType.LCM_FAST) {
                val scheduler = DiffusionSchedulers.LcmSession(effectiveSteps)
                var latents = DiffusionSchedulers.createRandomLatents(latentSize, seed, initSigma = 1.0f)

                for (stepIdx in 0 until effectiveSteps) {
                    checkNotCancelled()
                    val t = scheduler.timesteps[stepIdx]
                    val startPct = 16 + (((stepIdx + 0.25f) / effectiveSteps.toFloat()) * 68f).toInt()
                    onProgress(
                        GenerationProgress(
                            stageTitle = "⚡ Fast LCM (Step ${stepIdx + 1}/$effectiveSteps)...",
                            currentStep = stepIdx + 1,
                            totalSteps = effectiveSteps,
                            percent = startPct,
                            elapsedMs = System.currentTimeMillis() - startMs
                        )
                    )

                    val noisePred = runUnetSingleBatch(
                        env = env,
                        unetSession = unetSession,
                        latents = latents,
                        latentH = latentH,
                        latentW = latentW,
                        timestep = t,
                        encoderHiddenStates = posEmbeddings,
                        hiddenDim = hiddenDim,
                        guidanceScale = effectiveGuidance,
                        imageWidth = width,
                        imageHeight = height
                    )
                    latents = scheduler.step(noisePred, stepIdx, latents, seed)

                    val donePct = 16 + (((stepIdx + 1).toFloat() / effectiveSteps.toFloat()) * 68f).toInt()
                    onProgress(
                        GenerationProgress(
                            stageTitle = "⚡ Fast LCM (Step ${stepIdx + 1}/$effectiveSteps ✓)...",
                            currentStep = stepIdx + 1,
                            totalSteps = effectiveSteps,
                            percent = donePct,
                            elapsedMs = System.currentTimeMillis() - startMs
                        )
                    )
                }
                latents
            } else {
                val scheduler = DiffusionSchedulers.EulerSession(effectiveSteps)
                var latents = DiffusionSchedulers.createRandomLatents(
                    size = latentSize,
                    seed = seed,
                    initSigma = scheduler.initNoiseSigma
                )
                val twoPassCfg = effectiveGuidance > 1.15f && !unetRequiresStaticBatchTwo(unetSession)

                for (stepIdx in 0 until effectiveSteps) {
                    checkNotCancelled()
                    val t = scheduler.timesteps[stepIdx]
                    val pass1Pct = 16 + (((stepIdx + 0.18f) / effectiveSteps.toFloat()) * 68f).toInt()
                    onProgress(
                        GenerationProgress(
                            stageTitle = if (twoPassCfg) {
                                "Denoising (Step ${stepIdx + 1}/$effectiveSteps • Pass 1/2)..."
                            } else {
                                "Denoising (Step ${stepIdx + 1}/$effectiveSteps)..."
                            },
                            currentStep = stepIdx + 1,
                            totalSteps = effectiveSteps,
                            percent = pass1Pct,
                            elapsedMs = System.currentTimeMillis() - startMs
                        )
                    )

                    val scaledLatents = scheduler.scaleModelInput(latents, stepIdx)
                    val finalNoise = runEulerNoisePrediction(
                        env = env,
                        unetSession = unetSession,
                        scaledLatents = scaledLatents,
                        latentH = latentH,
                        latentW = latentW,
                        timestep = t,
                        posEmbeddings = posEmbeddings,
                        negEmbeddings = negEmbeddings,
                        hiddenDim = hiddenDim,
                        effectiveGuidance = effectiveGuidance,
                        imageWidth = width,
                        imageHeight = height,
                        onSecondPassStart = {
                            val pass2Pct = 16 + (((stepIdx + 0.58f) / effectiveSteps.toFloat()) * 68f).toInt()
                            onProgress(
                                GenerationProgress(
                                    stageTitle = "Denoising (Step ${stepIdx + 1}/$effectiveSteps • Pass 2/2)...",
                                    currentStep = stepIdx + 1,
                                    totalSteps = effectiveSteps,
                                    percent = pass2Pct,
                                    elapsedMs = System.currentTimeMillis() - startMs
                                )
                            )
                        }
                    )

                    latents = scheduler.step(finalNoise, stepIdx, latents)
                }
                latents
            }
        }
        System.gc()

        // 6. Stage 3: Decode Latents into Bitmap via VAE Decoder
        onProgress(
            GenerationProgress(
                stageTitle = "Decoding Final Photo (VAE Decoder)...",
                currentStep = executedTotalSteps,
                totalSteps = executedTotalSteps,
                percent = 90,
                elapsedMs = System.currentTimeMillis() - startMs
            )
        )
        checkNotCancelled()

        val scaledForVae = FloatArray(denoisedLatents.size) { i ->
            denoisedLatents[i] / 0.18215f
        }

        val bitmap = createSafeSession(env, vaeDecoderFile, inferenceMode).use { vaeSession ->
            runVaeDecoderToBitmap(
                env = env,
                vaeSession = vaeSession,
                scaledLatents = scaledForVae,
                latentH = latentH,
                latentW = latentW,
                outWidth = width,
                outHeight = height
            )
        }
        System.gc()

        val totalMs = System.currentTimeMillis() - startMs
        onProgress(
            GenerationProgress(
                stageTitle = "Completed!",
                currentStep = executedTotalSteps,
                totalSteps = executedTotalSteps,
                percent = 100,
                elapsedMs = totalMs
            )
        )

        TextToImageResult(
            bitmap = bitmap,
            finalGenerationInput = finalGenerationInput,
            normalizedClipPrompt = normalizedClipPrompt,
            seed = seed,
            steps = executedTotalSteps,
            resolutionLabel = "$width × $height",
            durationMs = totalMs
        )
        }
    }

    /**
     * Phase 5 Full Offline Image-to-Image Pipeline:
     * 1. Auto Center-Crop Reference Photo -> vae_encoder.onnx (released immediately)
     * 2. Final Generation Input = User Image Prompt + Saved Content Safety Prompt -> text_encoder.onnx
     * 3. Forward Noise Injection at [strength] -> UNet Denoising (LCM or Euler)
     * 4. Decode Latents -> vae_decoder.onnx -> Output Bitmap + Before/After source preview
     */
    suspend fun generateImageToImage(
        sourceImageUri: Uri,
        userPrompt: String,
        finalGenerationInput: String,
        negativePrompt: String,
        strength: Float,
        resolutionLabel: String,
        inferenceMode: String,
        samplerType: SamplerType,
        steps: Int,
        guidanceScale: Float,
        seed: Long,
        activeEngine: ActiveDiffusionEngine = ActiveDiffusionEngine.FAST_LCM,
        onProgress: (GenerationProgress) -> Unit
    ): Result<ImageToImageResult> = runCatching {
        currentCoroutineContext().ensureActive()
        generationMutex.withLock {
            currentCoroutineContext().ensureActive()
            cancelRequested.set(false)
            val startMs = System.currentTimeMillis()

        // 1. Validate prompt with reference photo safety check
        val safetyCheck = MandatorySafetyGuard.validateRequest(
            userPrompt = userPrompt,
            hasRealPersonReferencePhoto = true
        )
        if (!safetyCheck.allowed) {
            error(safetyCheck.messageEn ?: "Please adjust your prompt and try again.")
        }

        // 2. Verify all required ONNX models & selected Active Engine pair (including Slot 5 VAE Encoder)
        val vaeEncoderFile = requireCompleteSlotFile(DiffusionModelSlot.VAE_ENCODER, "Slot 5 (VAE Encoder)")
        val textEncoderFile = requireCompleteSlotFile(DiffusionModelSlot.TEXT_ENCODER, "Slot 3 (Text Encoder CLIP)")
        val unetFile = requireCompleteActiveEngineFile(activeEngine)
        val vaeDecoderFile = requireCompleteSlotFile(DiffusionModelSlot.VAE_DECODER, "Slot 4 (VAE Decoder)")

        val (width, height) = parseResolution(resolutionLabel)
        val latentW = width / 8
        val latentH = height / 8
        val latentSize = 4 * latentH * latentW
        val env = OrtEnvironment.getEnvironment()

        val precheckLcm = detectLikelyLcmModel(unetFile = unetFile, unetSession = null)
        val preSampler = if (samplerType == SamplerType.LCM_FAST && !precheckLcm) {
            SamplerType.EULER_STANDARD
        } else {
            samplerType
        }
        val preSteps = if (preSampler == SamplerType.EULER_STANDARD && samplerType == SamplerType.LCM_FAST) {
            steps.coerceAtLeast(12)
        } else {
            steps
        }

        var startStepIndex = DiffusionSchedulers.computeImg2ImgStartStep(preSteps, strength)
        var activeSteps = (preSteps - startStepIndex).coerceAtLeast(1)

        // 3. Auto Center-Crop Reference Photo & Save 1:1 Before Preview
        onProgress(
            GenerationProgress(
                stageTitle = "Auto-Cropping & Encoding Reference Photo...",
                currentStep = 0,
                totalSteps = activeSteps,
                percent = 5,
                elapsedMs = System.currentTimeMillis() - startMs
            )
        )
        checkNotCancelled()

        val croppedRefBitmap = ImageBitmapPreprocessor.loadAndCenterCropFromUri(
            context = context,
            uri = sourceImageUri,
            targetWidth = width,
            targetHeight = height
        )
        val croppedSourceFile = ImageBitmapPreprocessor.saveReferencePreview(context, croppedRefBitmap)

        // 4. Stage 0: Run VAE Encoder & immediately close session to free RAM
        val cleanImageLatents = try {
            createSafeSession(env, vaeEncoderFile, inferenceMode).use { vaeEncSession ->
                runVaeEncoder(
                    env = env,
                    vaeEncSession = vaeEncSession,
                    bitmap = croppedRefBitmap,
                    latentSize = latentSize,
                    seed = seed
                )
            }
        } finally {
            croppedRefBitmap.recycle()
        }
        System.gc()

        // 5. Stage 1: Encode Final Generation Input (User Prompt + Saved Content Safety Prompt)
        onProgress(
            GenerationProgress(
                stageTitle = "Encoding Prompt (CLIP Text Encoder)...",
                currentStep = 0,
                totalSteps = activeSteps,
                percent = 14,
                elapsedMs = System.currentTimeMillis() - startMs
            )
        )
        checkNotCancelled()

        val normalizedClipPrompt = OfflineTamilPromptTranslator.normalizeForClip(finalGenerationInput)
        val posTokenIds = tokenizer.encode(normalizedClipPrompt)
        val negTokenIds = tokenizer.encode(
            negativePrompt.ifBlank { "blurry, deformed, bad anatomy, low quality" }
        )

        val (posEmbeddings, negEmbeddings, hiddenDim) = createSafeSession(env, textEncoderFile, inferenceMode).use { teSession ->
            val posOut = runTextEncoder(env, teSession, posTokenIds)
            checkNotCancelled()
            val negOut = runTextEncoder(env, teSession, negTokenIds).first
            Triple(posOut.first, negOut, posOut.second)
        }
        System.gc()

        // 6. Stage 2: Inject Strength Noise + Run UNet Denoising Loop
        onProgress(
            GenerationProgress(
                stageTitle = "Loading UNet (Strength ${(strength * 100).toInt()}%)...",
                currentStep = 0,
                totalSteps = activeSteps,
                percent = 22,
                elapsedMs = System.currentTimeMillis() - startMs
            )
        )
        checkNotCancelled()

        val denoisedLatents = createSafeSession(
            env = env,
            modelFile = unetFile,
            inferenceMode = inferenceMode,
            onRepairStatus = { repairMsg ->
                onProgress(
                    GenerationProgress(
                        stageTitle = repairMsg,
                        currentStep = 0,
                        totalSteps = activeSteps,
                        percent = 23,
                        elapsedMs = System.currentTimeMillis() - startMs
                    )
                )
            }
        ).use { unetSession ->
            val isLcmModel = detectLikelyLcmModel(unetFile = unetFile, unetSession = unetSession)
            val effectiveSampler = if (samplerType == SamplerType.LCM_FAST && !isLcmModel) {
                SamplerType.EULER_STANDARD
            } else {
                samplerType
            }
            val effectiveSteps = if (effectiveSampler == SamplerType.EULER_STANDARD && samplerType == SamplerType.LCM_FAST) {
                steps.coerceAtLeast(12)
            } else {
                steps
            }
            val effectiveGuidance = if (effectiveSampler == SamplerType.EULER_STANDARD &&
                samplerType == SamplerType.LCM_FAST &&
                guidanceScale < 4.0f
            ) {
                7.0f
            } else {
                guidanceScale
            }
            startStepIndex = DiffusionSchedulers.computeImg2ImgStartStep(effectiveSteps, strength)
            activeSteps = (effectiveSteps - startStepIndex).coerceAtLeast(1)

            if (effectiveSampler == SamplerType.LCM_FAST) {
                val scheduler = DiffusionSchedulers.LcmSession(effectiveSteps)
                var latents = scheduler.addNoise(cleanImageLatents, startStepIndex, seed)
                var executed = 0

                for (stepIdx in startStepIndex until effectiveSteps) {
                    checkNotCancelled()
                    executed++
                    val t = scheduler.timesteps[stepIdx]
                    val pct = 24 + ((executed.toFloat() / activeSteps.toFloat()) * 64f).toInt()
                    onProgress(
                        GenerationProgress(
                            stageTitle = "Transforming (LCM Step $executed/$activeSteps)...",
                            currentStep = executed,
                            totalSteps = activeSteps,
                            percent = pct,
                            elapsedMs = System.currentTimeMillis() - startMs
                        )
                    )

                    val noisePred = runUnetSingleBatch(
                        env = env,
                        unetSession = unetSession,
                        latents = latents,
                        latentH = latentH,
                        latentW = latentW,
                        timestep = t,
                        encoderHiddenStates = posEmbeddings,
                        hiddenDim = hiddenDim,
                        guidanceScale = effectiveGuidance,
                        imageWidth = width,
                        imageHeight = height
                    )
                    latents = scheduler.step(noisePred, stepIdx, latents, seed)
                }
                latents
            } else {
                val scheduler = DiffusionSchedulers.EulerSession(effectiveSteps)
                var latents = scheduler.addNoise(cleanImageLatents, startStepIndex, seed)
                var executed = 0
                val twoPassCfg = effectiveGuidance > 1.15f && !unetRequiresStaticBatchTwo(unetSession)

                for (stepIdx in startStepIndex until effectiveSteps) {
                    checkNotCancelled()
                    executed++
                    val t = scheduler.timesteps[stepIdx]
                    val pass1Pct = 24 + (((executed - 0.8f) / activeSteps.toFloat()) * 64f).toInt()
                    onProgress(
                        GenerationProgress(
                            stageTitle = if (twoPassCfg) {
                                "Transforming (Step $executed/$activeSteps • Pass 1/2)..."
                            } else {
                                "Transforming (Step $executed/$activeSteps)..."
                            },
                            currentStep = executed,
                            totalSteps = activeSteps,
                            percent = pass1Pct,
                            elapsedMs = System.currentTimeMillis() - startMs
                        )
                    )

                    val scaledLatents = scheduler.scaleModelInput(latents, stepIdx)
                    val finalNoise = runEulerNoisePrediction(
                        env = env,
                        unetSession = unetSession,
                        scaledLatents = scaledLatents,
                        latentH = latentH,
                        latentW = latentW,
                        timestep = t,
                        posEmbeddings = posEmbeddings,
                        negEmbeddings = negEmbeddings,
                        hiddenDim = hiddenDim,
                        effectiveGuidance = effectiveGuidance,
                        imageWidth = width,
                        imageHeight = height,
                        onSecondPassStart = {
                            val pass2Pct = 24 + (((executed - 0.4f) / activeSteps.toFloat()) * 64f).toInt()
                            onProgress(
                                GenerationProgress(
                                    stageTitle = "Transforming (Step $executed/$activeSteps • Pass 2/2)...",
                                    currentStep = executed,
                                    totalSteps = activeSteps,
                                    percent = pass2Pct,
                                    elapsedMs = System.currentTimeMillis() - startMs
                                )
                            )
                        }
                    )

                    latents = scheduler.step(finalNoise, stepIdx, latents)
                }
                latents
            }
        }
        System.gc()

        // 7. Stage 3: Decode Latents into Final Bitmap via VAE Decoder
        onProgress(
            GenerationProgress(
                stageTitle = "Decoding Final Photo (VAE Decoder)...",
                currentStep = activeSteps,
                totalSteps = activeSteps,
                percent = 92,
                elapsedMs = System.currentTimeMillis() - startMs
            )
        )
        checkNotCancelled()

        val scaledForVae = FloatArray(denoisedLatents.size) { i ->
            denoisedLatents[i] / 0.18215f
        }

        val bitmap = createSafeSession(env, vaeDecoderFile, inferenceMode).use { vaeSession ->
            runVaeDecoderToBitmap(
                env = env,
                vaeSession = vaeSession,
                scaledLatents = scaledForVae,
                latentH = latentH,
                latentW = latentW,
                outWidth = width,
                outHeight = height
            )
        }
        System.gc()

        val totalMs = System.currentTimeMillis() - startMs
        onProgress(
            GenerationProgress(
                stageTitle = "Completed!",
                currentStep = activeSteps,
                totalSteps = activeSteps,
                percent = 100,
                elapsedMs = totalMs
            )
        )

        ImageToImageResult(
            bitmap = bitmap,
            croppedSourceFile = croppedSourceFile,
            finalGenerationInput = finalGenerationInput,
            normalizedClipPrompt = normalizedClipPrompt,
            strength = strength,
            seed = seed,
            activeSteps = activeSteps,
            resolutionLabel = "$width × $height",
            durationMs = totalMs
        )
        }
    }

    /**
     * Full Offline Avatar Customization Pipeline (100% Realistic DSLR Photography ONLY):
     * 1. MediaPipe Tasks Vision (`FaceLandmarker` + `PoseLandmarker` + `Multiclass ImageSegmenter`)
     *    extracts exact Face/Hair/Torso/Clothes landmarks & semantic mask on-device.
     * 2. Optional Slot 7 (`face_aging.onnx`) hardware-accelerated path for Age Progression if uploaded.
     * 3. Latent-Space Masked Diffusion Inpainting (`VAE Encoder -> CLIP -> Masked UNet -> VAE Decoder`)
     *    followed by Identity-Locked Feathered Photo Compositing so unmasked face/hands/background
     *    remain 100% razor-sharp original DSLR pixels.
     */
    suspend fun generateAvatarCustomization(
        sourceImageUri: Uri,
        params: AvatarCustomizationParams,
        userBrushStrokes: List<MaskBrushStroke>,
        visionEngine: MediaPipeAvatarVisionEngine,
        safetyTailPrompt: String,
        resolutionLabel: String,
        inferenceMode: String,
        samplerType: SamplerType,
        steps: Int,
        guidanceScale: Float,
        activeEngine: ActiveDiffusionEngine = ActiveDiffusionEngine.FAST_LCM,
        onProgress: (GenerationProgress) -> Unit
    ): Result<AvatarCustomizationResult> = runCatching {
        currentCoroutineContext().ensureActive()
        generationMutex.withLock {
            currentCoroutineContext().ensureActive()
            cancelRequested.set(false)
            val startMs = System.currentTimeMillis()

            // 1. Validate ONLY raw user-typed note (never block internal DSLR presets or Tamil-to-English translations)
            if (params.customRealisticNote.isNotBlank()) {
                val safetyCheck = MandatorySafetyGuard.validateRequest(
                    userPrompt = params.customRealisticNote,
                    hasRealPersonReferencePhoto = true
                )
                if (!safetyCheck.allowed) {
                    error(safetyCheck.messageEn ?: "Please adjust your optional note and try again.")
                }
            }

            val (builtPrompt, computedStrength) = AvatarCustomizationCatalog.buildRealisticPromptAndStrength(params)
            val finalPositiveInput = if (safetyTailPrompt.isNotBlank()) {
                "$builtPrompt, $safetyTailPrompt"
            } else {
                builtPrompt
            }

            // 2. Pre-flight check of required ONNX model slots BEFORE allocating bitmaps or running MediaPipe
            val slot7Candidate = if (params.activeTab == AvatarCustomTab.AGE) {
                modelManager.resolveExistingSlotFile(DiffusionModelSlot.FACE_AGING)
                    ?.takeIf { OnnxModelManager.findMissingCompanionFiles(it).isEmpty() }
            } else {
                null
            }

            val coreFiles = if (slot7Candidate == null) {
                val vaeEnc = requireCompleteSlotFile(DiffusionModelSlot.VAE_ENCODER, "Slot 5 (VAE Encoder)")
                val textEnc = requireCompleteSlotFile(DiffusionModelSlot.TEXT_ENCODER, "Slot 3 (Text Encoder CLIP)")
                val unet = requireCompleteActiveEngineFile(activeEngine)
                val vaeDec = requireCompleteSlotFile(DiffusionModelSlot.VAE_DECODER, "Slot 4 (VAE Decoder)")
                listOf(vaeEnc, textEnc, unet, vaeDec)
            } else {
                null
            }

            val (width, height) = parseResolution(resolutionLabel)
            val env = OrtEnvironment.getEnvironment()
            val seed = params.seed

            onProgress(
                GenerationProgress(
                    stageTitle = "Analyzing Face, Pose & Mask (MediaPipe Vision)...",
                    currentStep = 0,
                    totalSteps = steps.coerceAtLeast(4),
                    percent = 6,
                    elapsedMs = System.currentTimeMillis() - startMs
                )
            )
            checkNotCancelled()

            val originalCropped = ImageBitmapPreprocessor.loadAndCenterCropFromUri(
                context = context,
                uri = sourceImageUri,
                targetWidth = width,
                targetHeight = height
            )

            val croppedSourceFile: File
            val usedMediaPipe: Boolean
            val featheredMask: FloatArray
            var cleanImageLatents: FloatArray? = null

            try {
                croppedSourceFile = ImageBitmapPreprocessor.saveReferencePreview(context, originalCropped)

                val analysis = visionEngine.analyzePhoto(originalCropped)
                usedMediaPipe = analysis.usedMediaPipeFace || analysis.usedMediaPipePose || analysis.usedMediaPipeSegmenter
                featheredMask = visionEngine.buildFeatheredMask(
                    analysis = analysis,
                    params = params,
                    userStrokes = userBrushStrokes,
                    outWidth = width,
                    outHeight = height
                )

                // Check Optional Slot 7 Dedicated Face-Aging ONNX Model first when in AGE tab
                if (slot7Candidate != null) {
                    onProgress(
                        GenerationProgress(
                            stageTitle = "Running Slot 7 Face-Aging ONNX Engine (Age ${params.targetAge})...",
                            currentStep = 1,
                            totalSteps = 2,
                            percent = 45,
                            elapsedMs = System.currentTimeMillis() - startMs
                        )
                    )
                    val slot7AgedBitmap = runCatching {
                        createSafeSession(env, slot7Candidate, inferenceMode).use { agingSession ->
                            AvatarPhotoCompositor.tryRunOptionalSlot7AgingModel(
                                env = env,
                                session = agingSession,
                                sourceBitmap = originalCropped,
                                targetAge = params.targetAge
                            )
                        }
                    }.getOrNull()

                    if (slot7AgedBitmap != null) {
                        val composited = try {
                            AvatarPhotoCompositor.compositeIdentityLockedPhoto(
                                original = originalCropped,
                                generated = slot7AgedBitmap,
                                featheredMask = featheredMask
                            )
                        } finally {
                            slot7AgedBitmap.recycle()
                        }
                        val totalMs = System.currentTimeMillis() - startMs
                        onProgress(
                            GenerationProgress(
                                stageTitle = "Completed (Slot 7 Aging Engine)!",
                                currentStep = 2,
                                totalSteps = 2,
                                percent = 100,
                                elapsedMs = totalMs
                            )
                        )
                        return@withLock AvatarCustomizationResult(
                            bitmap = composited,
                            croppedSourceFile = croppedSourceFile,
                            tab = params.activeTab,
                            usedSlot7AgingModel = true,
                            usedMediaPipeVision = usedMediaPipe,
                            positivePrompt = builtPrompt,
                            effectiveStrength = computedStrength,
                            seed = seed,
                            activeSteps = 1,
                            resolutionLabel = "$width × $height",
                            durationMs = totalMs
                        )
                    }
                }

                // Encode original cropped photo into latent space BEFORE releasing originalCropped bitmap from RAM
                val vaeEncForFallback = coreFiles?.get(0)
                    ?: requireCompleteSlotFile(DiffusionModelSlot.VAE_ENCODER, "Slot 5 (VAE Encoder)")
                val latentSize = 4 * (height / 8) * (width / 8)

                onProgress(
                    GenerationProgress(
                        stageTitle = "Encoding Original Photo (VAE Encoder)...",
                        currentStep = 0,
                        totalSteps = steps.coerceAtLeast(4),
                        percent = 14,
                        elapsedMs = System.currentTimeMillis() - startMs
                    )
                )
                checkNotCancelled()

                cleanImageLatents = createSafeSession(env, vaeEncForFallback, inferenceMode).use { vaeEncSession ->
                    runVaeEncoder(
                        env = env,
                        vaeEncSession = vaeEncSession,
                        bitmap = originalCropped,
                        latentSize = latentSize,
                        seed = seed
                    )
                }
            } finally {
                originalCropped.recycle()
            }
            System.gc()

            // Core 1A–6 Latent-Space Masked Diffusion Inpainting Pipeline
            val textEncoderFile = coreFiles?.get(1)
                ?: requireCompleteSlotFile(DiffusionModelSlot.TEXT_ENCODER, "Slot 3 (Text Encoder CLIP)")
            val unetFile = coreFiles?.get(2)
                ?: requireCompleteActiveEngineFile(activeEngine)
            val vaeDecoderFile = coreFiles?.get(3)
                ?: requireCompleteSlotFile(DiffusionModelSlot.VAE_DECODER, "Slot 4 (VAE Decoder)")

            val encodedCleanLatents = cleanImageLatents
                ?: error("Failed to encode reference image latents.")

            val latentW = width / 8
            val latentH = height / 8
            val latentMask = AvatarPhotoCompositor.downsampleMaskToLatentGrid(
                pixelMask = featheredMask,
                imgW = width,
                imgH = height,
                latentW = latentW,
                latentH = latentH
            )

            onProgress(
                GenerationProgress(
                    stageTitle = "Encoding Realistic DSLR Attributes (CLIP)...",
                    currentStep = 0,
                    totalSteps = steps.coerceAtLeast(4),
                    percent = 22,
                    elapsedMs = System.currentTimeMillis() - startMs
                )
            )
            checkNotCancelled()

            val normalizedClipPrompt = OfflineTamilPromptTranslator.normalizeForClip(finalPositiveInput)
            val posTokenIds = tokenizer.encode(normalizedClipPrompt)
            val negTokenIds = tokenizer.encode(AvatarCustomizationCatalog.STRICT_ANTI_NON_PHOTO_NEGATIVE_PROMPT)

            val (posEmbeddings, negEmbeddings, hiddenDim) = createSafeSession(env, textEncoderFile, inferenceMode).use { teSession ->
                val posOut = runTextEncoder(env, teSession, posTokenIds)
                checkNotCancelled()
                val negOut = runTextEncoder(env, teSession, negTokenIds).first
                Triple(posOut.first, negOut, posOut.second)
            }
            System.gc()

            var activeSteps = steps.coerceAtLeast(1)
            val denoisedLatents = createSafeSession(
                env = env,
                modelFile = unetFile,
                inferenceMode = inferenceMode,
                onRepairStatus = { repairMsg ->
                    onProgress(
                        GenerationProgress(
                            stageTitle = repairMsg,
                            currentStep = 0,
                            totalSteps = activeSteps,
                            percent = 28,
                            elapsedMs = System.currentTimeMillis() - startMs
                        )
                    )
                }
            ).use { unetSession ->
                val isLcmModel = detectLikelyLcmModel(unetFile = unetFile, unetSession = unetSession)
                val effectiveSampler = if (samplerType == SamplerType.LCM_FAST && !isLcmModel) {
                    SamplerType.EULER_STANDARD
                } else {
                    samplerType
                }
                val effectiveSteps = if (effectiveSampler == SamplerType.EULER_STANDARD && samplerType == SamplerType.LCM_FAST) {
                    steps.coerceAtLeast(14)
                } else {
                    steps
                }
                val effectiveGuidance = if (effectiveSampler == SamplerType.EULER_STANDARD &&
                    samplerType == SamplerType.LCM_FAST &&
                    guidanceScale < 4.0f
                ) {
                    7.0f
                } else {
                    guidanceScale
                }
                val startStepIndex = DiffusionSchedulers.computeImg2ImgStartStep(effectiveSteps, computedStrength)
                activeSteps = (effectiveSteps - startStepIndex).coerceAtLeast(1)

                if (effectiveSampler == SamplerType.LCM_FAST) {
                    val scheduler = DiffusionSchedulers.LcmSession(effectiveSteps)
                    var latents = scheduler.addNoise(encodedCleanLatents, startStepIndex, seed)
                    var executed = 0

                    for (stepIdx in startStepIndex until effectiveSteps) {
                        checkNotCancelled()
                        executed++
                        val t = scheduler.timesteps[stepIdx]
                        val pct = 30 + ((executed.toFloat() / activeSteps.toFloat()) * 56f).toInt()
                        onProgress(
                            GenerationProgress(
                                stageTitle = "Inpainting ${params.activeTab.shortTitleEn} (LCM $executed/$activeSteps)...",
                                currentStep = executed,
                                totalSteps = activeSteps,
                                percent = pct,
                                elapsedMs = System.currentTimeMillis() - startMs
                            )
                        )

                        val noisePred = runUnetSingleBatch(
                            env = env,
                            unetSession = unetSession,
                            latents = latents,
                            latentH = latentH,
                            latentW = latentW,
                            timestep = t,
                            encoderHiddenStates = posEmbeddings,
                            hiddenDim = hiddenDim,
                            guidanceScale = effectiveGuidance,
                            imageWidth = width,
                            imageHeight = height
                        )
                        val editedNext = scheduler.step(noisePred, stepIdx, latents, seed)
                        // Use identical deterministic seed so reference trajectory shares the exact base Gaussian noise
                        val refAtNext = if (stepIdx + 1 < effectiveSteps) {
                            scheduler.addNoise(encodedCleanLatents, stepIdx + 1, seed)
                        } else {
                            encodedCleanLatents
                        }
                        latents = AvatarPhotoCompositor.blendLatentStep(editedNext, refAtNext, latentMask)
                    }
                    latents
                } else {
                    val scheduler = DiffusionSchedulers.EulerSession(effectiveSteps)
                    var latents = scheduler.addNoise(encodedCleanLatents, startStepIndex, seed)
                    var executed = 0
                    val twoPassCfg = effectiveGuidance > 1.15f && !unetRequiresStaticBatchTwo(unetSession)

                    for (stepIdx in startStepIndex until effectiveSteps) {
                        checkNotCancelled()
                        executed++
                        val t = scheduler.timesteps[stepIdx]
                        val pct = 30 + (((executed - 0.7f) / activeSteps.toFloat()) * 56f).toInt()
                        onProgress(
                            GenerationProgress(
                                stageTitle = if (twoPassCfg) {
                                    "Inpainting ${params.activeTab.shortTitleEn} (Step $executed/$activeSteps • 1/2)..."
                                } else {
                                    "Inpainting ${params.activeTab.shortTitleEn} (Step $executed/$activeSteps)..."
                                },
                                currentStep = executed,
                                totalSteps = activeSteps,
                                percent = pct,
                                elapsedMs = System.currentTimeMillis() - startMs
                            )
                        )

                        val scaledLatents = scheduler.scaleModelInput(latents, stepIdx)
                        val finalNoise = runEulerNoisePrediction(
                            env = env,
                            unetSession = unetSession,
                            scaledLatents = scaledLatents,
                            latentH = latentH,
                            latentW = latentW,
                            timestep = t,
                            posEmbeddings = posEmbeddings,
                            negEmbeddings = negEmbeddings,
                            hiddenDim = hiddenDim,
                            effectiveGuidance = effectiveGuidance,
                            imageWidth = width,
                            imageHeight = height,
                            onSecondPassStart = {
                                val pass2Pct = 30 + (((executed - 0.3f) / activeSteps.toFloat()) * 56f).toInt()
                                onProgress(
                                    GenerationProgress(
                                        stageTitle = "Inpainting ${params.activeTab.shortTitleEn} (Step $executed/$activeSteps • 2/2)...",
                                        currentStep = executed,
                                        totalSteps = activeSteps,
                                        percent = pass2Pct,
                                        elapsedMs = System.currentTimeMillis() - startMs
                                    )
                                )
                            }
                        )
                        val editedNext = scheduler.step(finalNoise, stepIdx, latents)
                        val refAtNext = if (stepIdx + 1 < effectiveSteps) {
                            scheduler.addNoise(encodedCleanLatents, stepIdx + 1, seed)
                        } else {
                            encodedCleanLatents
                        }
                        latents = AvatarPhotoCompositor.blendLatentStep(editedNext, refAtNext, latentMask)
                    }
                    latents
                }
            }
            System.gc()

            onProgress(
                GenerationProgress(
                    stageTitle = "Decoding & Locking Original Identity (VAE Decoder)...",
                    currentStep = activeSteps,
                    totalSteps = activeSteps,
                    percent = 92,
                    elapsedMs = System.currentTimeMillis() - startMs
                )
            )
            checkNotCancelled()

            val scaledForVae = FloatArray(denoisedLatents.size) { i ->
                denoisedLatents[i] / 0.18215f
            }

            val rawGenerated = createSafeSession(env, vaeDecoderFile, inferenceMode).use { vaeSession ->
                runVaeDecoderToBitmap(
                    env = env,
                    vaeSession = vaeSession,
                    scaledLatents = scaledForVae,
                    latentH = latentH,
                    latentW = latentW,
                    outWidth = width,
                    outHeight = height
                )
            }
            System.gc()

            val reloadedOriginal = ImageBitmapPreprocessor.loadAndCenterCropFromUri(
                context = context,
                uri = Uri.fromFile(croppedSourceFile),
                targetWidth = width,
                targetHeight = height
            )
            val finalComposited = try {
                AvatarPhotoCompositor.compositeIdentityLockedPhoto(
                    original = reloadedOriginal,
                    generated = rawGenerated,
                    featheredMask = featheredMask
                )
            } finally {
                rawGenerated.recycle()
                reloadedOriginal.recycle()
            }
            System.gc()

            val totalMs = System.currentTimeMillis() - startMs
            onProgress(
                GenerationProgress(
                    stageTitle = "Completed!",
                    currentStep = activeSteps,
                    totalSteps = activeSteps,
                    percent = 100,
                    elapsedMs = totalMs
                )
            )

            AvatarCustomizationResult(
                bitmap = finalComposited,
                croppedSourceFile = croppedSourceFile,
                tab = params.activeTab,
                usedSlot7AgingModel = false,
                usedMediaPipeVision = usedMediaPipe,
                positivePrompt = builtPrompt,
                effectiveStrength = computedStrength,
                seed = seed,
                activeSteps = activeSteps,
                resolutionLabel = "$width × $height",
                durationMs = totalMs
            )
        }
    }

    private fun runVaeEncoder(
        env: OrtEnvironment,
        vaeEncSession: OrtSession,
        bitmap: Bitmap,
        latentSize: Int,
        seed: Long
    ): FloatArray {
        val inputName = vaeEncSession.inputNames.first()
        val inInfo = vaeEncSession.inputInfo[inputName]?.info as? TensorInfo
        val inType = inInfo?.type ?: OnnxJavaType.FLOAT
        val inShape = inInfo?.shape
        val w = bitmap.width
        val h = bitmap.height

        val isNhwc = inShape != null && inShape.size == 4 && inShape[3] == 3L
        val normalizedRgb = ImageBitmapPreprocessor.bitmapToNormalizedFloats(bitmap, isNhwc = isNhwc)
        val tensorShape = if (isNhwc) {
            longArrayOf(1L, h.toLong(), w.toLong(), 3L)
        } else {
            longArrayOf(1L, 3L, h.toLong(), w.toLong())
        }

        val inputTensor = createFloatOrFp16Tensor(
            env = env,
            data = normalizedRgb,
            shape = tensorShape,
            targetType = inType
        )

        inputTensor.use { tensor ->
            runInterruptible(vaeEncSession, mapOf(inputName to tensor)).use { results ->
                val outTensor = results[0] as OnnxTensor
                val rawOut = extractFloatArray(outTensor)

                return when {
                    // Format A: Optimum / Diffusers latent_dist.sample -> [1, 4, H/8, W/8]
                    rawOut.size == latentSize -> {
                        FloatArray(latentSize) { i -> rawOut[i] * 0.18215f }
                    }
                    // Format B: DiagonalGaussianDistribution moments -> [1, 8, H/8, W/8] (4 mean + 4 logvar)
                    rawOut.size >= latentSize * 2 -> {
                        val rng = java.util.Random(seed + 17L)
                        FloatArray(latentSize) { i ->
                            val mean = rawOut[i].toDouble()
                            val logVar = rawOut[latentSize + i].toDouble().coerceIn(-30.0, 20.0)
                            val std = kotlin.math.exp(0.5 * logVar)
                            val sampled = mean + std * rng.nextGaussian()
                            (sampled * 0.18215).toFloat()
                        }
                    }
                    else -> {
                        FloatArray(latentSize) { i -> rawOut.getOrElse(i) { 0f } * 0.18215f }
                    }
                }
            }
        }
    }

    private suspend fun checkNotCancelled() {
        currentCoroutineContext().ensureActive()
        if (cancelRequested.get()) {
            throw CancellationException("Generation cancelled by user.")
        }
    }

    private fun detectLikelyLcmModel(unetFile: File, unetSession: OrtSession?): Boolean {
        if (unetSession != null && unetSession.inputNames.any { it.contains("timestep_cond", ignoreCase = true) }) {
            return true
        }
        val sourceMeta = File(unetFile.parentFile, "source_name.meta")
            .takeIf { it.exists() }
            ?.readText()
            .orEmpty()
        val companionBytes = unetFile.parentFile?.listFiles()
            ?.filter {
                it.isFile &&
                    it.absolutePath != unetFile.absolutePath &&
                    !it.name.endsWith(".meta") &&
                    !it.name.endsWith(".repair_tmp")
            }
            ?.sumOf { it.length() } ?: 0L
        val totalUnetBytes = unetFile.length() + companionBytes
        val schedulerCfgFile = File(modelManager.modelsRootDir, "tokenizer/scheduler_config.json")
        val schedulerCfgText = schedulerCfgFile.takeIf { it.exists() }?.let {
            runCatching { it.readText() }.getOrNull()
        }
        return isLikelyLcmByMetadataAndSize(
            sourceMeta = sourceMeta,
            totalUnetBytes = totalUnetBytes,
            schedulerConfigText = schedulerCfgText
        )
    }

    private fun requireCompleteActiveEngineFile(activeEngine: ActiveDiffusionEngine): File {
        val engineDir = modelManager.engineFolder(activeEngine)
        val linkState = OnnxModelManager.linkEnginePairFiles(engineDir)
        val otherEngine = if (activeEngine == ActiveDiffusionEngine.FAST_LCM) {
            ActiveDiffusionEngine.STANDARD_SD
        } else {
            ActiveDiffusionEngine.FAST_LCM
        }
        val otherReady = OnnxModelManager.linkEnginePairFiles(modelManager.engineFolder(otherEngine)).isPairLinked
        val switchHint = if (otherReady) {
            val otherTitle = if (otherEngine == ActiveDiffusionEngine.FAST_LCM) "⚡ Fast LCM" else "🎨 Standard SD"
            " (அல்லது மேலே உள்ள சுவிட்சில் $otherTitle தேர்வு செய்யவும் / Or switch to $otherTitle at the top!)"
        } else {
            ""
        }

        val file = modelManager.resolveEngineUnetFile(activeEngine)
        if (file == null || !linkState.isGraphInstalled) {
            val slotGraphLabel = if (activeEngine == ActiveDiffusionEngine.FAST_LCM) {
                "Slot 1A (Fast LCM model.onnx)"
            } else {
                "Slot 2A (Standard SD model.onnx)"
            }
            error("Missing $slotGraphLabel. Please upload it in the Models tab.$switchHint")
        }

        val missingCompanions = OnnxModelManager.findMissingCompanionFiles(file)
        if (missingCompanions.isNotEmpty() || !linkState.isPairLinked) {
            val slotWeightLabel = if (activeEngine == ActiveDiffusionEngine.FAST_LCM) {
                "Slot 1B (${missingCompanions.firstOrNull() ?: "model_int8.onnx.data"})"
            } else {
                "Slot 2B (${missingCompanions.firstOrNull() ?: "weights.pb"})"
            }
            error(
                "Missing $slotWeightLabel for ${if (activeEngine == ActiveDiffusionEngine.FAST_LCM) "Fast LCM" else "Standard SD"}. Please upload $slotWeightLabel in the Models tab.$switchHint"
            )
        }
        return file
    }

    private fun requireCompleteSlotFile(slot: DiffusionModelSlot, slotDisplayLabel: String = slot.canonicalFileName): File {
        val file = modelManager.resolveExistingSlotFile(slot)
            ?: error("Missing $slotDisplayLabel. Please upload it in the Models tab.")
        val missingCompanions = OnnxModelManager.findMissingCompanionFiles(file)
        if (missingCompanions.isNotEmpty()) {
            error(
                "$slotDisplayLabel (${OnnxModelManager.formatBytes(file.length())}) is incomplete — missing external weights: ${missingCompanions.joinToString()}."
            )
        }
        return file
    }

    private fun createSafeSession(
        env: OrtEnvironment,
        modelFile: File,
        inferenceMode: String,
        onRepairStatus: ((String) -> Unit)? = null
    ): OrtSession {
        OnnxProtoCodec.inspectModelFileHeader(modelFile)?.let { badHeader ->
            error("${modelFile.parentFile?.name ?: modelFile.name}: $badHeader")
        }
        val isFastLcmSlot = modelFile.parentFile?.name == DedicatedModelSlot.FAST_LCM_GRAPH.folderName
        if (isFastLcmSlot && OnnxQuantGraphRepairer.detectLegacyV1CorruptedGraph(modelFile)) {
            if (!modelManager.restoreCorruptedFastLcmGraphIfPossible(onRepairStatus)) {
                error(
                    "Slot 1A (unet_fast_lcm/model.onnx ~3.05 MB) needs a one-time refresh: tap 'Upload Slot 1A' and re-select model.onnx (keep Slot 1B 860 MB & all other slots untouched)."
                )
            }
        }

        val repairMode = OnnxQuantGraphRepairer.ensureCompatibleBeforeSession(modelFile, onRepairStatus)
        return try {
            openConfiguredSession(env, modelFile, inferenceMode, repairMode)
        } catch (firstError: Throwable) {
            val msg = firstError.message.orEmpty()
            if (isFastLcmSlot && msg.contains("INVALID_PROTOBUF", ignoreCase = true)) {
                if (modelManager.restoreCorruptedFastLcmGraphIfPossible(onRepairStatus)) {
                    val healedMode = OnnxQuantGraphRepairer.ensureCompatibleBeforeSession(modelFile, onRepairStatus)
                    return openConfiguredSession(env, modelFile, inferenceMode, healedMode)
                }
            }
            val needsQdqFallback = repairMode != ConvRepairMode.TIER2_QDQ_CONV &&
                (msg.contains("ConvInteger", ignoreCase = true) ||
                    msg.contains("NOT_IMPLEMENTED", ignoreCase = true))
            if (needsQdqFallback && OnnxQuantGraphRepairer.upgradeToQdqConv(modelFile, onRepairStatus)) {
                openConfiguredSession(
                    env = env,
                    modelFile = modelFile,
                    inferenceMode = inferenceMode,
                    repairMode = ConvRepairMode.TIER2_QDQ_CONV
                )
            } else {
                throw firstError
            }
        }
    }

    private fun openConfiguredSession(
        env: OrtEnvironment,
        modelFile: File,
        inferenceMode: String,
        repairMode: ConvRepairMode
    ): OrtSession {
        val preferNnapi = inferenceMode.contains("NNAPI", ignoreCase = true)
        val lowRamMode = inferenceMode.contains("Low-RAM", ignoreCase = true)
        val companionBytes = modelFile.parentFile?.listFiles()
            ?.filter {
                it.isFile &&
                    it.absolutePath != modelFile.absolutePath &&
                    !it.name.endsWith(".meta") &&
                    !it.name.endsWith(".repair_tmp")
            }
            ?.sumOf { it.length() } ?: 0L
        val totalBytes = modelFile.length() + companionBytes
        val isLargeExternalModel = totalBytes > 450L * 1024L * 1024L
        val isUltraLargeModel = totalBytes > 1_500L * 1024L * 1024L
        val isStreamingQdqLargeModel = repairMode == ConvRepairMode.TIER2_QDQ_CONV && isLargeExternalModel
        val perfThreads = computePerformanceThreadCount(Runtime.getRuntime().availableProcessors())

        fun applyBaseTuning(opts: OrtSession.SessionOptions, optLevel: OrtSession.SessionOptions.OptLevel) {
            opts.setOptimizationLevel(optLevel)
            opts.setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL)
            opts.setInterOpNumThreads(1)
            opts.setIntraOpNumThreads(perfThreads)
            runCatching { opts.addConfigEntry("session.intra_op.allow_spinning", "0") }
            if (isLargeExternalModel || lowRamMode || isStreamingQdqLargeModel) {
                runCatching { opts.addConfigEntry("session.disable_prepacking", "1") }
                opts.setCPUArenaAllocator(false)
                opts.setMemoryPatternOptimization(false)
            }
        }

        // For large Tier-2 QDQ rewritten models, use NO_OPT directly so ConstantFolding does not
        // pre-expand ~595 MB of 8-bit Conv weights into ~2.38 GB of FP32 weights in RAM.
        if (isStreamingQdqLargeModel) {
            OrtSession.SessionOptions().use { opts ->
                applyBaseTuning(opts, OrtSession.SessionOptions.OptLevel.NO_OPT)
                return env.createSession(modelFile.absolutePath, opts)
            }
        }

        if (preferNnapi && !isLargeExternalModel) {
            runCatching {
                OrtSession.SessionOptions().use { opts ->
                    applyBaseTuning(
                        opts,
                        if (lowRamMode) OrtSession.SessionOptions.OptLevel.BASIC_OPT
                        else OrtSession.SessionOptions.OptLevel.ALL_OPT
                    )
                    opts.addNnapi(EnumSet.noneOf(ai.onnxruntime.providers.NNAPIFlags::class.java))
                    return env.createSession(modelFile.absolutePath, opts)
                }
            }
        }

        // Fast ARM64 NEON/FP16 XNNPACK acceleration for self-contained Text Encoder & VAE models (< 450 MB)
        if (!isLargeExternalModel && !lowRamMode) {
            runCatching {
                OrtSession.SessionOptions().use { opts ->
                    applyBaseTuning(opts, OrtSession.SessionOptions.OptLevel.ALL_OPT)
                    opts.addXnnpack(mapOf("intra_op_num_threads" to perfThreads.toString()))
                    return env.createSession(modelFile.absolutePath, opts)
                }
            }
        }

        // Primary CPU Execution Provider path: ALL_OPT + disable_prepacking for ~860 MB INT8 LCM UNet,
        // BASIC_OPT + disable_prepacking for ultra-large >1.5 GB FP32 weights.pb
        val targetOptLevel = when {
            isUltraLargeModel || lowRamMode -> OrtSession.SessionOptions.OptLevel.BASIC_OPT
            else -> OrtSession.SessionOptions.OptLevel.ALL_OPT
        }
        val primaryError: Throwable = try {
            OrtSession.SessionOptions().use { opts ->
                applyBaseTuning(opts, targetOptLevel)
                return env.createSession(modelFile.absolutePath, opts)
            }
        } catch (t: Throwable) {
            val errMsg = t.message.orEmpty()
            if (errMsg.contains("ConvInteger", ignoreCase = true) ||
                errMsg.contains("NOT_IMPLEMENTED", ignoreCase = true)
            ) {
                throw t
            }
            t
        }

        // Safety fallback if a custom graph fusion pass fails (preserves primaryError if NO_OPT also fails)
        return try {
            OrtSession.SessionOptions().use { opts ->
                applyBaseTuning(opts, OrtSession.SessionOptions.OptLevel.NO_OPT)
                env.createSession(modelFile.absolutePath, opts)
            }
        } catch (_: Throwable) {
            throw primaryError
        }
    }

    private fun runTextEncoder(
        env: OrtEnvironment,
        session: OrtSession,
        tokenIds: IntArray
    ): Pair<FloatArray, Int> {
        val inputName = session.inputNames.firstOrNull { it.contains("input_ids", ignoreCase = true) }
            ?: session.inputNames.first()
        val info = session.inputInfo[inputName]?.info as? TensorInfo
        val seqLen = tokenIds.size.toLong()
        val shape = longArrayOf(1L, seqLen)

        val inputTensor = if (info?.type == OnnxJavaType.INT64) {
            val longs = LongArray(tokenIds.size) { tokenIds[it].toLong() }
            OnnxTensor.createTensor(env, LongBuffer.wrap(longs), shape)
        } else {
            OnnxTensor.createTensor(env, IntBuffer.wrap(tokenIds), shape)
        }

        inputTensor.use { tensor ->
            runInterruptible(session, mapOf(inputName to tensor)).use { result ->
                val outTensor = result[0] as OnnxTensor
                val floats = extractFloatArray(outTensor)
                val outShape = (outTensor.info as? TensorInfo)?.shape
                val hiddenDim = if (outShape != null && outShape.size >= 3) {
                    outShape[2].toInt()
                } else {
                    (floats.size / 77).coerceAtLeast(1)
                }
                return floats to hiddenDim
            }
        }
    }

    private fun unetRequiresStaticBatchTwo(unetSession: OrtSession): Boolean {
        for ((name, nodeInfo) in unetSession.inputInfo) {
            val ti = nodeInfo.info as? TensorInfo ?: continue
            val lower = name.lowercase(Locale.US)
            if ((lower == "sample" || lower.contains("latent")) && ti.shape.size >= 4 && ti.shape[0] == 2L) {
                return true
            }
        }
        return false
    }

    private suspend fun runEulerNoisePrediction(
        env: OrtEnvironment,
        unetSession: OrtSession,
        scaledLatents: FloatArray,
        latentH: Int,
        latentW: Int,
        timestep: Int,
        posEmbeddings: FloatArray,
        negEmbeddings: FloatArray,
        hiddenDim: Int,
        effectiveGuidance: Float,
        imageWidth: Int,
        imageHeight: Int,
        onSecondPassStart: (() -> Unit)? = null
    ): FloatArray {
        if (effectiveGuidance > 1.15f && unetRequiresStaticBatchTwo(unetSession)) {
            return runUnetBatchedCfgTwo(
                env = env,
                unetSession = unetSession,
                latents = scaledLatents,
                latentH = latentH,
                latentW = latentW,
                timestep = timestep,
                uncondHiddenStates = negEmbeddings,
                condHiddenStates = posEmbeddings,
                hiddenDim = hiddenDim,
                guidanceScale = effectiveGuidance,
                imageWidth = imageWidth,
                imageHeight = imageHeight
            )
        }

        val condNoise = runUnetSingleBatch(
            env = env,
            unetSession = unetSession,
            latents = scaledLatents,
            latentH = latentH,
            latentW = latentW,
            timestep = timestep,
            encoderHiddenStates = posEmbeddings,
            hiddenDim = hiddenDim,
            guidanceScale = effectiveGuidance,
            imageWidth = imageWidth,
            imageHeight = imageHeight
        )
        return if (effectiveGuidance > 1.15f) {
            checkNotCancelled()
            onSecondPassStart?.invoke()
            val uncondNoise = runUnetSingleBatch(
                env = env,
                unetSession = unetSession,
                latents = scaledLatents,
                latentH = latentH,
                latentW = latentW,
                timestep = timestep,
                encoderHiddenStates = negEmbeddings,
                hiddenDim = hiddenDim,
                guidanceScale = effectiveGuidance,
                imageWidth = imageWidth,
                imageHeight = imageHeight
            )
            FloatArray(condNoise.size) { i ->
                uncondNoise[i] + effectiveGuidance * (condNoise[i] - uncondNoise[i])
            }
        } else {
            condNoise
        }
    }

    private fun runUnetBatchedCfgTwo(
        env: OrtEnvironment,
        unetSession: OrtSession,
        latents: FloatArray,
        latentH: Int,
        latentW: Int,
        timestep: Int,
        uncondHiddenStates: FloatArray,
        condHiddenStates: FloatArray,
        hiddenDim: Int,
        guidanceScale: Float,
        imageWidth: Int,
        imageHeight: Int
    ): FloatArray {
        val latentSize = 4 * latentH * latentW
        val createdTensors = mutableListOf<OnnxTensor>()
        try {
            val inputs = LinkedHashMap<String, OnnxTensor>()
            for ((name, nodeInfo) in unetSession.inputInfo) {
                val ti = nodeInfo.info as? TensorInfo ?: continue
                val lower = name.lowercase(Locale.US)
                val tensor = when {
                    lower == "sample" || lower.contains("latent") -> {
                        val doubled = FloatArray(latentSize * 2)
                        System.arraycopy(latents, 0, doubled, 0, latentSize)
                        System.arraycopy(latents, 0, doubled, latentSize, latentSize)
                        createFloatOrFp16Tensor(
                            env = env,
                            data = doubled,
                            shape = longArrayOf(2L, 4L, latentH.toLong(), latentW.toLong()),
                            targetType = ti.type
                        )
                    }
                    lower == "timestep" || lower == "t" -> {
                        createTimestepTensor(env, timestep, ti, batchSize = 2)
                    }
                    lower.contains("encoder_hidden_states") || lower.contains("context") -> {
                        val expectedDim = if (ti.shape.size >= 3 && ti.shape[2] > 0) ti.shape[2].toInt() else hiddenDim
                        val uncondAdapted = adaptEmbeddingDimension(uncondHiddenStates, 77, expectedDim)
                        val condAdapted = adaptEmbeddingDimension(condHiddenStates, 77, expectedDim)
                        val packed = FloatArray(uncondAdapted.size + condAdapted.size)
                        System.arraycopy(uncondAdapted, 0, packed, 0, uncondAdapted.size)
                        System.arraycopy(condAdapted, 0, packed, uncondAdapted.size, condAdapted.size)
                        createFloatOrFp16Tensor(
                            env = env,
                            data = packed,
                            shape = longArrayOf(2L, 77L, expectedDim.toLong()),
                            targetType = ti.type
                        )
                    }
                    else -> {
                        val concreteShape = ti.shape.mapIndexed { idx, d ->
                            if (d <= 0L) (if (idx == 0) 2L else 1L) else d
                        }.toLongArray()
                        val elemCount = concreteShape.fold(1L) { a, b -> a * b }.toInt().coerceAtLeast(1)
                        createFloatOrFp16Tensor(
                            env = env,
                            data = FloatArray(elemCount),
                            shape = concreteShape,
                            targetType = ti.type
                        )
                    }
                }
                createdTensors.add(tensor)
                inputs[name] = tensor
            }

            runInterruptible(unetSession, inputs).use { results ->
                val outTensor = results[0] as OnnxTensor
                val raw = extractFloatArray(outTensor)
                return if (raw.size >= latentSize * 2) {
                    FloatArray(latentSize) { i ->
                        val uncond = raw[i]
                        val cond = raw[latentSize + i]
                        uncond + guidanceScale * (cond - uncond)
                    }
                } else {
                    raw.copyOfRange(0, minOf(raw.size, latentSize))
                }
            }
        } finally {
            createdTensors.forEach { runCatching { it.close() } }
        }
    }

    private fun runUnetSingleBatch(
        env: OrtEnvironment,
        unetSession: OrtSession,
        latents: FloatArray,
        latentH: Int,
        latentW: Int,
        timestep: Int,
        encoderHiddenStates: FloatArray,
        hiddenDim: Int,
        guidanceScale: Float,
        imageWidth: Int,
        imageHeight: Int
    ): FloatArray {
        val latentSize = 4 * latentH * latentW
        val forceBatchTwo = unetRequiresStaticBatchTwo(unetSession)
        val batchCount = if (forceBatchTwo) 2 else 1
        val createdTensors = mutableListOf<OnnxTensor>()
        try {
            val inputs = LinkedHashMap<String, OnnxTensor>()
            for ((name, nodeInfo) in unetSession.inputInfo) {
                val ti = nodeInfo.info as? TensorInfo ?: continue
                val lower = name.lowercase(Locale.US)
                val tensor = when {
                    lower == "sample" || lower.contains("latent") -> {
                        val data = if (batchCount == 2) {
                            FloatArray(latentSize * 2).also {
                                System.arraycopy(latents, 0, it, 0, latentSize)
                                System.arraycopy(latents, 0, it, latentSize, latentSize)
                            }
                        } else {
                            latents
                        }
                        createFloatOrFp16Tensor(
                            env = env,
                            data = data,
                            shape = longArrayOf(batchCount.toLong(), 4L, latentH.toLong(), latentW.toLong()),
                            targetType = ti.type
                        )
                    }
                    lower == "timestep" || lower == "t" -> {
                        createTimestepTensor(env, timestep, ti, batchSize = batchCount)
                    }
                    lower.contains("encoder_hidden_states") || lower.contains("context") -> {
                        val expectedDim = if (ti.shape.size >= 3 && ti.shape[2] > 0) {
                            ti.shape[2].toInt()
                        } else {
                            hiddenDim
                        }
                        val adaptedHidden = adaptEmbeddingDimension(encoderHiddenStates, 77, expectedDim)
                        val data = if (batchCount == 2) {
                            FloatArray(adaptedHidden.size * 2).also {
                                System.arraycopy(adaptedHidden, 0, it, 0, adaptedHidden.size)
                                System.arraycopy(adaptedHidden, 0, it, adaptedHidden.size, adaptedHidden.size)
                            }
                        } else {
                            adaptedHidden
                        }
                        createFloatOrFp16Tensor(
                            env = env,
                            data = data,
                            shape = longArrayOf(batchCount.toLong(), 77L, expectedDim.toLong()),
                            targetType = ti.type
                        )
                    }
                    lower.contains("timestep_cond") -> {
                        val condDim = if (ti.shape.size >= 2 && ti.shape[1] > 0) ti.shape[1].toInt() else 256
                        val wEmbed = DiffusionSchedulers.buildGuidanceScaleEmbedding(
                            w = (guidanceScale - 1.0f).coerceAtLeast(0f),
                            embeddingDim = condDim
                        )
                        val data = if (batchCount == 2) {
                            FloatArray(wEmbed.size * 2).also {
                                System.arraycopy(wEmbed, 0, it, 0, wEmbed.size)
                                System.arraycopy(wEmbed, 0, it, wEmbed.size, wEmbed.size)
                            }
                        } else {
                            wEmbed
                        }
                        createFloatOrFp16Tensor(
                            env = env,
                            data = data,
                            shape = longArrayOf(batchCount.toLong(), condDim.toLong()),
                            targetType = ti.type
                        )
                    }
                    lower.contains("time_ids") -> {
                        val singleTimeIds = floatArrayOf(
                            imageHeight.toFloat(), imageWidth.toFloat(),
                            0f, 0f,
                            imageHeight.toFloat(), imageWidth.toFloat()
                        )
                        val data = if (batchCount == 2) singleTimeIds + singleTimeIds else singleTimeIds
                        createFloatOrFp16Tensor(
                            env = env,
                            data = data,
                            shape = longArrayOf(batchCount.toLong(), 6L),
                            targetType = ti.type
                        )
                    }
                    else -> {
                        val concreteShape = ti.shape.mapIndexed { idx, d ->
                            if (d <= 0L) (if (idx == 0) batchCount.toLong() else 1L) else d
                        }.toLongArray()
                        val elemCount = concreteShape.fold(1L) { a, b -> a * b }.toInt().coerceAtLeast(1)
                        createFloatOrFp16Tensor(
                            env = env,
                            data = FloatArray(elemCount),
                            shape = concreteShape,
                            targetType = ti.type
                        )
                    }
                }
                createdTensors.add(tensor)
                inputs[name] = tensor
            }

            runInterruptible(unetSession, inputs).use { results ->
                val outTensor = results[0] as OnnxTensor
                val raw = extractFloatArray(outTensor)
                return if (batchCount == 2 && raw.size >= latentSize) {
                    raw.copyOfRange(0, latentSize)
                } else {
                    raw
                }
            }
        } finally {
            createdTensors.forEach { runCatching { it.close() } }
        }
    }

    private fun runVaeDecoderToBitmap(
        env: OrtEnvironment,
        vaeSession: OrtSession,
        scaledLatents: FloatArray,
        latentH: Int,
        latentW: Int,
        outWidth: Int,
        outHeight: Int
    ): Bitmap {
        val inputName = vaeSession.inputNames.first()
        val inInfo = vaeSession.inputInfo[inputName]?.info as? TensorInfo
        val inType = inInfo?.type ?: OnnxJavaType.FLOAT

        val inputTensor = createFloatOrFp16Tensor(
            env = env,
            data = scaledLatents,
            shape = longArrayOf(1L, 4L, latentH.toLong(), latentW.toLong()),
            targetType = inType
        )

        inputTensor.use { tensor ->
            runInterruptible(vaeSession, mapOf(inputName to tensor)).use { results ->
                val outTensor = results[0] as OnnxTensor
                val outShape = (outTensor.info as? TensorInfo)?.shape
                val rgbFloats = extractFloatArray(outTensor)

                val isNhwc = outShape != null && outShape.size == 4 && outShape[3] == 3L
                val pixelCount = outWidth * outHeight
                val pixels = IntArray(pixelCount)

                if (isNhwc) {
                    for (i in 0 until pixelCount) {
                        val base = i * 3
                        val r = floatToByte(rgbFloats.getOrElse(base) { 0f })
                        val g = floatToByte(rgbFloats.getOrElse(base + 1) { 0f })
                        val b = floatToByte(rgbFloats.getOrElse(base + 2) { 0f })
                        pixels[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
                    }
                } else {
                    // Standard NCHW: [1, 3, H, W]
                    val channelStride = pixelCount
                    for (i in 0 until pixelCount) {
                        val r = floatToByte(rgbFloats.getOrElse(i) { 0f })
                        val g = floatToByte(rgbFloats.getOrElse(channelStride + i) { 0f })
                        val b = floatToByte(rgbFloats.getOrElse(2 * channelStride + i) { 0f })
                        pixels[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
                    }
                }

                return Bitmap.createBitmap(pixels, outWidth, outHeight, Bitmap.Config.ARGB_8888)
            }
        }
    }

    private fun floatToByte(v: Float): Int =
        (((v / 2.0f) + 0.5f).coerceIn(0.0f, 1.0f) * 255.0f + 0.5f).toInt().coerceIn(0, 255)

    private fun adaptEmbeddingDimension(src: FloatArray, seqLen: Int, expectedDim: Int): FloatArray {
        val targetSize = seqLen * expectedDim
        if (src.size == targetSize) return src
        val actualDim = (src.size / seqLen).coerceAtLeast(1)
        val out = FloatArray(targetSize)
        val copyDim = minOf(actualDim, expectedDim)
        for (tok in 0 until seqLen) {
            System.arraycopy(src, tok * actualDim, out, tok * expectedDim, copyDim)
        }
        return out
    }

    private fun createTimestepTensor(
        env: OrtEnvironment,
        timestep: Int,
        info: TensorInfo,
        batchSize: Int = 1
    ): OnnxTensor {
        val isScalar = info.shape.isEmpty()
        val count = if (!isScalar && batchSize == 2 && info.shape.firstOrNull() == 2L) 2 else 1
        val shape = if (isScalar) longArrayOf() else longArrayOf(count.toLong())
        return when (info.type) {
            OnnxJavaType.INT64 -> {
                val arr = LongArray(count) { timestep.toLong() }
                OnnxTensor.createTensor(env, LongBuffer.wrap(arr), shape)
            }
            OnnxJavaType.INT32 -> {
                val arr = IntArray(count) { timestep }
                OnnxTensor.createTensor(env, IntBuffer.wrap(arr), shape)
            }
            OnnxJavaType.FLOAT16 -> {
                val half = floatToHalfBits(timestep.toFloat())
                val arr = ShortArray(count) { half }
                OnnxTensor.createTensor(env, ShortBuffer.wrap(arr), shape, OnnxJavaType.FLOAT16)
            }
            else -> {
                val arr = FloatArray(count) { timestep.toFloat() }
                OnnxTensor.createTensor(env, FloatBuffer.wrap(arr), shape)
            }
        }
    }

    private fun createFloatOrFp16Tensor(
        env: OrtEnvironment,
        data: FloatArray,
        shape: LongArray,
        targetType: OnnxJavaType
    ): OnnxTensor {
        return if (targetType == OnnxJavaType.FLOAT16) {
            val shorts = ShortArray(data.size) { idx -> floatToHalfBits(data[idx]) }
            OnnxTensor.createTensor(env, ShortBuffer.wrap(shorts), shape, OnnxJavaType.FLOAT16)
        } else {
            OnnxTensor.createTensor(env, FloatBuffer.wrap(data), shape)
        }
    }

    private fun extractFloatArray(tensor: OnnxTensor): FloatArray {
        val ti = tensor.info as? TensorInfo
        return if (ti?.type == OnnxJavaType.FLOAT16) {
            val sb = tensor.shortBuffer
            val out = FloatArray(sb.remaining())
            var i = 0
            while (sb.hasRemaining()) {
                out[i++] = halfBitsToFloat(sb.get())
            }
            out
        } else {
            val fb = tensor.floatBuffer
            val out = FloatArray(fb.remaining())
            fb.get(out)
            out
        }
    }

    private fun parseResolution(label: String): Pair<Int, Int> {
        val nums = Regex("\\d+").findAll(label).mapNotNull { it.value.toIntOrNull() }.toList()
        if (nums.size >= 2) {
            val w = (nums[0] / 8) * 8
            val h = (nums[1] / 8) * 8
            return w.coerceIn(256, 768) to h.coerceIn(256, 768)
        }
        return 512 to 512
    }

    companion object {
        /**
         * Matches optimal performance core count on mobile big.LITTLE SoCs (e.g., 4x Cortex-A78 on
         * 8-core Snapdragon 6 Gen 1) so intra-op barriers never wait on slower Cortex-A55 cores.
         */
        fun computePerformanceThreadCount(availableProcessors: Int): Int = when {
            availableProcessors >= 6 -> 4
            availableProcessors >= 4 -> 4
            else -> availableProcessors.coerceAtLeast(2)
        }

        fun isLikelyLcmByMetadataAndSize(
            sourceMeta: String,
            totalUnetBytes: Long,
            schedulerConfigText: String?
        ): Boolean {
            if (sourceMeta.contains("lcm", ignoreCase = true) ||
                sourceMeta.contains("int8", ignoreCase = true) ||
                sourceMeta.contains("turbo", ignoreCase = true) ||
                sourceMeta.contains("hyper", ignoreCase = true)
            ) {
                return true
            }
            // Full >2.3 GB FP32 SD 1.5 export without LCM metadata is standard SD 1.5
            if (totalUnetBytes > 2_300_000_000L) {
                return false
            }
            // Compact ~860 MB INT8 mobile UNet (e.g. model.onnx + model.onnx.data) is LCM-capable
            // even if a leftover SD 1.5 PNDMScheduler config was present in the same folder
            if (totalUnetBytes in 200_000_000L..1_450_000_000L) {
                return true
            }
            if (!schedulerConfigText.isNullOrBlank()) {
                if (schedulerConfigText.contains("LCMScheduler", ignoreCase = true)) return true
                if (schedulerConfigText.contains("PNDMScheduler", ignoreCase = true) ||
                    schedulerConfigText.contains("Euler", ignoreCase = true) ||
                    schedulerConfigText.contains("DDIM", ignoreCase = true) ||
                    schedulerConfigText.contains("LMSDiscrete", ignoreCase = true) ||
                    schedulerConfigText.contains("DPMSolver", ignoreCase = true)
                ) {
                    return false
                }
            }
            return true
        }

        // Pure IEEE-754 Float32 <-> Float16 conversion (works across all Android APIs & JVM tests)
        fun floatToHalfBits(fval: Float): Short {
            val fbits = java.lang.Float.floatToIntBits(fval)
            val sign = (fbits ushr 16) and 0x8000
            var valBits = (fbits and 0x7fffffff) + 0x1000
            if (valBits >= 0x47800000) {
                if ((fbits and 0x7fffffff) >= 0x47800000) {
                    if (valBits < 0x7f800000) return (sign or 0x7c00).toShort()
                    return (sign or 0x7c00 or ((fbits and 0x007fffff) ushr 13)).toShort()
                }
                return (sign or 0x7bff).toShort()
            }
            if (valBits >= 0x38800000) {
                return (sign or ((valBits - 0x38000000) ushr 13)).toShort()
            }
            if (valBits < 0x33000000) return sign.toShort()
            valBits = (fbits and 0x7fffffff) ushr 23
            return (sign or ((((fbits and 0x7fffff) or 0x800000) + (0x800000 ushr (valBits - 102))) ushr (126 - valBits))).toShort()
        }

        fun halfBitsToFloat(hbitsShort: Short): Float {
            val hbits = hbitsShort.toInt() and 0xFFFF
            var mant = hbits and 0x03ff
            var exp = hbits and 0x7c00
            if (exp == 0x7c00) {
                exp = 0x3fc00
            } else if (exp != 0) {
                exp += 0x1c000
            } else if (mant != 0) {
                exp = 0x1c400
                do {
                    mant = mant shl 1
                    exp -= 0x400
                } while ((mant and 0x400) == 0)
                mant = mant and 0x3ff
            }
            return java.lang.Float.intBitsToFloat(
                ((hbits and 0x8000) shl 16) or ((exp or mant) shl 13)
            )
        }
    }
}
