package com.example.ai

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import android.content.Context
import android.graphics.Bitmap
import android.net.Uri
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
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

class OnnxDiffusionEngine(
    private val context: Context,
    private val modelManager: OnnxModelManager
) {
    private val tokenizer = ClipTokenizer(context)
    private val cancelRequested = AtomicBoolean(false)

    fun requestCancel() {
        cancelRequested.set(true)
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
        for (slot in DiffusionModelSlot.entries) {
            currentCoroutineContext().ensureActive()
            val file = modelManager.resolveExistingSlotFile(slot)
            if (file == null) {
                lines.add("• ${slot.canonicalFileName}: Not uploaded yet")
                continue
            }
            val missingCompanions = OnnxModelManager.findMissingCompanionFiles(file)
            if (missingCompanions.isNotEmpty()) {
                lines.add(
                    "⚠️ ${slot.canonicalFileName} (${OnnxModelManager.formatBytes(file.length())}) -> Incomplete: missing ${missingCompanions.joinToString()} (See Settings → ONNX Model Download Links)"
                )
                incompleteCount++
                continue
            }
            val t0 = System.currentTimeMillis()
            val session = createSafeSession(env, file, inferenceMode)
            session.use { s ->
                val dt = System.currentTimeMillis() - t0
                val inputsDesc = s.inputInfo.entries.joinToString { (name, node) ->
                    val ti = node.info as? TensorInfo
                    "$name:${ti?.type ?: "?"}${ti?.shape?.contentToString() ?: ""}"
                }
                lines.add("✅ ${slot.canonicalFileName} (${dt}ms) -> $inputsDesc")
                verifiedCount++
            }
        }
        if (verifiedCount == 0 && incompleteCount == 0) {
            error("No .onnx files installed yet. Open Models tab and use One-Touch Folder Upload.")
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
        onProgress: (GenerationProgress) -> Unit
    ): Result<TextToImageResult> = runCatching {
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

        // 2. Verify required 3 Text-to-Image ONNX models & any external weights exist
        val textEncoderFile = requireCompleteSlotFile(DiffusionModelSlot.TEXT_ENCODER)
        val unetFile = requireCompleteSlotFile(DiffusionModelSlot.UNET)
        val vaeDecoderFile = requireCompleteSlotFile(DiffusionModelSlot.VAE_DECODER)

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
        val denoisedLatents = createSafeSession(env, unetFile, inferenceMode).use { unetSession ->
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
                    val pct = 16 + (((stepIdx + 1).toFloat() / effectiveSteps.toFloat()) * 68f).toInt()
                    onProgress(
                        GenerationProgress(
                            stageTitle = "Generating (LCM Step ${stepIdx + 1}/$effectiveSteps)...",
                            currentStep = stepIdx + 1,
                            totalSteps = effectiveSteps,
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
                var latents = DiffusionSchedulers.createRandomLatents(
                    size = latentSize,
                    seed = seed,
                    initSigma = scheduler.initNoiseSigma
                )

                for (stepIdx in 0 until effectiveSteps) {
                    checkNotCancelled()
                    val t = scheduler.timesteps[stepIdx]
                    val pct = 16 + (((stepIdx + 1).toFloat() / effectiveSteps.toFloat()) * 68f).toInt()
                    onProgress(
                        GenerationProgress(
                            stageTitle = "Denoising (Step ${stepIdx + 1}/$effectiveSteps)...",
                            currentStep = stepIdx + 1,
                            totalSteps = effectiveSteps,
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
                        imageHeight = height
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
        onProgress: (GenerationProgress) -> Unit
    ): Result<ImageToImageResult> = runCatching {
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

        // 2. Verify all 4 ONNX models & external weights exist (including vae_encoder.onnx)
        val vaeEncoderFile = requireCompleteSlotFile(DiffusionModelSlot.VAE_ENCODER)
        val textEncoderFile = requireCompleteSlotFile(DiffusionModelSlot.TEXT_ENCODER)
        val unetFile = requireCompleteSlotFile(DiffusionModelSlot.UNET)
        val vaeDecoderFile = requireCompleteSlotFile(DiffusionModelSlot.VAE_DECODER)

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

        val denoisedLatents = createSafeSession(env, unetFile, inferenceMode).use { unetSession ->
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

                for (stepIdx in startStepIndex until effectiveSteps) {
                    checkNotCancelled()
                    executed++
                    val t = scheduler.timesteps[stepIdx]
                    val pct = 24 + ((executed.toFloat() / activeSteps.toFloat()) * 64f).toInt()
                    onProgress(
                        GenerationProgress(
                            stageTitle = "Transforming (Step $executed/$activeSteps)...",
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
                        imageHeight = height
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
            vaeEncSession.run(mapOf(inputName to tensor)).use { results ->
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
        if (sourceMeta.contains("lcm", ignoreCase = true)) {
            return true
        }
        val schedulerCfgFile = File(modelManager.modelsRootDir, "tokenizer/scheduler_config.json")
        if (schedulerCfgFile.exists()) {
            val cfg = runCatching { schedulerCfgFile.readText() }.getOrDefault("")
            if (cfg.contains("LCMScheduler", ignoreCase = true)) return true
            if (cfg.contains("PNDMScheduler", ignoreCase = true) ||
                cfg.contains("Euler", ignoreCase = true) ||
                cfg.contains("DDIM", ignoreCase = true) ||
                cfg.contains("LMSDiscrete", ignoreCase = true) ||
                cfg.contains("DPMSolver", ignoreCase = true)
            ) {
                return false
            }
        }
        val companionBytes = unetFile.parentFile?.listFiles()
            ?.filter { it.isFile && it.absolutePath != unetFile.absolutePath && !it.name.endsWith(".meta") }
            ?.sumOf { it.length() } ?: 0L
        val totalUnetBytes = unetFile.length() + companionBytes
        // Full 3.44 GB FP32 SD 1.5 export without timestep_cond or LCM metadata is standard SD 1.5
        if (totalUnetBytes > 2_500_000_000L) {
            return false
        }
        return true
    }

    private fun requireCompleteSlotFile(slot: DiffusionModelSlot): File {
        val file = modelManager.resolveExistingSlotFile(slot)
            ?: error("Missing ${slot.canonicalFileName}. Please upload your ONNX folder in the Models tab.")
        val missingCompanions = OnnxModelManager.findMissingCompanionFiles(file)
        if (missingCompanions.isNotEmpty()) {
            error(
                "${slot.canonicalFileName} (${OnnxModelManager.formatBytes(file.length())}) is incomplete — missing external weights: ${missingCompanions.joinToString()}. Download the complete ONNX model pair from Settings → ONNX Model Download Links, or attach ${missingCompanions.first()} in the Models tab."
            )
        }
        return file
    }

    private fun createSafeSession(
        env: OrtEnvironment,
        modelFile: File,
        inferenceMode: String
    ): OrtSession {
        val preferNnapi = inferenceMode.contains("NNAPI", ignoreCase = true)
        val lowRamMode = inferenceMode.contains("Low-RAM", ignoreCase = true)
        val companionBytes = modelFile.parentFile?.listFiles()
            ?.filter { it.isFile && it.absolutePath != modelFile.absolutePath && !it.name.endsWith(".meta") }
            ?.sumOf { it.length() } ?: 0L
        val totalBytes = modelFile.length() + companionBytes
        val isLargeExternalModel = totalBytes > 500L * 1024L * 1024L
        val isUltraLargeModel = totalBytes > 1_200L * 1024L * 1024L
        val cpuThreads = Runtime.getRuntime().availableProcessors().coerceIn(4, 6)

        if (preferNnapi && !isLargeExternalModel) {
            runCatching {
                OrtSession.SessionOptions().use { opts ->
                    opts.setOptimizationLevel(
                        if (lowRamMode) OrtSession.SessionOptions.OptLevel.BASIC_OPT
                        else OrtSession.SessionOptions.OptLevel.ALL_OPT
                    )
                    opts.setIntraOpNumThreads(cpuThreads)
                    if (lowRamMode) {
                        opts.setCPUArenaAllocator(false)
                        opts.setMemoryPatternOptimization(false)
                    }
                    opts.addNnapi(EnumSet.noneOf(ai.onnxruntime.providers.NNAPIFlags::class.java))
                    return env.createSession(modelFile.absolutePath, opts)
                }
            }
        }
        OrtSession.SessionOptions().use { opts ->
            // Ultra-large (>1.2 GB, e.g. 3.44 GB FP32 weights.pb) uses NO_OPT so ORT never
            // duplicates gigabytes of external tensor initializers during constant folding.
            val optLevel = when {
                isUltraLargeModel -> OrtSession.SessionOptions.OptLevel.NO_OPT
                isLargeExternalModel || lowRamMode -> OrtSession.SessionOptions.OptLevel.BASIC_OPT
                else -> OrtSession.SessionOptions.OptLevel.ALL_OPT
            }
            opts.setOptimizationLevel(optLevel)
            opts.setIntraOpNumThreads(cpuThreads)
            if (isLargeExternalModel || lowRamMode) {
                opts.setCPUArenaAllocator(false)
                opts.setMemoryPatternOptimization(false)
            }
            return env.createSession(modelFile.absolutePath, opts)
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
            session.run(mapOf(inputName to tensor)).use { result ->
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
        imageHeight: Int
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

            unetSession.run(inputs).use { results ->
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

            unetSession.run(inputs).use { results ->
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
            vaeSession.run(mapOf(inputName to tensor)).use { results ->
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
