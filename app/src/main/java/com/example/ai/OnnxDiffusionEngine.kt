package com.example.ai

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import android.content.Context
import android.graphics.Bitmap
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
        for (slot in DiffusionModelSlot.entries) {
            currentCoroutineContext().ensureActive()
            val file = modelManager.resolveExistingSlotFile(slot)
            if (file == null) {
                lines.add("• ${slot.canonicalFileName}: Not uploaded yet")
                continue
            }
            val t0 = System.currentTimeMillis()
            val session = createSafeSession(env, file, preferNnapi)
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
        if (verifiedCount == 0) {
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

        // 2. Verify required 3 Text-to-Image ONNX models exist
        val textEncoderFile = modelManager.resolveExistingSlotFile(DiffusionModelSlot.TEXT_ENCODER)
            ?: error("Missing text_encoder.onnx. Please upload your ONNX folder in Models tab.")
        val unetFile = modelManager.resolveExistingSlotFile(DiffusionModelSlot.UNET)
            ?: error("Missing unet.onnx. Please upload your ONNX folder in Models tab.")
        val vaeDecoderFile = modelManager.resolveExistingSlotFile(DiffusionModelSlot.VAE_DECODER)
            ?: error("Missing vae_decoder.onnx. Please upload your ONNX folder in Models tab.")

        val (width, height) = parseResolution(resolutionLabel)
        val latentW = width / 8
        val latentH = height / 8
        val latentChannels = 4
        val latentSize = latentChannels * latentH * latentW
        val preferNnapi = inferenceMode.contains("NNAPI", ignoreCase = true)
        val env = OrtEnvironment.getEnvironment()

        // 3. Tokenize Final Generation Input (User Prompt + Saved Content Safety Prompt)
        onProgress(
            GenerationProgress(
                stageTitle = "Encoding Prompt (CLIP Text Encoder)...",
                currentStep = 0,
                totalSteps = steps,
                percent = 6,
                elapsedMs = System.currentTimeMillis() - startMs
            )
        )
        checkNotCancelled()

        val normalizedClipPrompt = OfflineTamilPromptTranslator.normalizeForClip(finalGenerationInput)
        val posTokenIds = tokenizer.encode(normalizedClipPrompt)
        val needClassifierFreeGuidance = samplerType == SamplerType.EULER_STANDARD && guidanceScale > 1.15f
        val negTokenIds = if (needClassifierFreeGuidance) {
            tokenizer.encode(negativePrompt.ifBlank { "blurry, deformed, bad anatomy, low quality" })
        } else {
            null
        }

        // 4. Stage 1: Run Text Encoder & immediately release session to save RAM
        val (posEmbeddings, negEmbeddings, hiddenDim) = createSafeSession(env, textEncoderFile, preferNnapi).use { teSession ->
            val posOut = runTextEncoder(env, teSession, posTokenIds)
            val negOut = if (negTokenIds != null) {
                checkNotCancelled()
                runTextEncoder(env, teSession, negTokenIds).first
            } else {
                null
            }
            Triple(posOut.first, negOut, posOut.second)
        }
        System.gc()

        onProgress(
            GenerationProgress(
                stageTitle = "Loading UNet Diffusion Engine...",
                currentStep = 0,
                totalSteps = steps,
                percent = 14,
                elapsedMs = System.currentTimeMillis() - startMs
            )
        )
        checkNotCancelled()

        // 5. Stage 2: Run UNet Denoising Loop (LCM or Euler) & release session after loop
        val denoisedLatents = createSafeSession(env, unetFile, preferNnapi).use { unetSession ->
            if (samplerType == SamplerType.LCM_FAST) {
                val scheduler = DiffusionSchedulers.LcmSession(steps)
                var latents = DiffusionSchedulers.createRandomLatents(latentSize, seed, initSigma = 1.0f)

                for (stepIdx in 0 until steps) {
                    checkNotCancelled()
                    val t = scheduler.timesteps[stepIdx]
                    val pct = 16 + (((stepIdx + 1).toFloat() / steps.toFloat()) * 68f).toInt()
                    onProgress(
                        GenerationProgress(
                            stageTitle = "Generating (LCM Step ${stepIdx + 1}/$steps)...",
                            currentStep = stepIdx + 1,
                            totalSteps = steps,
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
                        guidanceScale = guidanceScale,
                        imageWidth = width,
                        imageHeight = height
                    )
                    latents = scheduler.step(noisePred, stepIdx, latents, seed)
                }
                latents
            } else {
                val scheduler = DiffusionSchedulers.EulerSession(steps)
                var latents = DiffusionSchedulers.createRandomLatents(
                    size = latentSize,
                    seed = seed,
                    initSigma = scheduler.initNoiseSigma
                )

                for (stepIdx in 0 until steps) {
                    checkNotCancelled()
                    val t = scheduler.timesteps[stepIdx]
                    val pct = 16 + (((stepIdx + 1).toFloat() / steps.toFloat()) * 68f).toInt()
                    onProgress(
                        GenerationProgress(
                            stageTitle = "Denoising (Euler Step ${stepIdx + 1}/$steps)...",
                            currentStep = stepIdx + 1,
                            totalSteps = steps,
                            percent = pct,
                            elapsedMs = System.currentTimeMillis() - startMs
                        )
                    )

                    val scaledLatents = scheduler.scaleModelInput(latents, stepIdx)
                    val condNoise = runUnetSingleBatch(
                        env = env,
                        unetSession = unetSession,
                        latents = scaledLatents,
                        latentH = latentH,
                        latentW = latentW,
                        timestep = t,
                        encoderHiddenStates = posEmbeddings,
                        hiddenDim = hiddenDim,
                        guidanceScale = guidanceScale,
                        imageWidth = width,
                        imageHeight = height
                    )

                    val finalNoise = if (negEmbeddings != null) {
                        checkNotCancelled()
                        val uncondNoise = runUnetSingleBatch(
                            env = env,
                            unetSession = unetSession,
                            latents = scaledLatents,
                            latentH = latentH,
                            latentW = latentW,
                            timestep = t,
                            encoderHiddenStates = negEmbeddings,
                            hiddenDim = hiddenDim,
                            guidanceScale = guidanceScale,
                            imageWidth = width,
                            imageHeight = height
                        )
                        FloatArray(condNoise.size) { i ->
                            uncondNoise[i] + guidanceScale * (condNoise[i] - uncondNoise[i])
                        }
                    } else {
                        condNoise
                    }

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
                currentStep = steps,
                totalSteps = steps,
                percent = 90,
                elapsedMs = System.currentTimeMillis() - startMs
            )
        )
        checkNotCancelled()

        val scaledForVae = FloatArray(denoisedLatents.size) { i ->
            denoisedLatents[i] / 0.18215f
        }

        val bitmap = createSafeSession(env, vaeDecoderFile, preferNnapi).use { vaeSession ->
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
                currentStep = steps,
                totalSteps = steps,
                percent = 100,
                elapsedMs = totalMs
            )
        )

        TextToImageResult(
            bitmap = bitmap,
            finalGenerationInput = finalGenerationInput,
            normalizedClipPrompt = normalizedClipPrompt,
            seed = seed,
            steps = steps,
            resolutionLabel = "$width × $height",
            durationMs = totalMs
        )
    }

    private suspend fun checkNotCancelled() {
        currentCoroutineContext().ensureActive()
        if (cancelRequested.get()) {
            throw CancellationException("Generation cancelled by user.")
        }
    }

    private fun createSafeSession(
        env: OrtEnvironment,
        modelFile: File,
        preferNnapi: Boolean
    ): OrtSession {
        if (preferNnapi) {
            runCatching {
                OrtSession.SessionOptions().use { opts ->
                    opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
                    opts.setIntraOpNumThreads(4)
                    opts.addNnapi(EnumSet.noneOf(ai.onnxruntime.providers.NNAPIFlags::class.java))
                    return env.createSession(modelFile.absolutePath, opts)
                }
            }
        }
        OrtSession.SessionOptions().use { opts ->
            opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            opts.setIntraOpNumThreads(4)
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
        val createdTensors = mutableListOf<OnnxTensor>()
        try {
            val inputs = LinkedHashMap<String, OnnxTensor>()
            for ((name, nodeInfo) in unetSession.inputInfo) {
                val ti = nodeInfo.info as? TensorInfo ?: continue
                val lower = name.lowercase(Locale.US)
                val tensor = when {
                    lower == "sample" || lower.contains("latent") -> {
                        createFloatOrFp16Tensor(
                            env = env,
                            data = latents,
                            shape = longArrayOf(1L, 4L, latentH.toLong(), latentW.toLong()),
                            targetType = ti.type
                        )
                    }
                    lower == "timestep" || lower == "t" -> {
                        createTimestepTensor(env, timestep, ti)
                    }
                    lower.contains("encoder_hidden_states") || lower.contains("context") -> {
                        val expectedDim = if (ti.shape.size >= 3 && ti.shape[2] > 0) {
                            ti.shape[2].toInt()
                        } else {
                            hiddenDim
                        }
                        val adaptedHidden = adaptEmbeddingDimension(encoderHiddenStates, 77, expectedDim)
                        createFloatOrFp16Tensor(
                            env = env,
                            data = adaptedHidden,
                            shape = longArrayOf(1L, 77L, expectedDim.toLong()),
                            targetType = ti.type
                        )
                    }
                    lower.contains("timestep_cond") -> {
                        val condDim = if (ti.shape.size >= 2 && ti.shape[1] > 0) ti.shape[1].toInt() else 256
                        val wEmbed = DiffusionSchedulers.buildGuidanceScaleEmbedding(
                            w = (guidanceScale - 1.0f).coerceAtLeast(0f),
                            embeddingDim = condDim
                        )
                        createFloatOrFp16Tensor(
                            env = env,
                            data = wEmbed,
                            shape = longArrayOf(1L, condDim.toLong()),
                            targetType = ti.type
                        )
                    }
                    lower.contains("time_ids") -> {
                        val timeIds = floatArrayOf(
                            imageHeight.toFloat(), imageWidth.toFloat(),
                            0f, 0f,
                            imageHeight.toFloat(), imageWidth.toFloat()
                        )
                        createFloatOrFp16Tensor(
                            env = env,
                            data = timeIds,
                            shape = longArrayOf(1L, 6L),
                            targetType = ti.type
                        )
                    }
                    else -> {
                        val concreteShape = ti.shape.map { if (it <= 0L) 1L else it }.toLongArray()
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
                return extractFloatArray(outTensor)
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
        info: TensorInfo
    ): OnnxTensor {
        val isScalar = info.shape.isEmpty()
        val shape = if (isScalar) longArrayOf() else longArrayOf(1L)
        return when (info.type) {
            OnnxJavaType.INT64 -> OnnxTensor.createTensor(env, LongBuffer.wrap(longArrayOf(timestep.toLong())), shape)
            OnnxJavaType.INT32 -> OnnxTensor.createTensor(env, IntBuffer.wrap(intArrayOf(timestep)), shape)
            OnnxJavaType.FLOAT16 -> {
                val half = floatToHalfBits(timestep.toFloat())
                OnnxTensor.createTensor(env, ShortBuffer.wrap(shortArrayOf(half)), shape, OnnxJavaType.FLOAT16)
            }
            else -> OnnxTensor.createTensor(env, FloatBuffer.wrap(floatArrayOf(timestep.toFloat())), shape)
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
