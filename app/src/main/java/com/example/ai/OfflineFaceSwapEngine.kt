package com.example.ai

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import android.content.Context
import android.graphics.Bitmap
import android.graphics.PointF
import android.graphics.RectF
import android.net.Uri
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import java.io.BufferedInputStream
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.ShortBuffer
import java.util.EnumSet
import kotlin.math.exp
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

data class FaceSwapResult(
    val swappedBitmap: Bitmap,
    val targetPreviewFile: File,
    val durationMs: Long,
    val modelSummary: String
)

/**
 * Self-contained Offline ONNX Face Swap Engine compatible with nvmk-dream/face-swap models:
 *  - det_10g.onnx (SCRFD 5-point landmark detector)
 *  - w600k_r50.onnx / w600k_mbf.onnx (512-D ArcFace identity recognizer, with 2-model fallback)
 *  - inswapper_128.onnx (128x128 + embedded [512,512] emap) OR hyperswap_*_256.onnx (256x256)
 */
class OfflineFaceSwapEngine(
    private val context: Context,
    private val modelManager: OnnxModelManager
) {
    data class SimpleDetectedFace(
        val bbox: RectF,
        val score: Float,
        val landmarks5: List<PointF> // [leftEye, rightEye, nose, leftMouth, rightMouth]
    ) {
        val area: Float get() = bbox.width() * bbox.height()
    }

    suspend fun executeOfflineFaceSwap(
        sourceFaceUri: Uri,
        targetImageUri: Uri?,
        targetFallbackFilePath: String?,
        enableSkinToneMatch: Boolean,
        blendStrength: Float,
        inferenceMode: String,
        onProgress: (GenerationProgress) -> Unit
    ): Result<FaceSwapResult> = runCatching {
        val t0 = System.currentTimeMillis()
        val detFile = modelManager.resolveExistingSlotFile(DiffusionModelSlot.FACE_DETECTOR)
            ?: error("Missing det_10g.onnx. Upload your Face Swap models in the Models tab.")
        val swapFile = modelManager.resolveExistingSlotFile(DiffusionModelSlot.FACE_SWAPPER)
            ?: error("Missing inswapper_128.onnx / hyperswap. Upload your Face Swap models in the Models tab.")
        val recFile = modelManager.resolveExistingSlotFile(DiffusionModelSlot.FACE_RECOGNIZER)

        val preferNnapi = inferenceMode.contains("NNAPI", ignoreCase = true)
        val env = OrtEnvironment.getEnvironment()

        onProgress(GenerationProgress("Loading Source & Target Photos...", 1, 5, 10, System.currentTimeMillis() - t0))
        currentCoroutineContext().ensureActive()

        val sourceBitmap = decodeFlexibleBitmap(sourceFaceUri, null, maxSide = 1024)
        val targetBitmap = decodeFlexibleBitmap(targetImageUri, targetFallbackFilePath, maxSide = 1024)
        val targetPreviewFile = ImageBitmapPreprocessor.saveReferencePreview(context, targetBitmap)

        try {
            // Stage 1: Detect 5-point landmarks on Source & Target using det_10g.onnx
            onProgress(GenerationProgress("Detecting 5-Point Landmarks (det_10g)...", 2, 5, 28, System.currentTimeMillis() - t0))
            currentCoroutineContext().ensureActive()

            val (srcFace, tgtFace) = createSession(env, detFile, preferNnapi = false).use { detSession ->
                val sFaces = detectScrfdFaces(env, detSession, sourceBitmap)
                if (sFaces.isEmpty()) error("No clear face detected in Source photo.")
                val tFaces = detectScrfdFaces(env, detSession, targetBitmap)
                if (tFaces.isEmpty()) error("No clear face detected in Target photo.")
                sFaces.maxBy { it.area * it.score } to tFaces.maxBy { it.area * it.score }
            }
            System.gc()

            // Stage 2: Extract 512-D ArcFace identity + optional 512x512 emap projection
            onProgress(GenerationProgress("Extracting 512-D Face Identity...", 3, 5, 50, System.currentTimeMillis() - t0))
            currentCoroutineContext().ensureActive()

            val alignedSrc112 = warpFace5Pt(sourceBitmap, srcFace.landmarks5, cropSize = 112, arcFaceTemplate = true)
            val rawEmbedding512 = try {
                if (recFile != null && recFile.exists()) {
                    createSession(env, recFile, preferNnapi).use { recSession ->
                        runArcFaceEmbedding(env, recSession, alignedSrc112)
                    }
                } else {
                    computeFallbackIdentity512(alignedSrc112)
                }
            } finally {
                alignedSrc112.recycle()
            }
            val normedArcFace = l2Normalize(rawEmbedding512)

            val isHyperSwap = swapFile.name.contains("hyperswap", ignoreCase = true)
            val emap = if (isHyperSwap) null else loadOrExtractEmap512x512(swapFile)
            val latentIdentity512 = if (emap != null && emap.size == 512 * 512) {
                val proj = FloatArray(512)
                for (col in 0 until 512) {
                    var sum = 0f
                    for (row in 0 until 512) {
                        sum += normedArcFace[row] * emap[row * 512 + col]
                    }
                    proj[col] = sum
                }
                l2Normalize(proj)
            } else {
                normedArcFace
            }
            System.gc()

            // Stage 3: Run InSwapper-128 or HyperSwap-256
            onProgress(GenerationProgress("Synthesizing Swapped Face...", 4, 5, 74, System.currentTimeMillis() - t0))
            currentCoroutineContext().ensureActive()

            val (swappedCrop, forwardMatrix, cropRes) = createSession(env, swapFile, preferNnapi).use { swapSession ->
                val res = detectSwapModelResolution(swapSession)
                val useArcTemplate = res == 112
                val mForward = estimateSimilarityMatrix5Pt(tgtFace.landmarks5, res, useArcTemplate)
                val alignedTarget = warpAffineCrop(targetBitmap, mForward, res)
                val outCrop = try {
                    runSwapSession(env, swapSession, alignedTarget, latentIdentity512, res)
                } finally {
                    alignedTarget.recycle()
                }
                Triple(outCrop, mForward, res)
            }
            System.gc()

            // Stage 4: Color Match & Feathered Inverse-Warp Blend onto Full Target Photo
            onProgress(GenerationProgress("Blending & Matching Skin Tone...", 5, 5, 92, System.currentTimeMillis() - t0))
            currentCoroutineContext().ensureActive()

            val finalBitmap = try {
                blendSwappedCropOntoTarget(
                    targetBitmap = targetBitmap,
                    swappedCrop = swappedCrop,
                    forwardMatrix = forwardMatrix,
                    cropSize = cropRes,
                    enableColorMatch = enableSkinToneMatch,
                    blendStrength = blendStrength.coerceIn(0.4f, 1.0f)
                )
            } finally {
                swappedCrop.recycle()
            }

            val totalMs = System.currentTimeMillis() - t0
            onProgress(GenerationProgress("Completed!", 5, 5, 100, totalMs))

            FaceSwapResult(
                swappedBitmap = finalBitmap,
                targetPreviewFile = targetPreviewFile,
                durationMs = totalMs,
                modelSummary = "${swapFile.name} (${cropRes}px)"
            )
        } finally {
            sourceBitmap.recycle()
            targetBitmap.recycle()
        }
    }

    private fun decodeFlexibleBitmap(uri: Uri?, fallbackPath: String?, maxSide: Int): Bitmap {
        if (uri != null) {
            return ImageBitmapPreprocessor.loadAndCenterCropFromUri(context, uri, maxSide, maxSide)
        }
        if (!fallbackPath.isNullOrBlank() && File(fallbackPath).exists()) {
            val raw = android.graphics.BitmapFactory.decodeFile(fallbackPath)
                ?: error("Could not decode image.")
            return ImageBitmapPreprocessor.centerCropAndResize(raw, maxSide, maxSide).also {
                if (it !== raw) raw.recycle()
            }
        }
        error("Please select an image first.")
    }

    private fun createSession(env: OrtEnvironment, file: File, preferNnapi: Boolean): OrtSession {
        if (preferNnapi) {
            runCatching {
                OrtSession.SessionOptions().use { opts ->
                    opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
                    opts.setIntraOpNumThreads(4)
                    opts.addNnapi(EnumSet.noneOf(ai.onnxruntime.providers.NNAPIFlags::class.java))
                    return env.createSession(file.absolutePath, opts)
                }
            }
        }
        OrtSession.SessionOptions().use { opts ->
            opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            opts.setIntraOpNumThreads(4)
            return env.createSession(file.absolutePath, opts)
        }
    }

    // =========================================================================
    // SCRFD-10G 5-Point Face Detector ([1, 3, 640, 640], strides 8/16/32)
    // =========================================================================
    private fun detectScrfdFaces(
        env: OrtEnvironment,
        session: OrtSession,
        bitmap: Bitmap,
        confThreshold: Float = 0.35f
    ): List<SimpleDetectedFace> {
        val inputSize = 640
        val scale = min(inputSize.toFloat() / bitmap.width, inputSize.toFloat() / bitmap.height)
        val scaledW = (bitmap.width * scale).toInt().coerceIn(1, inputSize)
        val scaledH = (bitmap.height * scale).toInt().coerceIn(1, inputSize)

        val resized = Bitmap.createScaledBitmap(bitmap, scaledW, scaledH, true)
        val letterbox = Bitmap.createBitmap(inputSize, inputSize, Bitmap.Config.ARGB_8888)
        android.graphics.Canvas(letterbox).drawBitmap(resized, 0f, 0f, null)
        if (resized !== bitmap) resized.recycle()

        val px = IntArray(inputSize * inputSize)
        letterbox.getPixels(px, 0, inputSize, 0, 0, inputSize, inputSize)
        letterbox.recycle()

        val plane = inputSize * inputSize
        val inputFloats = FloatArray(3 * plane)
        for (i in 0 until plane) {
            val c = px[i]
            inputFloats[i] = (((c ushr 16) and 0xFF) - 127.5f) / 128.0f
            inputFloats[plane + i] = (((c ushr 8) and 0xFF) - 127.5f) / 128.0f
            inputFloats[2 * plane + i] = ((c and 0xFF) - 127.5f) / 128.0f
        }

        val inputName = session.inputNames.first()
        val inInfo = session.inputInfo[inputName]?.info as? TensorInfo
        val tensor = createFloatOrFp16(env, inputFloats, longArrayOf(1, 3, 640, 640), inInfo?.type)

        val candidates = mutableListOf<SimpleDetectedFace>()
        tensor.use { t ->
            session.run(mapOf(inputName to t)).use { outputs ->
                val outArrays = (0 until outputs.size()).map { idx ->
                    extractFloats(outputs[idx] as OnnxTensor)
                }
                val strides = intArrayOf(8, 16, 32)
                val anchorCounts = intArrayOf(12800, 3200, 800)

                for (sIdx in strides.indices) {
                    val stride = strides[sIdx]
                    val nAnchors = anchorCounts[sIdx]
                    val featW = inputSize / stride
                    val scoreArr = outArrays.firstOrNull { it.size == nAnchors } ?: continue
                    val bboxArr = outArrays.firstOrNull { it.size == nAnchors * 4 } ?: continue
                    val kpsArr = outArrays.firstOrNull { it.size == nAnchors * 10 }

                    for (a in 0 until nAnchors) {
                        val rawScore = scoreArr[a]
                        val prob = if (rawScore in 0.0f..1.0f) rawScore else (1.0f / (1.0f + exp(-rawScore)))
                        if (prob < confThreshold) continue

                        val cellIdx = a / 2
                        val cy = (cellIdx / featW) * stride.toFloat()
                        val cx = (cellIdx % featW) * stride.toFloat()

                        val x1 = ((cx - bboxArr[a * 4] * stride) / scale).coerceIn(0f, bitmap.width.toFloat())
                        val y1 = ((cy - bboxArr[a * 4 + 1] * stride) / scale).coerceIn(0f, bitmap.height.toFloat())
                        val x2 = ((cx + bboxArr[a * 4 + 2] * stride) / scale).coerceIn(0f, bitmap.width.toFloat())
                        val y2 = ((cy + bboxArr[a * 4 + 3] * stride) / scale).coerceIn(0f, bitmap.height.toFloat())

                        val pts = if (kpsArr != null) {
                            List(5) { k ->
                                PointF(
                                    ((cx + kpsArr[a * 10 + k * 2] * stride) / scale),
                                    ((cy + kpsArr[a * 10 + k * 2 + 1] * stride) / scale)
                                )
                            }
                        } else {
                            val w = x2 - x1
                            val h = y2 - y1
                            listOf(
                                PointF(x1 + 0.31f * w, y1 + 0.38f * h),
                                PointF(x1 + 0.69f * w, y1 + 0.38f * h),
                                PointF(x1 + 0.50f * w, y1 + 0.56f * h),
                                PointF(x1 + 0.35f * w, y1 + 0.76f * h),
                                PointF(x1 + 0.65f * w, y1 + 0.76f * h)
                            )
                        }
                        if (x2 - x1 > 16f && y2 - y1 > 16f) {
                            candidates.add(SimpleDetectedFace(RectF(x1, y1, x2, y2), prob, pts))
                        }
                    }
                }
            }
        }
        return nmsFaces(candidates, 0.4f)
    }

    private fun nmsFaces(faces: List<SimpleDetectedFace>, iouThresh: Float): List<SimpleDetectedFace> {
        val sorted = faces.sortedByDescending { it.score }.toMutableList()
        val kept = mutableListOf<SimpleDetectedFace>()
        while (sorted.isNotEmpty()) {
            val best = sorted.removeAt(0)
            kept.add(best)
            sorted.removeAll { other -> computeIou(best.bbox, other.bbox) > iouThresh }
        }
        return kept
    }

    private fun computeIou(a: RectF, b: RectF): Float {
        val ix1 = max(a.left, b.left)
        val iy1 = max(a.top, b.top)
        val ix2 = min(a.right, b.right)
        val iy2 = min(a.bottom, b.bottom)
        val iw = (ix2 - ix1).coerceAtLeast(0f)
        val ih = (iy2 - iy1).coerceAtLeast(0f)
        val inter = iw * ih
        val union = a.width() * a.height() + b.width() * b.height() - inter
        return if (union <= 0f) 0f else inter / union
    }

    // =========================================================================
    // 5-Point Umeyama Similarity Alignment & Affine Crop
    // =========================================================================
    private fun estimateSimilarityMatrix5Pt(
        srcPts: List<PointF>,
        cropSize: Int,
        arcFace112Template: Boolean
    ): FloatArray {
        // Canonical InsightFace 5-point landmarks
        val base112 = arrayOf(
            floatArrayOf(38.2946f, 51.6963f),
            floatArrayOf(73.5318f, 51.5014f),
            floatArrayOf(56.0252f, 71.7366f),
            floatArrayOf(41.5493f, 92.3655f),
            floatArrayOf(70.7299f, 92.2041f)
        )
        val dstPts = Array(5) { i ->
            if (arcFace112Template && cropSize == 112) {
                floatArrayOf(base112[i][0], base112[i][1])
            } else {
                // InsightFace InSwapper / HyperSwap centered template (ratio = cropSize / 128)
                val ratio = cropSize.toFloat() / 128.0f
                floatArrayOf((base112[i][0] + 8.0f) * ratio, base112[i][1] * ratio)
            }
        }

        var srcMeanX = 0f
        var srcMeanY = 0f
        var dstMeanX = 0f
        var dstMeanY = 0f
        for (i in 0 until 5) {
            srcMeanX += srcPts[i].x
            srcMeanY += srcPts[i].y
            dstMeanX += dstPts[i][0]
            dstMeanY += dstPts[i][1]
        }
        srcMeanX /= 5f; srcMeanY /= 5f; dstMeanX /= 5f; dstMeanY /= 5f

        var srcVar = 0f
        var covXX = 0f
        var covXY = 0f
        for (i in 0 until 5) {
            val sx = srcPts[i].x - srcMeanX
            val sy = srcPts[i].y - srcMeanY
            val dx = dstPts[i][0] - dstMeanX
            val dy = dstPts[i][1] - dstMeanY
            srcVar += sx * sx + sy * sy
            covXX += sx * dx + sy * dy
            covXY += sx * dy - sy * dx
        }
        val denom = srcVar.coerceAtLeast(1e-6f)
        val a = covXX / denom
        val b = covXY / denom
        val tx = dstMeanX - (a * srcMeanX - b * srcMeanY)
        val ty = dstMeanY - (b * srcMeanX + a * srcMeanY)
        return floatArrayOf(a, -b, tx, b, a, ty)
    }

    private fun invertAffine2x3(m: FloatArray): FloatArray {
        val a = m[0]; val b = m[1]; val tx = m[2]
        val c = m[3]; val d = m[4]; val ty = m[5]
        val det = (a * d - b * c).let { if (kotlin.math.abs(it) < 1e-7f) 1e-7f else it }
        val ia = d / det
        val ib = -b / det
        val ic = -c / det
        val id = a / det
        val itx = -(ia * tx + ib * ty)
        val ity = -(ic * tx + id * ty)
        return floatArrayOf(ia, ib, itx, ic, id, ity)
    }

    private fun warpFace5Pt(src: Bitmap, pts: List<PointF>, cropSize: Int, arcFaceTemplate: Boolean): Bitmap {
        val m = estimateSimilarityMatrix5Pt(pts, cropSize, arcFaceTemplate)
        return warpAffineCrop(src, m, cropSize)
    }

    private fun warpAffineCrop(src: Bitmap, forwardM: FloatArray, cropSize: Int): Bitmap {
        val inv = invertAffine2x3(forwardM)
        val srcW = src.width
        val srcH = src.height
        val srcPx = IntArray(srcW * srcH)
        src.getPixels(srcPx, 0, srcW, 0, 0, srcW, srcH)

        val outPx = IntArray(cropSize * cropSize)
        for (y in 0 until cropSize) {
            for (x in 0 until cropSize) {
                val sx = inv[0] * x + inv[1] * y + inv[2]
                val sy = inv[3] * x + inv[4] * y + inv[5]
                outPx[y * cropSize + x] = sampleBilinear(srcPx, srcW, srcH, sx, sy)
            }
        }
        return Bitmap.createBitmap(outPx, cropSize, cropSize, Bitmap.Config.ARGB_8888)
    }

    private fun sampleBilinear(px: IntArray, w: Int, h: Int, x: Float, y: Float): Int {
        val cx = x.coerceIn(0f, (w - 1).toFloat())
        val cy = y.coerceIn(0f, (h - 1).toFloat())
        val x0 = cx.toInt()
        val y0 = cy.toInt()
        val x1 = (x0 + 1).coerceAtMost(w - 1)
        val y1 = (y0 + 1).coerceAtMost(h - 1)
        val fx = cx - x0
        val fy = cy - y0

        val c00 = px[y0 * w + x0]
        val c10 = px[y0 * w + x1]
        val c01 = px[y1 * w + x0]
        val c11 = px[y1 * w + x1]

        fun chan(shift: Int): Int {
            val v00 = (c00 ushr shift) and 0xFF
            val v10 = (c10 ushr shift) and 0xFF
            val v01 = (c01 ushr shift) and 0xFF
            val v11 = (c11 ushr shift) and 0xFF
            val top = v00 + (v10 - v00) * fx
            val bot = v01 + (v11 - v01) * fx
            return (top + (bot - top) * fy + 0.5f).toInt().coerceIn(0, 255)
        }
        return (0xFF shl 24) or (chan(16) shl 16) or (chan(8) shl 8) or chan(0)
    }

    // =========================================================================
    // ArcFace 512-D Identity + InSwapper/HyperSwap Inference
    // =========================================================================
    private fun runArcFaceEmbedding(env: OrtEnvironment, session: OrtSession, aligned112: Bitmap): FloatArray {
        val px = IntArray(112 * 112)
        aligned112.getPixels(px, 0, 112, 0, 0, 112, 112)
        val plane = 112 * 112
        val floats = FloatArray(3 * plane)
        for (i in 0 until plane) {
            val c = px[i]
            floats[i] = (((c ushr 16) and 0xFF) - 127.5f) / 127.5f
            floats[plane + i] = (((c ushr 8) and 0xFF) - 127.5f) / 127.5f
            floats[2 * plane + i] = ((c and 0xFF) - 127.5f) / 127.5f
        }
        val inName = session.inputNames.first()
        val inInfo = session.inputInfo[inName]?.info as? TensorInfo
        createFloatOrFp16(env, floats, longArrayOf(1, 3, 112, 112), inInfo?.type).use { tensor ->
            session.run(mapOf(inName to tensor)).use { res ->
                return extractFloats(res[0] as OnnxTensor)
            }
        }
    }

    private fun computeFallbackIdentity512(aligned112: Bitmap): FloatArray {
        val small = Bitmap.createScaledBitmap(aligned112, 16, 16, true)
        val px = IntArray(256)
        small.getPixels(px, 0, 16, 0, 0, 16, 16)
        if (small !== aligned112) small.recycle()
        val vec = FloatArray(512)
        for (i in 0 until 256) {
            val c = px[i]
            vec[i] = (((c ushr 16) and 0xFF) - 127.5f) / 127.5f
            vec[256 + i] = (((c ushr 8) and 0xFF) - 127.5f) / 127.5f
        }
        return vec
    }

    private fun detectSwapModelResolution(session: OrtSession): Int {
        for ((_, node) in session.inputInfo) {
            val shape = (node.info as? TensorInfo)?.shape ?: continue
            if (shape.size == 4 && shape[2] > 0) {
                return shape[2].toInt()
            }
        }
        return 128
    }

    private fun runSwapSession(
        env: OrtEnvironment,
        session: OrtSession,
        alignedTarget: Bitmap,
        sourceLatent512: FloatArray,
        res: Int
    ): Bitmap {
        val plane = res * res
        val px = IntArray(plane)
        alignedTarget.getPixels(px, 0, res, 0, 0, res, res)

        val isHyperSwap = res >= 256
        val targetFloats = FloatArray(3 * plane)
        for (i in 0 until plane) {
            val c = px[i]
            val r = (c ushr 16) and 0xFF
            val g = (c ushr 8) and 0xFF
            val b = c and 0xFF
            if (isHyperSwap) {
                // HyperSwap 256 expects [-1, 1] RGB
                targetFloats[i] = (r / 127.5f) - 1.0f
                targetFloats[plane + i] = (g / 127.5f) - 1.0f
                targetFloats[2 * plane + i] = (b / 127.5f) - 1.0f
            } else {
                // InsightFace inswapper_128 expects [0, 1] RGB
                targetFloats[i] = r / 255.0f
                targetFloats[plane + i] = g / 255.0f
                targetFloats[2 * plane + i] = b / 255.0f
            }
        }

        val created = mutableListOf<OnnxTensor>()
        try {
            val inputs = LinkedHashMap<String, OnnxTensor>()
            for ((name, node) in session.inputInfo) {
                val ti = node.info as? TensorInfo ?: continue
                val tensor = if (ti.shape.size == 4) {
                    createFloatOrFp16(env, targetFloats, longArrayOf(1, 3, res.toLong(), res.toLong()), ti.type)
                } else {
                    createFloatOrFp16(env, sourceLatent512, longArrayOf(1, 512), ti.type)
                }
                created.add(tensor)
                inputs[name] = tensor
            }

            session.run(inputs).use { results ->
                val outFloats = extractFloats(results[0] as OnnxTensor)
                val minVal = outFloats.minOrNull() ?: 0f
                val minusOneRange = minVal < -0.08f
                val outPx = IntArray(plane)
                for (i in 0 until plane) {
                    fun toByte(v: Float): Int {
                        val norm = if (minusOneRange) (v + 1.0f) * 0.5f else v
                        return (norm.coerceIn(0f, 1f) * 255f + 0.5f).toInt().coerceIn(0, 255)
                    }
                    val r = toByte(outFloats.getOrElse(i) { 0f })
                    val g = toByte(outFloats.getOrElse(plane + i) { 0f })
                    val b = toByte(outFloats.getOrElse(2 * plane + i) { 0f })
                    outPx[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
                }
                return Bitmap.createBitmap(outPx, res, res, Bitmap.Config.ARGB_8888)
            }
        } finally {
            created.forEach { runCatching { it.close() } }
        }
    }

    private fun blendSwappedCropOntoTarget(
        targetBitmap: Bitmap,
        swappedCrop: Bitmap,
        forwardMatrix: FloatArray,
        cropSize: Int,
        enableColorMatch: Boolean,
        blendStrength: Float
    ): Bitmap {
        val w = targetBitmap.width
        val h = targetBitmap.height
        val tgtPx = IntArray(w * h)
        targetBitmap.getPixels(tgtPx, 0, w, 0, 0, w, h)

        val cropPx = IntArray(cropSize * cropSize)
        swappedCrop.getPixels(cropPx, 0, cropSize, 0, 0, cropSize, cropSize)

        // Optional mean skin-tone offset transfer in inner face region
        var rShift = 0f; var gShift = 0f; var bShift = 0f
        if (enableColorMatch) {
            val alignedTgt = warpAffineCrop(targetBitmap, forwardMatrix, cropSize)
            val alignedTgtPx = IntArray(cropSize * cropSize)
            alignedTgt.getPixels(alignedTgtPx, 0, cropSize, 0, 0, cropSize, cropSize)
            alignedTgt.recycle()

            var sr = 0L; var sg = 0L; var sb = 0L
            var tr = 0L; var tg = 0L; var tb = 0L
            var cnt = 0
            val cMin = (cropSize * 0.25f).toInt()
            val cMax = (cropSize * 0.75f).toInt()
            for (cy in cMin until cMax) {
                for (cx in cMin until cMax) {
                    val idx = cy * cropSize + cx
                    val sc = cropPx[idx]
                    val tc = alignedTgtPx[idx]
                    sr += (sc ushr 16) and 0xFF; sg += (sc ushr 8) and 0xFF; sb += sc and 0xFF
                    tr += (tc ushr 16) and 0xFF; tg += (tc ushr 8) and 0xFF; tb += tc and 0xFF
                    cnt++
                }
            }
            if (cnt > 0) {
                rShift = ((tr - sr).toFloat() / cnt) * 0.45f
                gShift = ((tg - sg).toFloat() / cnt) * 0.45f
                bShift = ((tb - sb).toFloat() / cnt) * 0.45f
            }
        }

        val half = cropSize * 0.5f
        val rx = cropSize * 0.41f
        val ry = cropSize * 0.45f

        for (y in 0 until h) {
            for (x in 0 until w) {
                val u = forwardMatrix[0] * x + forwardMatrix[1] * y + forwardMatrix[2]
                val v = forwardMatrix[3] * x + forwardMatrix[4] * y + forwardMatrix[5]
                if (u < 1f || u >= cropSize - 1f || v < 1f || v >= cropSize - 1f) continue

                val dx = (u - half) / rx
                val dy = (v - half * 1.03f) / ry
                val dist = sqrt(dx * dx + dy * dy)
                if (dist >= 1.0f) continue

                // Smooth cosine/feathered falloff from 0.62 -> 1.0
                val feather = when {
                    dist <= 0.62f -> 1.0f
                    else -> ((1.0f - dist) / 0.38f).coerceIn(0f, 1f)
                } * blendStrength

                val swappedColor = sampleBilinear(cropPx, cropSize, cropSize, u, v)
                val baseColor = tgtPx[y * w + x]

                val sr = (((swappedColor ushr 16) and 0xFF) + rShift).coerceIn(0f, 255f)
                val sg = (((swappedColor ushr 8) and 0xFF) + gShift).coerceIn(0f, 255f)
                val sb = ((swappedColor and 0xFF) + bShift).coerceIn(0f, 255f)

                val br = (baseColor ushr 16) and 0xFF
                val bg = (baseColor ushr 8) and 0xFF
                val bb = baseColor and 0xFF

                val nr = (br + (sr - br) * feather + 0.5f).toInt().coerceIn(0, 255)
                val ng = (bg + (sg - bg) * feather + 0.5f).toInt().coerceIn(0, 255)
                val nb = (bb + (sb - bb) * feather + 0.5f).toInt().coerceIn(0, 255)

                tgtPx[y * w + x] = (0xFF shl 24) or (nr shl 16) or (ng shl 8) or nb
            }
        }
        return Bitmap.createBitmap(tgtPx, w, h, Bitmap.Config.ARGB_8888)
    }

    private fun l2Normalize(vec: FloatArray): FloatArray {
        var sumSq = 0.0
        for (v in vec) sumSq += v.toDouble() * v.toDouble()
        val norm = sqrt(sumSq).coerceAtLeast(1e-8)
        return FloatArray(vec.size) { i -> (vec[i] / norm).toFloat() }
    }

    private fun createFloatOrFp16(
        env: OrtEnvironment,
        data: FloatArray,
        shape: LongArray,
        type: OnnxJavaType?
    ): OnnxTensor {
        return if (type == OnnxJavaType.FLOAT16) {
            val shorts = ShortArray(data.size) { i -> OnnxDiffusionEngine.floatToHalfBits(data[i]) }
            OnnxTensor.createTensor(env, ShortBuffer.wrap(shorts), shape, OnnxJavaType.FLOAT16)
        } else {
            OnnxTensor.createTensor(env, FloatBuffer.wrap(data), shape)
        }
    }

    private fun extractFloats(tensor: OnnxTensor): FloatArray {
        val ti = tensor.info as? TensorInfo
        return if (ti?.type == OnnxJavaType.FLOAT16) {
            val sb = tensor.shortBuffer
            FloatArray(sb.remaining()) { OnnxDiffusionEngine.halfBitsToFloat(sb.get()) }
        } else {
            val fb = tensor.floatBuffer
            val out = FloatArray(fb.remaining())
            fb.get(out)
            out
        }
    }

    // =========================================================================
    // Lightweight ONNX Protobuf Scanner for embedded [512, 512] emap matrix
    // =========================================================================
    private fun loadOrExtractEmap512x512(inswapperFile: File): FloatArray? {
        val cacheFile = File(modelManager.modelsRootDir, "inswapper_emap_512x512.bin")
        val expectedBytes = 512 * 512 * 4
        if (cacheFile.exists() && cacheFile.length() == expectedBytes.toLong()) {
            return runCatching {
                val floats = FloatArray(512 * 512)
                ByteBuffer.wrap(cacheFile.readBytes()).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(floats)
                floats
            }.getOrNull()
        }

        val extracted = runCatching { scanProtobufFor512x512Tensor(inswapperFile) }.getOrNull()
        if (extracted != null && extracted.size == 512 * 512) {
            runCatching {
                val buf = ByteBuffer.allocate(expectedBytes).order(ByteOrder.LITTLE_ENDIAN)
                buf.asFloatBuffer().put(extracted)
                FileOutputStream(cacheFile).use { it.write(buf.array()) }
            }
        }
        return extracted
    }

    private fun scanProtobufFor512x512Tensor(file: File): FloatArray? {
        BufferedInputStream(FileInputStream(file), 256 * 1024).use { stream ->
            var bytesRead = 0L
            val totalLen = file.length()

            fun readByte(): Int {
                if (bytesRead >= totalLen) return -1
                val b = stream.read()
                if (b >= 0) bytesRead++
                return b
            }

            fun readVarint(): Long {
                var result = 0L
                var shift = 0
                while (shift < 64) {
                    val b = readByte()
                    if (b < 0) return -1L
                    result = result or ((b.toLong() and 0x7FL) shl shift)
                    if ((b and 0x80) == 0) return result
                    shift += 7
                }
                return -1L
            }

            fun skipBytes(count: Long) {
                var rem = count
                while (rem > 0) {
                    val skipped = stream.skip(rem)
                    if (skipped <= 0) {
                        if (readByte() < 0) break
                    } else {
                        bytesRead += skipped
                        rem -= skipped
                    }
                }
            }

            fun skipWire(wire: Int) {
                when (wire) {
                    0 -> readVarint()
                    1 -> skipBytes(8)
                    2 -> { val l = readVarint(); if (l > 0) skipBytes(l) }
                    5 -> skipBytes(4)
                }
            }

            var lastMatch: FloatArray? = null
            while (bytesRead < totalLen) {
                val tag = readVarint()
                if (tag <= 0) break
                val field = (tag ushr 3).toInt()
                val wire = (tag and 0x07).toInt()
                if (field == 7 && wire == 2) {
                    val graphLen = readVarint()
                    val graphEnd = bytesRead + graphLen
                    while (bytesRead < graphEnd) {
                        val gTag = readVarint()
                        if (gTag <= 0) break
                        val gField = (gTag ushr 3).toInt()
                        val gWire = (gTag and 0x07).toInt()
                        if (gField == 5 && gWire == 2) {
                            val tLen = readVarint()
                            val tEnd = bytesRead + tLen
                            if (tLen in 1_048_576L..1_350_000L) {
                                val dims = mutableListOf<Long>()
                                var rawFloats: FloatArray? = null
                                while (bytesRead < tEnd) {
                                    val fTag = readVarint()
                                    if (fTag <= 0) break
                                    val fNum = (fTag ushr 3).toInt()
                                    val fWire = (fTag and 0x07).toInt()
                                    when {
                                        fNum == 1 && fWire == 0 -> dims.add(readVarint())
                                        fNum == 1 && fWire == 2 -> {
                                            val pLen = readVarint()
                                            val pEnd = bytesRead + pLen
                                            while (bytesRead < pEnd) dims.add(readVarint())
                                        }
                                        (fNum == 9 || fNum == 4) && fWire == 2 -> {
                                            val dLen = readVarint().toInt()
                                            if (dLen == 512 * 512 * 4) {
                                                val raw = ByteArray(dLen)
                                                var off = 0
                                                while (off < dLen) {
                                                    val r = stream.read(raw, off, dLen - off)
                                                    if (r <= 0) break
                                                    off += r
                                                    bytesRead += r
                                                }
                                                val fl = FloatArray(512 * 512)
                                                ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(fl)
                                                rawFloats = fl
                                            } else {
                                                skipBytes(dLen.toLong())
                                            }
                                        }
                                        else -> skipWire(fWire)
                                    }
                                }
                                if (bytesRead < tEnd) skipBytes(tEnd - bytesRead)
                                if (dims == listOf(512L, 512L) && rawFloats != null) {
                                    lastMatch = rawFloats
                                }
                            } else {
                                skipBytes(tLen)
                            }
                        } else {
                            skipWire(gWire)
                        }
                    }
                } else {
                    skipWire(wire)
                }
            }
            return lastMatch
        }
    }
}
