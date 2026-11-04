package com.example.ai

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.graphics.PointF
import android.graphics.RectF
import android.media.FaceDetector
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.framework.image.ByteBufferExtractor
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker
import com.google.mediapipe.tasks.vision.imagesegmenter.ImageSegmenter
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.ceil
import kotlin.math.exp
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

data class MediaPipeTasksStatus(
    val faceLandmarkerInstalled: Boolean = false,
    val poseLandmarkerInstalled: Boolean = false,
    val multiclassSegmenterInstalled: Boolean = false,
    val totalInstalledBytes: Long = 0L
) {
    val allTasksReady: Boolean
        get() = faceLandmarkerInstalled && poseLandmarkerInstalled && multiclassSegmenterInstalled

    val installedCount: Int
        get() = (if (faceLandmarkerInstalled) 1 else 0) +
            (if (poseLandmarkerInstalled) 1 else 0) +
            (if (multiclassSegmenterInstalled) 1 else 0)
}

/**
 * Normalized (0..1) anatomical & semantic analysis extracted on-device by MediaPipe Tasks Vision
 * (`FaceLandmarker` + `PoseLandmarker` + `ImageSegmenter` multiclass).
 */
data class AvatarVisionAnalysis(
    val imageWidth: Int,
    val imageHeight: Int,
    val usedMediaPipeFace: Boolean,
    val usedMediaPipePose: Boolean,
    val usedMediaPipeSegmenter: Boolean,
    val faceDetected: Boolean,
    val poseDetected: Boolean,
    val faceBoxNorm: RectF,
    val faceOvalNorm: List<PointF>,
    val leftEyeCenterNorm: PointF?,
    val rightEyeCenterNorm: PointF?,
    val leftShoulderNorm: PointF?,
    val rightShoulderNorm: PointF?,
    val leftHipNorm: PointF?,
    val rightHipNorm: PointF?,
    val handCentersNorm: List<PointF>,
    /**
     * Per-pixel semantic category from MediaPipe `selfie_multiclass_256x256.tflite`:
     * 0 = Background, 1 = Hair, 2 = Body-Skin, 3 = Face-Skin, 4 = Clothes, 5 = Accessories.
     * Null when falling back to offline geometric + FaceDetector landmarks.
     */
    val categoryWidth: Int = 0,
    val categoryHeight: Int = 0,
    val categoryMask: ByteArray? = null
)

/**
 * Official MediaPipe Tasks Vision engine (`FaceLandmarker`, `PoseLandmarker`, `ImageSegmenter`)
 * storing `.task`/`.tflite` bundles persistently inside `IG_Models/mediapipe_tasks/`.
 * Includes automatic offline fallback (`android.media.FaceDetector` + anatomical geometry)
 * so Avatar Customization works 100% offline at all times without crashing.
 */
class MediaPipeAvatarVisionEngine(
    private val context: Context,
    private val modelManager: OnnxModelManager
) {
    val visionTasksDir: File
        get() = File(modelManager.modelsRootDir, VISION_FOLDER_NAME).apply {
            if (!exists()) mkdirs()
        }

    val faceTaskFile: File get() = File(visionTasksDir, FACE_TASK_FILENAME)
    val poseTaskFile: File get() = File(visionTasksDir, POSE_TASK_FILENAME)
    val segmenterModelFile: File get() = File(visionTasksDir, MULTICLASS_SEG_FILENAME)

    fun inspectTasksStatus(): MediaPipeTasksStatus {
        val fOk = faceTaskFile.exists() && faceTaskFile.length() > 500_000L
        val pOk = poseTaskFile.exists() && poseTaskFile.length() > 500_000L
        val sOk = segmenterModelFile.exists() && segmenterModelFile.length() > 200_000L
        val bytes = (if (fOk) faceTaskFile.length() else 0L) +
            (if (pOk) poseTaskFile.length() else 0L) +
            (if (sOk) segmenterModelFile.length() else 0L)
        return MediaPipeTasksStatus(
            faceLandmarkerInstalled = fOk,
            poseLandmarkerInstalled = pOk,
            multiclassSegmenterInstalled = sOk,
            totalInstalledBytes = bytes
        )
    }

    suspend fun downloadVisionTasksIfNeeded(
        onProgress: (String, Int) -> Unit
    ): Result<MediaPipeTasksStatus> = withContext(Dispatchers.IO) {
        runCatching {
            val targets = listOf(
                Triple("MediaPipe Face Landmarker", FACE_TASK_URL, faceTaskFile),
                Triple("MediaPipe Pose Landmarker", POSE_TASK_URL, poseTaskFile),
                Triple("MediaPipe Multiclass Segmenter", MULTICLASS_SEG_URL, segmenterModelFile)
            )
            targets.forEachIndexed { idx, (label, url, destFile) ->
                currentCoroutineContext().ensureActive()
                if (destFile.exists() && destFile.length() > 250_000L) {
                    val donePct = (((idx + 1) * 100) / targets.size)
                    onProgress("$label ready ✓", donePct)
                } else {
                    val basePct = (idx * 100) / targets.size
                    downloadSingleTaskAsset(url, destFile) { filePct ->
                        val overall = (basePct + (filePct / targets.size)).coerceIn(1, 99)
                        onProgress("Downloading $label ($filePct%)...", overall)
                    }
                }
            }
            onProgress("MediaPipe Vision Tasks Ready (3/3) ✓", 100)
            inspectTasksStatus()
        }
    }

    private suspend fun downloadSingleTaskAsset(
        sourceUrl: String,
        destFile: File,
        onPercent: (Int) -> Unit
    ) {
        val partFile = File(destFile.parentFile, "${destFile.name}.download_part")
        var currentUrl = sourceUrl
        var redirects = 0
        while (redirects < 6) {
            currentCoroutineContext().ensureActive()
            val conn = (URL(currentUrl).openConnection() as HttpURLConnection).apply {
                connectTimeout = 20_000
                readTimeout = 30_000
                instanceFollowRedirects = true
                setRequestProperty("User-Agent", "ImageGenerator-Android/2.3")
            }
            try {
                val code = conn.responseCode
                if (code in setOf(301, 302, 303, 307, 308)) {
                    val loc = conn.getHeaderField("Location") ?: error("Redirect missing Location")
                    currentUrl = URL(URL(currentUrl), loc).toString()
                    redirects++
                    continue
                }
                if (code !in 200..299) {
                    error("HTTP $code downloading ${destFile.name}")
                }
                val totalLen = conn.contentLengthLong
                conn.inputStream.use { input ->
                    FileOutputStream(partFile, false).use { output ->
                        val buf = ByteArray(128 * 1024)
                        var copied = 0L
                        while (true) {
                            currentCoroutineContext().ensureActive()
                            val r = input.read(buf)
                            if (r <= 0) break
                            output.write(buf, 0, r)
                            copied += r
                            if (totalLen > 0L) {
                                onPercent(((copied * 100L) / totalLen).toInt().coerceIn(1, 99))
                            }
                        }
                    }
                }
                if (partFile.length() < 100_000L) {
                    partFile.delete()
                    error("Downloaded ${destFile.name} was incomplete.")
                }
                if (destFile.exists()) destFile.delete()
                if (!partFile.renameTo(destFile)) {
                    partFile.copyTo(destFile, overwrite = true)
                    partFile.delete()
                }
                return
            } catch (ce: CancellationException) {
                throw ce
            } finally {
                conn.disconnect()
            }
        }
        error("Too many redirects downloading ${destFile.name}")
    }

    /**
     * Runs MediaPipe FaceLandmarker + PoseLandmarker + Multiclass ImageSegmenter sequentially
     * on an ARGB_8888 bitmap (closing each native graph immediately to conserve RAM).
     */
    suspend fun analyzePhoto(sourceBitmap: Bitmap): AvatarVisionAnalysis = withContext(Dispatchers.Default) {
        val argb = if (sourceBitmap.config == Bitmap.Config.ARGB_8888) {
            sourceBitmap
        } else {
            sourceBitmap.copy(Bitmap.Config.ARGB_8888, false)
        }
        val w = argb.width.coerceAtLeast(1)
        val h = argb.height.coerceAtLeast(1)
        val status = inspectTasksStatus()

        var usedMpFace = false
        var usedMpPose = false
        var usedMpSeg = false
        var faceDetected = false
        var poseDetected = false

        var faceBox = RectF(0.28f, 0.10f, 0.72f, 0.48f)
        val faceOval = mutableListOf<PointF>()
        var leftEye: PointF? = null
        var rightEye: PointF? = null

        var leftShoulder: PointF? = null
        var rightShoulder: PointF? = null
        var leftHip: PointF? = null
        var rightHip: PointF? = null
        val handPoints = mutableListOf<PointF>()

        var catW = 0
        var catH = 0
        var catBytes: ByteArray? = null

        // 1. MediaPipe FaceLandmarker (478 3D Face Mesh points)
        if (status.faceLandmarkerInstalled) {
            runCatching {
                val mpImage = BitmapImageBuilder(argb).build()
                val baseOpts = BaseOptions.builder()
                    .setModelAssetBuffer(mapFileReadOnly(faceTaskFile))
                    .build()
                val opts = FaceLandmarker.FaceLandmarkerOptions.builder()
                    .setBaseOptions(baseOpts)
                    .setRunningMode(RunningMode.IMAGE)
                    .setNumFaces(1)
                    .build()
                FaceLandmarker.createFromOptions(context, opts).use { landmarker ->
                    val res = landmarker.detect(mpImage)
                    val lmList = res.faceLandmarks().firstOrNull()
                    if (!lmList.isNullOrEmpty()) {
                        usedMpFace = true
                        faceDetected = true
                        var minX = 1f
                        var minY = 1f
                        var maxX = 0f
                        var maxY = 0f
                        for (pt in lmList) {
                            minX = min(minX, pt.x())
                            minY = min(minY, pt.y())
                            maxX = max(maxX, pt.x())
                            maxY = max(maxY, pt.y())
                        }
                        faceBox = RectF(
                            minX.coerceIn(0f, 1f),
                            minY.coerceIn(0f, 1f),
                            maxX.coerceIn(0f, 1f),
                            maxY.coerceIn(0f, 1f)
                        )
                        for (idx in FACE_OVAL_INDICES) {
                            if (idx < lmList.size) {
                                faceOval.add(PointF(lmList[idx].x().coerceIn(0f, 1f), lmList[idx].y().coerceIn(0f, 1f)))
                            }
                        }
                        if (lmList.size > 386) {
                            leftEye = PointF(lmList[159].x(), lmList[159].y())
                            rightEye = PointF(lmList[386].x(), lmList[386].y())
                        }
                    }
                }
            }
        }

        // Offline Fallback FaceDetector if MediaPipe Face task isn't downloaded yet
        if (!faceDetected) {
            runCatching {
                val rgb565 = argb.copy(Bitmap.Config.RGB_565, false)
                val detector = FaceDetector(rgb565.width, rgb565.height, 1)
                val faces = arrayOfNulls<FaceDetector.Face>(1)
                val found = detector.findFaces(rgb565, faces)
                val f = faces.firstOrNull()
                if (found > 0 && f != null) {
                    val mid = PointF()
                    f.getMidPoint(mid)
                    val eyeDist = f.eyesDistance()
                    if (eyeDist > 8f) {
                        faceDetected = true
                        val cx = (mid.x / w.toFloat()).coerceIn(0.1f, 0.9f)
                        val cy = (mid.y / h.toFloat()).coerceIn(0.1f, 0.85f)
                        val rx = ((eyeDist * 1.18f) / w.toFloat()).coerceIn(0.08f, 0.42f)
                        val ryTop = ((eyeDist * 1.25f) / h.toFloat()).coerceIn(0.08f, 0.40f)
                        val ryBot = ((eyeDist * 1.55f) / h.toFloat()).coerceIn(0.10f, 0.45f)
                        faceBox = RectF(
                            (cx - rx).coerceIn(0f, 1f),
                            (cy - ryTop).coerceIn(0f, 1f),
                            (cx + rx).coerceIn(0f, 1f),
                            (cy + ryBot).coerceIn(0f, 1f)
                        )
                        leftEye = PointF((cx - (eyeDist * 0.5f / w)).coerceIn(0f, 1f), cy)
                        rightEye = PointF((cx + (eyeDist * 0.5f / w)).coerceIn(0f, 1f), cy)
                    }
                }
                rgb565.recycle()
            }
        }

        // 2. MediaPipe PoseLandmarker (33 BlazePose body keypoints)
        if (status.poseLandmarkerInstalled) {
            runCatching {
                val mpImage = BitmapImageBuilder(argb).build()
                val baseOpts = BaseOptions.builder()
                    .setModelAssetBuffer(mapFileReadOnly(poseTaskFile))
                    .build()
                val opts = PoseLandmarker.PoseLandmarkerOptions.builder()
                    .setBaseOptions(baseOpts)
                    .setRunningMode(RunningMode.IMAGE)
                    .setNumPoses(1)
                    .build()
                PoseLandmarker.createFromOptions(context, opts).use { poseLandmarker ->
                    val res = poseLandmarker.detect(mpImage)
                    val pList = res.landmarks().firstOrNull()
                    if (pList != null && pList.size >= 25) {
                        usedMpPose = true
                        poseDetected = true
                        leftShoulder = PointF(pList[11].x().coerceIn(0f, 1f), pList[11].y().coerceIn(0f, 1f))
                        rightShoulder = PointF(pList[12].x().coerceIn(0f, 1f), pList[12].y().coerceIn(0f, 1f))
                        leftHip = PointF(pList[23].x().coerceIn(0f, 1f), pList[23].y().coerceIn(0f, 1f))
                        rightHip = PointF(pList[24].x().coerceIn(0f, 1f), pList[24].y().coerceIn(0f, 1f))
                        for (hIdx in listOf(15, 16, 17, 18, 19, 20)) {
                            if (hIdx < pList.size) {
                                val hx = pList[hIdx].x()
                                val hy = pList[hIdx].y()
                                if (hx in 0.01f..0.99f && hy in 0.01f..0.99f) {
                                    handPoints.add(PointF(hx, hy))
                                }
                            }
                        }
                    }
                }
            }
        }

        // 3. MediaPipe Multiclass ImageSegmenter (Background, Hair, Body-Skin, Face-Skin, Clothes, Others)
        if (status.multiclassSegmenterInstalled) {
            runCatching {
                val mpImage = BitmapImageBuilder(argb).build()
                val baseOpts = BaseOptions.builder()
                    .setModelAssetBuffer(mapFileReadOnly(segmenterModelFile))
                    .build()
                val opts = ImageSegmenter.ImageSegmenterOptions.builder()
                    .setBaseOptions(baseOpts)
                    .setRunningMode(RunningMode.IMAGE)
                    .setOutputCategoryMask(true)
                    .setOutputConfidenceMasks(false)
                    .build()
                ImageSegmenter.createFromOptions(context, opts).use { segmenter ->
                    val segResult = segmenter.segment(mpImage)
                    val optMask = segResult.categoryMask()
                    if (optMask.isPresent) {
                        val maskImg = optMask.get()
                        catW = maskImg.width
                        catH = maskImg.height
                        val buf = ByteBufferExtractor.extract(maskImg)
                        buf.rewind()
                        val raw = ByteArray(catW * catH)
                        buf.get(raw, 0, min(raw.size, buf.remaining()))
                        catBytes = raw
                        usedMpSeg = true
                    }
                }
            }
        }

        if (argb !== sourceBitmap) {
            argb.recycle()
        }

        AvatarVisionAnalysis(
            imageWidth = w,
            imageHeight = h,
            usedMediaPipeFace = usedMpFace,
            usedMediaPipePose = usedMpPose,
            usedMediaPipeSegmenter = usedMpSeg,
            faceDetected = faceDetected,
            poseDetected = poseDetected,
            faceBoxNorm = faceBox,
            faceOvalNorm = faceOval,
            leftEyeCenterNorm = leftEye,
            rightEyeCenterNorm = rightEye,
            leftShoulderNorm = leftShoulder,
            rightShoulderNorm = rightShoulder,
            leftHipNorm = leftHip,
            rightHipNorm = rightHip,
            handCentersNorm = handPoints,
            categoryWidth = catW,
            categoryHeight = catH,
            categoryMask = catBytes
        )
    }

    /**
     * Builds a smooth, feathered `FloatArray` mask (`[outWidth * outHeight]` in `0f..1f`)
     * tailored to the active module (`AGE`, `BODY_ABS`, `HAIRSTYLE`, `OUTFIT`) + user brush/eraser strokes,
     * while enforcing Face & Hand Identity Protection.
     */
    fun buildFeatheredMask(
        analysis: AvatarVisionAnalysis,
        params: AvatarCustomizationParams,
        userStrokes: List<MaskBrushStroke>,
        outWidth: Int,
        outHeight: Int
    ): FloatArray {
        val maskBmp = Bitmap.createBitmap(outWidth, outHeight, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(maskBmp)
        canvas.drawColor(Color.BLACK)

        val whitePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.WHITE
            style = Paint.Style.FILL
        }
        val blackPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.BLACK
            style = Paint.Style.FILL
        }

        val fb = RectF(
            analysis.faceBoxNorm.left * outWidth,
            analysis.faceBoxNorm.top * outHeight,
            analysis.faceBoxNorm.right * outWidth,
            analysis.faceBoxNorm.bottom * outHeight
        )
        val faceW = fb.width().coerceAtLeast(outWidth * 0.18f)
        val faceH = fb.height().coerceAtLeast(outHeight * 0.18f)
        val faceCx = fb.centerX()

        val hasSeg = analysis.categoryMask != null && analysis.categoryWidth > 0 && analysis.categoryHeight > 0

        // 1. Seed base mask from MediaPipe Multiclass Segmentation if available
        if (hasSeg) {
            val cat = analysis.categoryMask!!
            val cw = analysis.categoryWidth
            val ch = analysis.categoryHeight
            val px = IntArray(outWidth * outHeight)
            for (y in 0 until outHeight) {
                val sy = ((y.toFloat() / outHeight) * ch).toInt().coerceIn(0, ch - 1)
                val rowOff = sy * cw
                val outOff = y * outWidth
                for (x in 0 until outWidth) {
                    val sx = ((x.toFloat() / outWidth) * cw).toInt().coerceIn(0, cw - 1)
                    val cls = cat[rowOff + sx].toInt() and 0xFF
                    // Multiclass indices: 0=BG, 1=Hair, 2=Body-skin, 3=Face-skin, 4=Clothes, 5=Accessories
                    val active = when (params.activeTab) {
                        AvatarCustomTab.AGE -> cls == 3 || (cls == 1 && params.silverHairRatio >= 0.20f)
                        AvatarCustomTab.BODY_ABS -> (cls == 2 || cls == 4) && y > (fb.bottom - faceH * 0.05f)
                        AvatarCustomTab.HAIRSTYLE -> cls == 1
                        AvatarCustomTab.OUTFIT -> cls == 4 || (cls == 2 && y > fb.bottom + faceH * 0.18f)
                    }
                    px[outOff + x] = if (active) Color.WHITE else Color.BLACK
                }
            }
            maskBmp.setPixels(px, 0, outWidth, 0, 0, outWidth, outHeight)
        }

        // 2. Anatomical Landmark Envelope Augmentation & Identity Lock Subtraction
        when (params.activeTab) {
            AvatarCustomTab.AGE -> {
                if (!hasSeg) {
                    drawFaceOvalOrRect(canvas, analysis, fb, outWidth, outHeight, scale = 1.06f, paint = whitePaint)
                    if (params.silverHairRatio >= 0.25f) {
                        val hairRect = RectF(
                            (faceCx - faceW * 0.80f).coerceAtLeast(0f),
                            (fb.top - faceH * 0.45f).coerceAtLeast(0f),
                            (faceCx + faceW * 0.80f).coerceAtMost(outWidth.toFloat()),
                            fb.top + faceH * 0.35f
                        )
                        canvas.drawOval(hairRect, whitePaint)
                    }
                }
            }

            AvatarCustomTab.BODY_ABS -> {
                val torsoRect = computeTorsoRect(analysis, fb, faceW, faceH, outWidth, outHeight)
                if (!hasSeg) {
                    canvas.drawRoundRect(torsoRect, faceW * 0.28f, faceW * 0.28f, whitePaint)
                } else {
                    // Ensure lower abdomen region inside pose torso is well-covered
                    val coreRect = RectF(
                        torsoRect.left + torsoRect.width() * 0.12f,
                        torsoRect.top + torsoRect.height() * 0.15f,
                        torsoRect.right - torsoRect.width() * 0.12f,
                        torsoRect.bottom
                    )
                    canvas.drawRoundRect(coreRect, faceW * 0.22f, faceW * 0.22f, whitePaint)
                }
                // HARD IDENTITY LOCK: Protect entire Face + Head above chin!
                drawFaceOvalOrRect(canvas, analysis, fb, outWidth, outHeight, scale = 1.12f, paint = blackPaint)
            }

            AvatarCustomTab.HAIRSTYLE -> {
                // Expand hair growth envelope so long waves / curls / braids have room around head & shoulders
                val extraLen = (0.25f + params.hairLengthScale * 0.85f) * faceH
                val sideHalo = (0.68f + params.hairVolumeDensity * 0.36f) * faceW
                val crownTop = (fb.top - faceH * (0.42f + params.hairVolumeDensity * 0.22f)).coerceAtLeast(0f)
                val crownOval = RectF(
                    (faceCx - sideHalo).coerceAtLeast(0f),
                    crownTop,
                    (faceCx + sideHalo).coerceAtMost(outWidth.toFloat()),
                    (fb.bottom + extraLen).coerceAtMost(outHeight.toFloat())
                )
                if (!hasSeg || params.hairLengthScale > 0.45f || params.hairVolumeDensity > 0.55f) {
                    canvas.drawOval(crownOval, whitePaint)
                }
                // HARD IDENTITY LOCK: Subtract inner Face Oval (eyes, nose, lips, cheeks stay 100% original!)
                val innerFaceScale = (0.86f + (params.faceIdentityLock - 0.5f) * 0.22f).coerceIn(0.82f, 0.98f)
                val protectedFaceRect = RectF(
                    fb.left + faceW * (1f - innerFaceScale) * 0.5f,
                    fb.top + faceH * 0.14f,
                    fb.right - faceW * (1f - innerFaceScale) * 0.5f,
                    fb.bottom
                )
                drawFaceOvalOrRect(canvas, analysis, protectedFaceRect, outWidth, outHeight, scale = innerFaceScale, paint = blackPaint)
            }

            AvatarCustomTab.OUTFIT -> {
                val garmentRect = computeGarmentRect(analysis, fb, faceW, faceH, outWidth, outHeight)
                if (!hasSeg) {
                    canvas.drawRoundRect(garmentRect, faceW * 0.30f, faceW * 0.30f, whitePaint)
                } else {
                    // Bridge slight collar/shoulder gaps inside pose envelope
                    val torsoBridge = RectF(
                        garmentRect.left + garmentRect.width() * 0.14f,
                        garmentRect.top + garmentRect.height() * 0.08f,
                        garmentRect.right - garmentRect.width() * 0.14f,
                        garmentRect.bottom
                    )
                    canvas.drawRoundRect(torsoBridge, faceW * 0.22f, faceW * 0.22f, whitePaint)
                }
                // HARD IDENTITY LOCK: Protect Face + Neck chin + Detected Hands!
                drawFaceOvalOrRect(canvas, analysis, fb, outWidth, outHeight, scale = 1.10f, paint = blackPaint)
                val handRadius = max(faceW * 0.26f, 18f)
                for (hp in analysis.handCentersNorm) {
                    canvas.drawCircle(hp.x * outWidth, hp.y * outHeight, handRadius, blackPaint)
                }
            }
        }

        // 3. Apply Interactive User Brush / Eraser Touch Strokes
        for (stroke in userStrokes) {
            if (stroke.normalizedPoints.isEmpty()) continue
            val rPx = (stroke.normalizedRadius * min(outWidth, outHeight)).coerceAtLeast(6f)
            val strokePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
                color = if (stroke.isErase) Color.BLACK else Color.WHITE
                style = Paint.Style.STROKE
                strokeCap = Paint.Cap.ROUND
                strokeJoin = Paint.Join.ROUND
                strokeWidth = rPx * 2f
            }
            if (stroke.normalizedPoints.size == 1) {
                val p0 = stroke.normalizedPoints.first()
                canvas.drawCircle(
                    p0.first * outWidth,
                    p0.second * outHeight,
                    rPx,
                    if (stroke.isErase) blackPaint else whitePaint
                )
            } else {
                val path = Path()
                val first = stroke.normalizedPoints.first()
                path.moveTo(first.first * outWidth, first.second * outHeight)
                for (i in 1 until stroke.normalizedPoints.size) {
                    val pt = stroke.normalizedPoints[i]
                    path.lineTo(pt.first * outWidth, pt.second * outHeight)
                }
                canvas.drawPath(path, strokePaint)
            }
        }

        // 4. Convert to FloatArray [0..1], apply Morphological Expand/Shrink (`maskExpandPx`) + Separable Gaussian Feather (`maskFeatherPx`)
        val rawPixels = IntArray(outWidth * outHeight)
        maskBmp.getPixels(rawPixels, 0, outWidth, 0, 0, outWidth, outHeight)
        maskBmp.recycle()

        var mask = FloatArray(outWidth * outHeight) { idx ->
            ((rawPixels[idx] shr 16) and 0xFF) / 255f
        }

        val expandRadius = params.maskExpandPx.roundToInt().coerceIn(-10, 26)
        if (expandRadius != 0) {
            mask = morphExpandOrShrink(mask, outWidth, outHeight, expandRadius)
        }

        // Subtle eye-iris attenuation in AGE mode so pupil gaze & identity stay crisp
        if (params.activeTab == AvatarCustomTab.AGE && params.facialBoneConsistency > 0.65f) {
            val eyeAtten = (1.15f - params.facialBoneConsistency * 0.55f).coerceIn(0.45f, 0.85f)
            val eyeR = faceW * 0.11f
            listOfNotNull(analysis.leftEyeCenterNorm, analysis.rightEyeCenterNorm).forEach { eye ->
                attenuateCircle(mask, outWidth, outHeight, eye.x * outWidth, eye.y * outHeight, eyeR, eyeAtten)
            }
        }

        val featherSigma = (params.maskFeatherPx * (outWidth / 512f)).coerceIn(2f, 28f)
        return gaussianBlurSeparable(mask, outWidth, outHeight, featherSigma)
    }

    private fun drawFaceOvalOrRect(
        canvas: Canvas,
        analysis: AvatarVisionAnalysis,
        fallbackRect: RectF,
        outWidth: Int,
        outHeight: Int,
        scale: Float,
        paint: Paint
    ) {
        if (analysis.faceOvalNorm.size >= 8) {
            var cx = 0f
            var cy = 0f
            analysis.faceOvalNorm.forEach {
                cx += it.x * outWidth
                cy += it.y * outHeight
            }
            cx /= analysis.faceOvalNorm.size
            cy /= analysis.faceOvalNorm.size
            val path = Path()
            analysis.faceOvalNorm.forEachIndexed { idx, pt ->
                val px = cx + ((pt.x * outWidth) - cx) * scale
                val py = cy + ((pt.y * outHeight) - cy) * scale
                if (idx == 0) path.moveTo(px, py) else path.lineTo(px, py)
            }
            path.close()
            canvas.drawPath(path, paint)
        } else {
            val cx = fallbackRect.centerX()
            val cy = fallbackRect.centerY()
            val hw = (fallbackRect.width() * 0.5f) * scale
            val hh = (fallbackRect.height() * 0.5f) * scale
            canvas.drawOval(RectF(cx - hw, cy - hh, cx + hw, cy + hh), paint)
        }
    }

    private fun computeTorsoRect(
        analysis: AvatarVisionAnalysis,
        fb: RectF,
        faceW: Float,
        faceH: Float,
        outW: Int,
        outH: Int
    ): RectF {
        val ls = analysis.leftShoulderNorm
        val rs = analysis.rightShoulderNorm
        val lh = analysis.leftHipNorm
        val rh = analysis.rightHipNorm
        if (ls != null && rs != null) {
            val minSx = min(ls.x, rs.x) * outW
            val maxSx = max(ls.x, rs.x) * outW
            val shoulderSpan = (maxSx - minSx).coerceAtLeast(faceW * 1.5f)
            val topY = max(fb.bottom + faceH * 0.06f, (min(ls.y, rs.y) * outH) - faceH * 0.08f)
            val bottomY = if (lh != null && rh != null) {
                (max(lh.y, rh.y) * outH + faceH * 0.25f).coerceAtMost(outH.toFloat())
            } else {
                outH.toFloat()
            }
            return RectF(
                (minSx - shoulderSpan * 0.18f).coerceAtLeast(0f),
                topY.coerceAtMost(outH * 0.88f),
                (maxSx + shoulderSpan * 0.18f).coerceAtMost(outW.toFloat()),
                bottomY.coerceAtLeast(topY + faceH * 0.5f)
            )
        }
        val cx = fb.centerX()
        val halfSpan = faceW * 1.15f
        val topY = (fb.bottom + faceH * 0.10f).coerceAtMost(outH * 0.82f)
        return RectF(
            (cx - halfSpan).coerceAtLeast(0f),
            topY,
            (cx + halfSpan).coerceAtMost(outW.toFloat()),
            outH.toFloat()
        )
    }

    private fun computeGarmentRect(
        analysis: AvatarVisionAnalysis,
        fb: RectF,
        faceW: Float,
        faceH: Float,
        outW: Int,
        outH: Int
    ): RectF {
        val torso = computeTorsoRect(analysis, fb, faceW, faceH, outW, outH)
        val extraSleeve = faceW * 0.35f
        return RectF(
            (torso.left - extraSleeve).coerceAtLeast(0f),
            (fb.bottom + faceH * 0.04f).coerceAtMost(torso.top),
            (torso.right + extraSleeve).coerceAtMost(outW.toFloat()),
            outH.toFloat()
        )
    }

    private fun attenuateCircle(
        mask: FloatArray,
        w: Int,
        h: Int,
        cx: Float,
        cy: Float,
        radius: Float,
        minFactor: Float
    ) {
        val r = ceil(radius).toInt()
        val x0 = (cx.toInt() - r).coerceAtLeast(0)
        val x1 = (cx.toInt() + r).coerceAtMost(w - 1)
        val y0 = (cy.toInt() - r).coerceAtLeast(0)
        val y1 = (cy.toInt() + r).coerceAtMost(h - 1)
        for (y in y0..y1) {
            for (x in x0..x1) {
                val d = hypot(x - cx, y - cy)
                if (d < radius) {
                    val t = (d / radius).coerceIn(0f, 1f)
                    val factor = minFactor + (1f - minFactor) * t
                    mask[y * w + x] *= factor
                }
            }
        }
    }

    private fun morphExpandOrShrink(
        src: FloatArray,
        w: Int,
        h: Int,
        radiusPx: Int
    ): FloatArray {
        val r = kotlin.math.abs(radiusPx).coerceIn(1, 26)
        val dilate = radiusPx > 0
        val horiz = FloatArray(w * h)
        for (y in 0 until h) {
            val row = y * w
            for (x in 0 until w) {
                var acc = if (dilate) 0f else 1f
                val x0 = (x - r).coerceAtLeast(0)
                val x1 = (x + r).coerceAtMost(w - 1)
                for (kx in x0..x1) {
                    val v = src[row + kx]
                    acc = if (dilate) max(acc, v) else min(acc, v)
                }
                horiz[row + x] = acc
            }
        }
        val out = FloatArray(w * h)
        for (x in 0 until w) {
            for (y in 0 until h) {
                var acc = if (dilate) 0f else 1f
                val y0 = (y - r).coerceAtLeast(0)
                val y1 = (y + r).coerceAtMost(h - 1)
                for (ky in y0..y1) {
                    val v = horiz[ky * w + x]
                    acc = if (dilate) max(acc, v) else min(acc, v)
                }
                out[y * w + x] = acc
            }
        }
        return out
    }

    private fun gaussianBlurSeparable(
        src: FloatArray,
        w: Int,
        h: Int,
        sigma: Float
    ): FloatArray {
        val radius = ceil(sigma * 2.2f).toInt().coerceIn(2, 36)
        val kernel = FloatArray(radius * 2 + 1)
        var sum = 0f
        val twoSigmaSq = 2f * sigma * sigma
        for (i in -radius..radius) {
            val wt = exp(-(i * i) / twoSigmaSq)
            kernel[i + radius] = wt
            sum += wt
        }
        for (i in kernel.indices) kernel[i] /= sum

        val temp = FloatArray(w * h)
        for (y in 0 until h) {
            val row = y * w
            for (x in 0 until w) {
                var acc = 0f
                for (k in -radius..radius) {
                    val sx = (x + k).coerceIn(0, w - 1)
                    acc += src[row + sx] * kernel[k + radius]
                }
                temp[row + x] = acc
            }
        }
        val out = FloatArray(w * h)
        for (x in 0 until w) {
            for (y in 0 until h) {
                var acc = 0f
                for (k in -radius..radius) {
                    val sy = (y + k).coerceIn(0, h - 1)
                    acc += temp[sy * w + x] * kernel[k + radius]
                }
                out[y * w + x] = acc.coerceIn(0f, 1f)
            }
        }
        return out
    }

    private fun mapFileReadOnly(file: File): MappedByteBuffer {
        FileInputStream(file).use { fis ->
            return fis.channel.map(FileChannel.MapMode.READ_ONLY, 0L, file.length())
        }
    }

    companion object {
        const val VISION_FOLDER_NAME = "mediapipe_tasks"
        const val FACE_TASK_FILENAME = "face_landmarker.task"
        const val POSE_TASK_FILENAME = "pose_landmarker_lite.task"
        const val MULTICLASS_SEG_FILENAME = "selfie_multiclass_256x256.tflite"

        const val FACE_TASK_URL =
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        const val POSE_TASK_URL =
            "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
        const val MULTICLASS_SEG_URL =
            "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite"

        // MediaPipe 478 Face Mesh outer oval contour indices in clockwise order
        private val FACE_OVAL_INDICES = intArrayOf(
            10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
            397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
            172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
        )
    }
}
