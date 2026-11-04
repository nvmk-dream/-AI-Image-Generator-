package com.example.ai

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import android.graphics.Bitmap
import android.graphics.Color
import java.nio.FloatBuffer
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

object AvatarPhotoCompositor {

    /**
     * Downsamples a `[imgW * imgH]` pixel mask (`0f..1f`) into `[latentW * latentH]` (`imgW/8 * imgH/8`)
     * with a 1-cell context halo so the UNet generates natural lighting & seam transitions.
     */
    fun downsampleMaskToLatentGrid(
        pixelMask: FloatArray,
        imgW: Int,
        imgH: Int,
        latentW: Int,
        latentH: Int
    ): FloatArray {
        val raw = FloatArray(latentW * latentH)
        val cellW = (imgW.toFloat() / latentW).coerceAtLeast(1f)
        val cellH = (imgH.toFloat() / latentH).coerceAtLeast(1f)

        for (ly in 0 until latentH) {
            val y0 = (ly * cellH).toInt().coerceIn(0, imgH - 1)
            val y1 = (((ly + 1) * cellH).toInt() - 1).coerceIn(y0, imgH - 1)
            for (lx in 0 until latentW) {
                val x0 = (lx * cellW).toInt().coerceIn(0, imgW - 1)
                val x1 = (((lx + 1) * cellW).toInt() - 1).coerceIn(x0, imgW - 1)
                var sum = 0f
                var count = 0
                for (py in y0..y1) {
                    val row = py * imgW
                    for (px in x0..x1) {
                        sum += pixelMask[row + px]
                        count++
                    }
                }
                raw[ly * latentW + lx] = if (count > 0) (sum / count).coerceIn(0f, 1f) else 0f
            }
        }

        // Soft 3x3 max-average context halo in latent space so boundary edges never clip abruptly
        val out = FloatArray(latentW * latentH)
        for (ly in 0 until latentH) {
            for (lx in 0 until latentW) {
                val center = raw[ly * latentW + lx]
                var localMax = center
                for (dy in -1..1) {
                    val ny = (ly + dy).coerceIn(0, latentH - 1)
                    for (dx in -1..1) {
                        val nx = (lx + dx).coerceIn(0, latentW - 1)
                        localMax = max(localMax, raw[ny * latentW + nx])
                    }
                }
                out[ly * latentW + lx] = (center * 0.65f + localMax * 0.35f).coerceIn(0f, 1f)
            }
        }
        return out
    }

    /**
     * Spatial 4-channel latent blending (`[1, 4, latentH, latentW]`):
     * `z = m * edited + (1 - m) * reference`
     */
    fun blendLatentStep(
        editedLatents: FloatArray,
        referenceLatents: FloatArray,
        latentMask: FloatArray
    ): FloatArray {
        val spatial = latentMask.size
        if (spatial == 0 || editedLatents.size != referenceLatents.size) return editedLatents
        val channels = editedLatents.size / spatial
        val blended = FloatArray(editedLatents.size)
        for (c in 0 until channels) {
            val chOff = c * spatial
            for (i in 0 until spatial) {
                val m = latentMask[i]
                val idx = chOff + i
                blended[idx] = m * editedLatents[idx] + (1f - m) * referenceLatents[idx]
            }
        }
        return blended
    }

    /**
     * Composites the newly generated DSLR region onto the 100% untouched original photograph
     * using border color-harmony matching + feathered sub-pixel alpha blending.
     */
    fun compositeIdentityLockedPhoto(
        original: Bitmap,
        generated: Bitmap,
        featheredMask: FloatArray
    ): Bitmap {
        val w = original.width
        val h = original.height
        val scaledGen = if (generated.width == w && generated.height == h) {
            generated
        } else {
            Bitmap.createScaledBitmap(generated, w, h, true)
        }

        val origPx = IntArray(w * h)
        val genPx = IntArray(w * h)
        original.getPixels(origPx, 0, w, 0, 0, w, h)
        scaledGen.getPixels(genPx, 0, w, 0, 0, w, h)
        if (scaledGen !== generated) scaledGen.recycle()

        // Measure subtle ambient exposure balance along the feathered transition band (0.15..0.55)
        var origR = 0.0
        var origG = 0.0
        var origB = 0.0
        var genR = 0.0
        var genG = 0.0
        var genB = 0.0
        var ringCount = 0

        for (i in featheredMask.indices) {
            val m = featheredMask[i]
            if (m in 0.15f..0.55f) {
                val op = origPx[i]
                val gp = genPx[i]
                origR += (op shr 16) and 0xFF
                origG += (op shr 8) and 0xFF
                origB += op and 0xFF
                genR += (gp shr 16) and 0xFF
                genG += (gp shr 8) and 0xFF
                genB += gp and 0xFF
                ringCount++
            }
        }

        // Gentle 35% exposure harmony clamp (0.86..1.14) so lighting matches without washing out detail
        val gainR = if (ringCount > 64 && genR > 1.0) (1.0 + ((origR / genR) - 1.0) * 0.35).coerceIn(0.86, 1.14).toFloat() else 1f
        val gainG = if (ringCount > 64 && genG > 1.0) (1.0 + ((origG / genG) - 1.0) * 0.35).coerceIn(0.86, 1.14).toFloat() else 1f
        val gainB = if (ringCount > 64 && genB > 1.0) (1.0 + ((origB / genB) - 1.0) * 0.35).coerceIn(0.86, 1.14).toFloat() else 1f

        val outPx = IntArray(w * h)
        for (i in outPx.indices) {
            val alpha = featheredMask.getOrElse(i) { 0f }.coerceIn(0f, 1f)
            if (alpha <= 0.004f) {
                outPx[i] = origPx[i]
            } else {
                val op = origPx[i]
                val gp = genPx[i]
                val orC = (op shr 16) and 0xFF
                val ogC = (op shr 8) and 0xFF
                val obC = op and 0xFF

                val grC = (((gp shr 16) and 0xFF) * gainR).coerceIn(0f, 255f)
                val ggC = (((gp shr 8) and 0xFF) * gainG).coerceIn(0f, 255f)
                val gbC = ((gp and 0xFF) * gainB).coerceIn(0f, 255f)

                val inv = 1f - alpha
                val r = (orC * inv + grC * alpha).roundToInt().coerceIn(0, 255)
                val g = (ogC * inv + ggC * alpha).roundToInt().coerceIn(0, 255)
                val b = (obC * inv + gbC * alpha).roundToInt().coerceIn(0, 255)
                outPx[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
            }
        }

        return Bitmap.createBitmap(outPx, w, h, Bitmap.Config.ARGB_8888)
    }

    /**
     * Executes an optional Slot 7 Face-Aging GAN / SAM ONNX model (`[1, 4, H, W]` with age channel
     * or `[1, 3, H, W]` + scalar age tensor) if uploaded by the user. Returns null if the custom
     * model uses an unsupported signature so the caller falls back seamlessly to Core 1A–6.
     */
    fun tryRunOptionalSlot7AgingModel(
        env: OrtEnvironment,
        session: OrtSession,
        sourceBitmap: Bitmap,
        targetAge: Int
    ): Bitmap? = runCatching {
        val inputEntries = session.inputInfo.entries.toList()
        if (inputEntries.isEmpty()) return null
        val primaryEntry = inputEntries.first()
        val primaryInfo = primaryEntry.value.info as? TensorInfo ?: return null
        val shape = primaryInfo.shape
        if (shape.size != 4) return null

        val channels = if (shape[1] in 3L..4L) shape[1].toInt() else return null
        val modelH = if (shape[2] > 0L) shape[2].toInt() else 256
        val modelW = if (shape[3] > 0L) shape[3].toInt() else 256

        val resized = Bitmap.createScaledBitmap(sourceBitmap, modelW, modelH, true)
        val px = IntArray(modelW * modelH)
        resized.getPixels(px, 0, modelW, 0, 0, modelW, modelH)
        if (resized !== sourceBitmap) resized.recycle()

        val plane = modelW * modelH
        val normAge = (targetAge.coerceIn(16, 80) / 100f)
        val inputFloats = FloatArray(channels * plane)
        for (i in 0 until plane) {
            val c = px[i]
            inputFloats[i] = (((c shr 16) and 0xFF) / 127.5f) - 1f
            inputFloats[plane + i] = (((c shr 8) and 0xFF) / 127.5f) - 1f
            inputFloats[2 * plane + i] = ((c and 0xFF) / 127.5f) - 1f
            if (channels == 4) {
                inputFloats[3 * plane + i] = normAge
            }
        }

        val inputs = LinkedHashMap<String, OnnxTensor>()
        try {
            inputs[primaryEntry.key] = OnnxTensor.createTensor(
                env,
                FloatBuffer.wrap(inputFloats),
                longArrayOf(1L, channels.toLong(), modelH.toLong(), modelW.toLong())
            )
            if (inputEntries.size >= 2) {
                val ageEntry = inputEntries[1]
                val ageShape = (ageEntry.value.info as? TensorInfo)?.shape ?: longArrayOf(1L)
                val ageTensorShape = if (ageShape.isEmpty()) longArrayOf(1L) else ageShape.map { if (it > 0) it else 1L }.toLongArray()
                val elemCount = ageTensorShape.fold(1L) { a, b -> a * b }.toInt().coerceAtLeast(1)
                val ageArr = FloatArray(elemCount) { normAge }
                inputs[ageEntry.key] = OnnxTensor.createTensor(env, FloatBuffer.wrap(ageArr), ageTensorShape)
            }

            session.run(inputs).use { res ->
                val outTensor = res[0] as? OnnxTensor ?: return null
                val outShape = outTensor.info.shape
                if (outShape.size != 4 || outShape[1] != 3L) return null
                val outH = outShape[2].toInt()
                val outW = outShape[3].toInt()
                val outPlane = outH * outW
                val buf = outTensor.floatBuffer
                buf.rewind()
                val floats = FloatArray(3 * outPlane)
                buf.get(floats, 0, min(floats.size, buf.remaining()))

                val outPx = IntArray(outPlane)
                for (i in 0 until outPlane) {
                    val r = (((floats[i] + 1f) * 0.5f).coerceIn(0f, 1f) * 255f).roundToInt()
                    val g = (((floats[outPlane + i] + 1f) * 0.5f).coerceIn(0f, 1f) * 255f).roundToInt()
                    val b = (((floats[2 * outPlane + i] + 1f) * 0.5f).coerceIn(0f, 1f) * 255f).roundToInt()
                    outPx[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
                }
                val rawOut = Bitmap.createBitmap(outPx, outW, outH, Bitmap.Config.ARGB_8888)
                return if (outW == sourceBitmap.width && outH == sourceBitmap.height) {
                    rawOut
                } else {
                    val scaled = Bitmap.createScaledBitmap(rawOut, sourceBitmap.width, sourceBitmap.height, true)
                    rawOut.recycle()
                    scaled
                }
            }
        } finally {
            inputs.values.forEach { runCatching { it.close() } }
        }
    }.getOrNull()

    /**
     * Generates a translucent glowing preview overlay bitmap for the interactive Mask Canvas.
     */
    fun buildMaskOverlayBitmap(
        width: Int,
        height: Int,
        featheredMask: FloatArray,
        tab: AvatarCustomTab
    ): Bitmap {
        val tintRgb = when (tab) {
            AvatarCustomTab.AGE -> Triple(255, 160, 64)       // Warm Amber glow for Face-Aging mask
            AvatarCustomTab.BODY_ABS -> Triple(0, 229, 255)   // Electric Cyan glow for Torso/Abs mask
            AvatarCustomTab.HAIRSTYLE -> Triple(224, 64, 251) // Neon Magenta glow for Hair Crown mask
            AvatarCustomTab.OUTFIT -> Triple(105, 240, 174)   // Emerald Mint glow for Garment/Outfit mask
        }
        val px = IntArray(width * height)
        for (i in px.indices) {
            val m = featheredMask.getOrElse(i) { 0f }.coerceIn(0f, 1f)
            if (m > 0.04f) {
                val a = (m * 145f).roundToInt().coerceIn(0, 175)
                px[i] = Color.argb(a, tintRgb.first, tintRgb.second, tintRgb.third)
            } else {
                px[i] = Color.TRANSPARENT
            }
        }
        return Bitmap.createBitmap(px, width, height, Bitmap.Config.ARGB_8888)
    }
}
