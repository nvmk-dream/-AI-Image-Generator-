package com.example.ai

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Build
import java.io.File
import java.io.FileOutputStream
import kotlin.math.max
import kotlin.math.min

/**
 * Offline Bitmap loader, EXIF-aware decoder, Auto Center-Crop resizer,
 * and [-1, 1] RGB tensor converter for vae_encoder.onnx (Phase 5 Image-to-Image).
 */
object ImageBitmapPreprocessor {

    fun loadAndCenterCropFromUri(
        context: Context,
        uri: Uri,
        targetWidth: Int,
        targetHeight: Int
    ): Bitmap {
        val localFile = if (uri.scheme == "file" && !uri.path.isNullOrBlank()) {
            File(uri.path!!)
        } else {
            null
        }
        val decoded = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            val source = if (localFile != null && localFile.exists()) {
                ImageDecoder.createSource(localFile)
            } else {
                ImageDecoder.createSource(context.contentResolver, uri)
            }
            ImageDecoder.decodeBitmap(source) { decoder, info, _ ->
                decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
                decoder.isMutableRequired = false
                val srcW = info.size.width.coerceAtLeast(1)
                val srcH = info.size.height.coerceAtLeast(1)
                val maxTarget = max(targetWidth, targetHeight) * 2
                val minSide = min(srcW, srcH)
                if (minSide > maxTarget) {
                    val scale = maxTarget.toFloat() / minSide.toFloat()
                    decoder.setTargetSize(
                        (srcW * scale).toInt().coerceAtLeast(targetWidth),
                        (srcH * scale).toInt().coerceAtLeast(targetHeight)
                    )
                }
            }
        } else {
            val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
            if (localFile != null && localFile.exists()) {
                BitmapFactory.decodeFile(localFile.absolutePath, bounds)
            } else {
                context.contentResolver.openInputStream(uri)?.use { stream ->
                    BitmapFactory.decodeStream(stream, null, bounds)
                }
            }
            var sampleSize = 1
            val minDim = min(bounds.outWidth.coerceAtLeast(1), bounds.outHeight.coerceAtLeast(1))
            val desired = max(targetWidth, targetHeight) * 2
            while (minDim / (sampleSize * 2) >= desired) {
                sampleSize *= 2
            }
            val decodeOpts = BitmapFactory.Options().apply {
                inSampleSize = sampleSize
                inPreferredConfig = Bitmap.Config.ARGB_8888
            }
            if (localFile != null && localFile.exists()) {
                BitmapFactory.decodeFile(localFile.absolutePath, decodeOpts)
                    ?: error("Unable to decode saved reference image.")
            } else {
                context.contentResolver.openInputStream(uri)?.use { stream ->
                    BitmapFactory.decodeStream(stream, null, decodeOpts)
                } ?: error("Unable to open selected reference image.")
            }
        }

        val cropped = centerCropAndResize(decoded, targetWidth, targetHeight)
        if (cropped !== decoded) {
            decoded.recycle()
        }
        return cropped
    }

    /**
     * Center-crops [src] to match [targetWidth]:[targetHeight] aspect ratio without stretching,
     * then smoothly scales to exact [targetWidth] x [targetHeight] in ARGB_8888.
     */
    fun centerCropAndResize(
        src: Bitmap,
        targetWidth: Int,
        targetHeight: Int
    ): Bitmap {
        val srcW = src.width.coerceAtLeast(1)
        val srcH = src.height.coerceAtLeast(1)
        val targetAspect = targetWidth.toFloat() / targetHeight.toFloat()
        val srcAspect = srcW.toFloat() / srcH.toFloat()

        val cropW: Int
        val cropH: Int
        val cropX: Int
        val cropY: Int

        if (srcAspect > targetAspect) {
            // Source is wider than target -> crop left & right
            cropH = srcH
            cropW = (srcH * targetAspect).toInt().coerceIn(1, srcW)
            cropX = ((srcW - cropW) / 2).coerceAtLeast(0)
            cropY = 0
        } else {
            // Source is taller than target -> crop top & bottom
            cropW = srcW
            cropH = (srcW / targetAspect).toInt().coerceIn(1, srcH)
            cropX = 0
            cropY = ((srcH - cropH) / 2).coerceAtLeast(0)
        }

        val sub = if (cropX == 0 && cropY == 0 && cropW == srcW && cropH == srcH) {
            src
        } else {
            Bitmap.createBitmap(src, cropX, cropY, cropW, cropH)
        }

        val scaled = if (sub.width == targetWidth && sub.height == targetHeight) {
            sub
        } else {
            Bitmap.createScaledBitmap(sub, targetWidth, targetHeight, true).also {
                if (sub !== src && it !== sub) sub.recycle()
            }
        }

        return if (scaled.config == Bitmap.Config.ARGB_8888) {
            scaled
        } else {
            scaled.copy(Bitmap.Config.ARGB_8888, false).also {
                if (scaled !== src) scaled.recycle()
            }
        }
    }

    /**
     * Converts ARGB_8888 [bitmap] into float tensor normalized to [-1.0f, 1.0f]
     * for Stable Diffusion / LCM vae_encoder.onnx.
     */
    fun bitmapToNormalizedFloats(bitmap: Bitmap, isNhwc: Boolean = false): FloatArray {
        val w = bitmap.width
        val h = bitmap.height
        val pixelCount = w * h
        val pixels = IntArray(pixelCount)
        bitmap.getPixels(pixels, 0, w, 0, 0, w, h)
        return pixelsToNormalizedFloats(pixels, isNhwc)
    }

    fun pixelsToNormalizedFloats(pixels: IntArray, isNhwc: Boolean = false): FloatArray {
        val pixelCount = pixels.size
        val out = FloatArray(pixelCount * 3)
        if (isNhwc) {
            for (i in 0 until pixelCount) {
                val px = pixels[i]
                val r = ((px ushr 16) and 0xFF) / 127.5f - 1.0f
                val g = ((px ushr 8) and 0xFF) / 127.5f - 1.0f
                val b = (px and 0xFF) / 127.5f - 1.0f
                val base = i * 3
                out[base] = r
                out[base + 1] = g
                out[base + 2] = b
            }
        } else {
            // Standard NCHW: [1, 3, H, W]
            for (i in 0 until pixelCount) {
                val px = pixels[i]
                out[i] = ((px ushr 16) and 0xFF) / 127.5f - 1.0f
                out[pixelCount + i] = ((px ushr 8) and 0xFF) / 127.5f - 1.0f
                out[2 * pixelCount + i] = (px and 0xFF) / 127.5f - 1.0f
            }
        }
        return out
    }

    fun saveReferencePreview(context: Context, bitmap: Bitmap): File {
        val dir = File(context.cacheDir, "img2img_sources").apply { if (!exists()) mkdirs() }
        // Keep cache tidy (max 8 recent previews)
        dir.listFiles()?.sortedByDescending { it.lastModified() }?.drop(7)?.forEach {
            runCatching { it.delete() }
        }
        val file = File(dir, "ref_${System.currentTimeMillis()}.png")
        FileOutputStream(file).use { out ->
            bitmap.compress(Bitmap.CompressFormat.PNG, 100, out)
            out.flush()
        }
        return file
    }
}
