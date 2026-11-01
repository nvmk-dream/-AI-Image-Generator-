package com.example.ai

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.ColorMatrix
import android.graphics.ColorMatrixColorFilter
import android.graphics.Paint
import android.net.Uri

enum class EditorFilterPreset(
    val labelEn: String,
    val labelTa: String,
    val brightness: Float,
    val contrast: Float,
    val saturation: Float,
    val warmth: Float
) {
    ORIGINAL("Original", "அசல்", 0f, 1.0f, 1.0f, 0f),
    DREAMY_PINK("Dreamy Glow", "இளஞ்சிவப்பு ஒளி", 8f, 1.06f, 1.18f, 12f),
    GOLDEN_HOUR("Golden Hour", "தங்க மாலை", 6f, 1.08f, 1.22f, 26f),
    VIVID_CINEMA("Vivid Cinema", "சினிமா நிறம்", 2f, 1.18f, 1.30f, -6f),
    SOFT_PASTEL("Soft Pastel", "மென்மையான நிறம்", 12f, 0.94f, 0.86f, 8f),
    CLASSIC_BW("Classic B&W", "கருப்பு வெள்ளை", 4f, 1.15f, 0.0f, 0f)
}

data class ImageEditorAdjustments(
    val brightness: Float = 0f,     // -50f .. +50f
    val contrast: Float = 1.0f,     // 0.50f .. 1.50f
    val saturation: Float = 1.0f,   // 0.0f .. 2.0f
    val warmth: Float = 0f          // -50f .. +50f
)

object OfflineImageEditorEngine {

    /**
     * Builds a composite 4x5 Android ColorMatrix for Brightness, Contrast, Saturation, and Warmth
     * using pure Kotlin math (safe for both Android Canvas and JVM unit tests).
     */
    fun buildColorMatrixValues(adj: ImageEditorAdjustments): FloatArray {
        val sat = adj.saturation.coerceIn(0f, 2.5f)
        val invSat = 1.0f - sat
        val lumR = 0.213f * invSat
        val lumG = 0.715f * invSat
        val lumB = 0.072f * invSat

        val s00 = lumR + sat
        val s01 = lumG
        val s02 = lumB
        val s10 = lumR
        val s11 = lumG + sat
        val s12 = lumB
        val s20 = lumR
        val s21 = lumG
        val s22 = lumB + sat

        val c = adj.contrast.coerceIn(0.4f, 1.8f)
        val b = adj.brightness.coerceIn(-80f, 80f) * 2.2f
        val translate = (1f - c) * 127.5f + b

        val w = adj.warmth.coerceIn(-60f, 60f)
        val rScale = c * (1.0f + (w * 0.0022f))
        val gScale = c
        val bScale = c * (1.0f - (w * 0.0022f))

        return floatArrayOf(
            rScale * s00, rScale * s01, rScale * s02, 0f, translate + (w * 0.35f),
            gScale * s10, gScale * s11, gScale * s12, 0f, translate,
            bScale * s20, bScale * s21, bScale * s22, 0f, translate - (w * 0.35f),
            0f, 0f, 0f, 1f, 0f
        )
    }

    fun renderEditedBitmap(
        context: Context,
        sourceUri: Uri?,
        fallbackFilePath: String?,
        adjustments: ImageEditorAdjustments,
        maxDimension: Int = 1024
    ): Bitmap {
        val baseBitmap: Bitmap = when {
            sourceUri != null -> {
                ImageBitmapPreprocessor.loadAndCenterCropFromUri(
                    context = context,
                    uri = sourceUri,
                    targetWidth = maxDimension,
                    targetHeight = maxDimension
                )
            }
            !fallbackFilePath.isNullOrBlank() -> {
                android.graphics.BitmapFactory.decodeFile(fallbackFilePath)
                    ?: error("Could not decode image file.")
            }
            else -> error("Please choose an image first.")
        }

        return applyAdjustmentsToBitmap(baseBitmap, adjustments).also { out ->
            if (out !== baseBitmap) baseBitmap.recycle()
        }
    }

    fun applyAdjustmentsToBitmap(source: Bitmap, adjustments: ImageEditorAdjustments): Bitmap {
        val out = Bitmap.createBitmap(source.width, source.height, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(out)
        val paint = Paint(Paint.ANTI_ALIAS_FLAG or Paint.FILTER_BITMAP_FLAG).apply {
            colorFilter = ColorMatrixColorFilter(ColorMatrix(buildColorMatrixValues(adjustments)))
        }
        canvas.drawBitmap(source, 0f, 0f, paint)
        return out
    }
}
