package com.example.ai

/**
 * "Auto (best)" defaults so non-technical users get a realistic, full-body image without touching sliders.
 */
object AutoQuality {
    const val RESOLUTION = "512 × 768"
    const val LCM_STEPS = 8

    private val styleChangeWords = listOf(
        "painting", "anime", "cartoon", "sketch", "watercolor", "illustration", "3d", "comic", "pixar",
        "ஓவியம்", "கார்ட்டூன்"
    )

    /** Outfit/background change keeps the person (55%); a full art-style change needs more (75%). */
    fun img2imgStrengthFor(prompt: String): Float =
        if (styleChangeWords.any { prompt.contains(it, ignoreCase = true) }) 0.75f else 0.55f
}
