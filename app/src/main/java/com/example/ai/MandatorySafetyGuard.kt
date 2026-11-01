package com.example.ai

import java.util.Locale

object MandatorySafetyGuard {

    private val minorIndicators = listOf(
        "child", "children", "kid", "kids", "minor", "underage", "toddler", "infant",
        "preteen", "loli", "shota", "schoolgirl", "schoolboy",
        "குழந்தை", "சிறுவன்", "சிறுமி", "சிறார்", "பள்ளிப் பெண்"
    )

    private val explicitSexualIndicators = listOf(
        "nude", "naked", "nsfw", "explicit", "sex", "sexual", "erotic", "porn",
        "topless", "undress", "strip",
        "நிர்வாண", "ஆபாச", "காம", "பாலியல்"
    )

    private val minorUnsafeIntimateIndicators = explicitSexualIndicators + listOf(
        "lingerie", "bikini", "seductive", "sensual"
    )

    private val prohibitedCoreTerms = listOf(
        "csam", "child porn", "non-consensual", "revenge porn", "deepfake porn",
        "without consent", "rape", "sexual assault", "self-harm", "suicide", "terrorist attack",
        "சம்மதமில்லாத", "பாலியல் வன்கொடுமை", "தற்கொலை"
    )

    data class CheckResult(
        val allowed: Boolean,
        val messageEn: String? = null
    )

    fun validateRequest(
        userPrompt: String,
        hasRealPersonReferencePhoto: Boolean = false
    ): CheckResult {
        val text = userPrompt.lowercase(Locale.US)

        if (prohibitedCoreTerms.any { text.contains(it) }) {
            return CheckResult(
                allowed = false,
                messageEn = "Please adjust your prompt and try again."
            )
        }

        val mentionsMinor = minorIndicators.any { word ->
            Regex("""\b${Regex.escape(word)}\b""").containsMatchIn(text) ||
                (word.any { it.code in 0x0B80..0x0BFF } && text.contains(word))
        }

        if (mentionsMinor && minorUnsafeIntimateIndicators.any { text.contains(it) }) {
            return CheckResult(
                allowed = false,
                messageEn = "Please adjust your prompt and try again."
            )
        }

        if (hasRealPersonReferencePhoto && explicitSexualIndicators.any { text.contains(it) }) {
            return CheckResult(
                allowed = false,
                messageEn = "Please adjust your prompt and try again."
            )
        }

        return CheckResult(allowed = true)
    }
}
