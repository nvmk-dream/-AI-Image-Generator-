package com.example.ai

import com.example.data.ContentSafetyDefaults
import java.util.Locale

object MandatorySafetyGuard {

    private val minorIndicators = listOf(
        "child", "children", "kid", "kids", "minor", "underage", "toddler", "infant",
        "preteen", "loli", "shota", "schoolgirl", "schoolboy",
        "குழந்தை", "சிறுவன்", "சிறுமி", "சிறார்", "பள்ளிப் பெண்"
    )

    private val explicitNudityIndicators = listOf(
        "nude", "naked", "nsfw", "explicit", "erotic", "porn", "pornography",
        "நிர்வாண", "ஆபாச"
    )

    private val minorUnsafeIntimateIndicators = explicitNudityIndicators + listOf(
        "sexual", "topless", "undress", "lingerie", "bikini", "seductive", "sensual",
        "காம", "பாலியல்"
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

    /**
     * Whole-word / whole-phrase matcher so normal English or Tamil-translated DSLR & fashion words
     * (e.g., "drape", "trapezius", "striped", "pinstripe", "unisex", "shirtless", "infantry")
     * never falsely collide with substrings like "rape", "strip", "sex", or "infant".
     */
    internal fun containsWholeTerm(normalizedLowerText: String, term: String): Boolean {
        val cleanTerm = term.trim().lowercase(Locale.US)
        if (cleanTerm.isEmpty()) return false

        val hasTamilScript = cleanTerm.any { it.code in 0x0B80..0x0BFF }
        return if (hasTamilScript) {
            // Match at Tamil word start boundary (ignoring preceding spaces/punctuation)
            val pattern = Regex("""(?<![\u0B80-\u0BFF])${Regex.escape(cleanTerm)}""")
            pattern.containsMatchIn(normalizedLowerText)
        } else {
            val pattern = Regex("""(?<![a-z0-9])${Regex.escape(cleanTerm)}(?![a-z0-9])""")
            pattern.containsMatchIn(normalizedLowerText)
        }
    }

    fun validateRequest(
        userPrompt: String,
        hasRealPersonReferencePhoto: Boolean = false
    ): CheckResult {
        if (userPrompt.isBlank()) return CheckResult(allowed = true)

        // Strip built-in safety disclaimer templates first so safety boilerplate never self-triggers
        val stripped = userPrompt
            .replace(ContentSafetyDefaults.STRICT_TEMPLATE_PROMPT, " ", ignoreCase = true)
            .replace(ContentSafetyDefaults.DEFAULT_CONTENT_SAFETY_PROMPT, " ", ignoreCase = true)
            .lowercase(Locale.US)

        if (prohibitedCoreTerms.any { term -> containsWholeTerm(stripped, term) }) {
            return CheckResult(
                allowed = false,
                messageEn = "Please adjust your prompt and try again."
            )
        }

        val mentionsMinor = minorIndicators.any { word -> containsWholeTerm(stripped, word) }

        if (mentionsMinor && minorUnsafeIntimateIndicators.any { term -> containsWholeTerm(stripped, term) }) {
            return CheckResult(
                allowed = false,
                messageEn = "Please adjust your prompt and try again."
            )
        }

        if (hasRealPersonReferencePhoto && explicitNudityIndicators.any { term -> containsWholeTerm(stripped, term) }) {
            return CheckResult(
                allowed = false,
                messageEn = "Please adjust your prompt and try again."
            )
        }

        return CheckResult(allowed = true)
    }
}
