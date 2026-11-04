package com.example.ai

import com.example.data.ContentSafetyDefaults
import java.util.Locale

object MandatorySafetyGuard {

    private val minorIndicators = listOf(
        "child", "children", "kid", "kids", "minor", "underage", "toddler", "infant",
        "preteen", "loli", "shota", "schoolgirl", "schoolboy",
        "குழந்தை", "சிறுவன்", "சிறுமி", "சிறார்", "பள்ளிப் பெண்"
    )

    // Keep only the two strict code-level blocks requested by the app owner:
    // 1) child-focused sexual content
    // 2) violence / gore / harmful attack content
    private val childSexualIndicators = listOf(
        "nude", "naked", "nsfw", "explicit", "erotic", "porn", "pornography",
        "sexual", "topless", "undress", "lingerie", "bikini", "seductive", "sensual",
        "நிர்வாண", "ஆபாச", "காம", "பாலியல்"
    )

    private val violenceIndicators = listOf(
        "murder", "kill", "killing", "massacre", "bomb", "explosion", "attack", "terrorist",
        "shooting", "stab", "stabbing", "behead", "decapitate", "bloodbath", "gore",
        "self-harm", "suicide", "assault", "rape", "violent", "weapon attack",
        "கொலை", "தற்கொலை", "சுட்டுக் கொலை", "வன்முறை", "போர்", "தாக்குதல்", "படுகொலை"
    )

    /** Extra words the user added in Settings → Content Safety. Only adds blocks, never removes built-in ones. */
    @Volatile
    var userExtraBlockedTerms: List<String> = emptyList()

    private fun firstMatch(text: String, terms: List<String>): String? =
        terms.firstOrNull { term -> containsWholeTerm(text, term) }

    data class CheckResult(
        val allowed: Boolean,
        val messageEn: String? = null
    )

    /**
     * Whole-word / whole-phrase matcher so normal English or Tamil-translated fashion words
     * never falsely collide with substrings like "rape", "strip", "sex", or "infant".
     */
    internal fun containsWholeTerm(normalizedLowerText: String, term: String): Boolean {
        val cleanTerm = term.trim().lowercase(Locale.US)
        if (cleanTerm.isEmpty()) return false

        val hasTamilScript = cleanTerm.any { it.code in 0x0B80..0x0BFF }
        return if (hasTamilScript) {
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

        val stripped = userPrompt
            .replace(ContentSafetyDefaults.STRICT_TEMPLATE_PROMPT, " ", ignoreCase = true)
            .replace(ContentSafetyDefaults.DEFAULT_CONTENT_SAFETY_PROMPT, " ", ignoreCase = true)
            .replace(ContentSafetyDefaults.ADULT_OPEN_TEMPLATE_PROMPT, " ", ignoreCase = true)
            .lowercase(Locale.US)

        val minorWord = firstMatch(stripped, minorIndicators)
        val childSexualWord = minorWord?.let { firstMatch(stripped, childSexualIndicators) }
        if (minorWord != null && childSexualWord != null) {
            return CheckResult(
                allowed = false,
                messageEn = "Blocked: child-focused sexual content is not allowed. " +
                    "(குழந்தை சார்ந்த பாலியல் / நிர்வாண / erotic உள்ளடக்கம் தடுக்கப்பட்டது.) Please change it and try again."
            )
        }

        val violentWord = firstMatch(stripped, violenceIndicators)
        if (violentWord != null) {
            return CheckResult(
                allowed = false,
                messageEn = "Blocked: violent / harmful / gore content is not allowed. " +
                    "(வன்முறை / தாக்குதல் / கொலை / தற்கொலை / இரத்தம் கொண்ட உள்ளடக்கம் தடுக்கப்பட்டது.) Please change it and try again."
            )
        }

        firstMatch(stripped, userExtraBlockedTerms)?.let { term ->
            return CheckResult(
                allowed = false,
                messageEn = "Blocked by your own block list: \"$term\". " +
                    "(நீங்க சேர்த்த block word.) Remove it in Settings → Content Safety → My Block Words."
            )
        }

        return CheckResult(allowed = true)
    }
}
