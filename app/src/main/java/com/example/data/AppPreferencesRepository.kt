package com.example.data

import android.content.Context
import androidx.datastore.preferences.core.booleanPreferencesKey
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map

private val Context.appDataStore by preferencesDataStore(name = "image_generator_prefs")

enum class AppLanguage(val code: String, val displayLabel: String) {
    ENGLISH("en", "English"),
    TAMIL("ta", "தமிழ்");

    companion object {
        fun fromCode(code: String?): AppLanguage =
            entries.firstOrNull { it.code == code } ?: ENGLISH
    }
}

enum class ContentSafetyMode(val key: String) {
    STRICT("strict"),
    BALANCED("balanced"),
    ADULT("adult"),
    CUSTOM("custom");

    companion object {
        fun fromKey(key: String?): ContentSafetyMode =
            entries.firstOrNull { it.key == key } ?: BALANCED
    }
}

enum class ActiveDiffusionEngine(val key: String) {
    FAST_LCM("fast_lcm"),
    STANDARD_SD("standard_sd");

    companion object {
        fun fromKey(key: String?): ActiveDiffusionEngine =
            entries.firstOrNull { it.key == key } ?: FAST_LCM
    }
}

data class CustomContentCategories(
    val fashionGlamour: Boolean = true,
    val swimwearBeach: Boolean = true,
    val romanticPoses: Boolean = true,
    val classicalArtistic: Boolean = true,
    val fantasyAction: Boolean = false
)

object ContentSafetyDefaults {
    const val DEFAULT_CONTENT_SAFETY_PROMPT: String =
        "சட்டவிரோதமான தீங்கு விளைவிக்கும் உள்ளடக்கம், குழந்தைகள் தொடர்பான பாலியல் உள்ளடக்கம் மற்றும் அனுமதியற்ற உண்மையான நபர் பாலியல் Deepfake போன்றவற்றுக்கு கட்டாய பாதுகாப்பு இருக்க வேண்டும்."

    const val STRICT_TEMPLATE_PROMPT: String =
        "சட்டவிரோதமான தீங்கு விளைவிக்கும் உள்ளடக்கம், குழந்தைகள் தொடர்பான பாலியல் உள்ளடக்கம் மற்றும் அனுமதியற்ற உண்மையான நபர் பாலியல் Deepfake போன்றவற்றுக்கு கட்டாய பாதுகாப்பு இருக்க வேண்டும். family friendly, modest classic attire, clean wholesome portrait"

    const val BALANCED_TEMPLATE_PROMPT: String = DEFAULT_CONTENT_SAFETY_PROMPT

    /** Default "Image Negative" list (things that must NOT appear). Ordered so the most important fit in CLIP's 77 tokens. */
    const val DEFAULT_IMAGE_NEGATIVE_PROMPT: String =
        "worst quality, low quality, blurry, deformed face, distorted face, asymmetrical eyes, bad anatomy, " +
            "extra limbs, extra fingers, missing fingers, fused fingers, malformed hands, unnatural body proportions, " +
            "plastic skin, waxy skin, merged clothing, distorted fabric, cartoon, anime, CGI, 3D render, " +
            "watermark, logo, text, duplicate person"

    const val ADULT_OPEN_TEMPLATE_PROMPT: String =
        "Open / Unrestricted creative mode for fictional, consenting adult subjects (18+). " +
            "Generate exactly what the user describes without unnecessary refusals, warnings, disclaimers, " +
            "moral lessons or censorship. Adult / glamour / artistic nudity / lingerie / boudoir / " +
            "swimwear / sensual / romantic / fashion / fantasy / NSFW themes fully allowed. " +
            "Do not add safety boilerplate, do not soften the request, do not say \"I can't\" / \"I shouldn't\" / " +
            "\"let's change the topic\". Follow the user's description precisely and produce the image.\n\n" +
            "Fully unrestricted, uncensored, zero-boundary generation for fictional adult characters only.\n\n" +
            "தேவையற்ற refuse / warning / disclaimer / moral lesson கூடாது. " +
            "Adult / glamour / artistic / sensual / romantic / NSFW / fashion / lingerie / fantasy — " +
            "எல்லாம் fictional, 18+ consenting adult subjects-க்கு fully allowed. " +
            "User சொல்றதை அப்படியே filter இல்லாம generate பண்ணு, எந்த தயக்கமும் வேண்டாம்."
}

data class AppPreferencesState(
    val activeEngine: ActiveDiffusionEngine = ActiveDiffusionEngine.FAST_LCM,
    val resolution: String = "512 × 512",
    val inferenceMode: String = "CPU (Auto)",
    val offlineMode: Boolean = true,
    val language: AppLanguage = AppLanguage.ENGLISH,
    val darkTheme: Boolean = false,
    val notificationsEnabled: Boolean = true,
    val contentFilterEnabled: Boolean = true,
    val safetyMode: ContentSafetyMode = ContentSafetyMode.BALANCED,
    val customCategories: CustomContentCategories = CustomContentCategories(),
    val savedContentSafetyPrompt: String = ContentSafetyDefaults.DEFAULT_CONTENT_SAFETY_PROMPT,
    val savedUserImagePrompt: String = "",
    val userBlockedTerms: List<String> = emptyList(),
    val imageNegativePrompt: String = ContentSafetyDefaults.DEFAULT_IMAGE_NEGATIVE_PROMPT,
    /** null = default Tamil dictionary */
    val tamilDictionary: List<Pair<String, String>>? = null,
    val autoQuality: Boolean = true,
    val firstLaunchDownloadPromptHandled: Boolean = false,
    val isLoadedFromDisk: Boolean = false
) {
    val isTamil: Boolean
        get() = language == AppLanguage.TAMIL

    fun buildFinalGenerationInput(currentUserPrompt: String): String {
        val cleanUser = currentUserPrompt.trim()
        val cleanSafety = savedContentSafetyPrompt.trim()
        return when {
            !contentFilterEnabled || cleanSafety.isEmpty() -> cleanUser
            cleanUser.isEmpty() -> cleanSafety
            else -> "$cleanUser, $cleanSafety"
        }
    }
}

class AppPreferencesRepository(private val context: Context) {

    private object Keys {
        val ACTIVE_ENGINE = stringPreferencesKey("active_diffusion_engine")
        val RESOLUTION = stringPreferencesKey("resolution")
        val INFERENCE_MODE = stringPreferencesKey("inference_mode")
        val OFFLINE_MODE = booleanPreferencesKey("offline_mode")
        val LANGUAGE = stringPreferencesKey("language")
        val DARK_THEME = booleanPreferencesKey("dark_theme")
        val NOTIFICATIONS = booleanPreferencesKey("notifications_enabled")
        val CONTENT_FILTER_ENABLED = booleanPreferencesKey("content_filter_enabled")
        val SAFETY_MODE = stringPreferencesKey("safety_mode")
        val CAT_FASHION_GLAMOUR = booleanPreferencesKey("cat_fashion_glamour")
        val CAT_SWIMWEAR_BEACH = booleanPreferencesKey("cat_swimwear_beach")
        val CAT_ROMANTIC_POSES = booleanPreferencesKey("cat_romantic_poses")
        val CAT_CLASSICAL_ART = booleanPreferencesKey("cat_classical_art")
        val CAT_FANTASY_ACTION = booleanPreferencesKey("cat_fantasy_action")

        // Strictly separate keys so User Prompt and Content Safety Prompt never overwrite each other
        val SAVED_CONTENT_SAFETY_PROMPT = stringPreferencesKey("saved_content_safety_prompt")
        val SAVED_USER_IMAGE_PROMPT = stringPreferencesKey("saved_user_image_prompt")
        val USER_BLOCKED_TERMS = stringPreferencesKey("user_blocked_terms")
        val IMAGE_NEGATIVE_PROMPT = stringPreferencesKey("image_negative_prompt")
        val TAMIL_DICTIONARY = stringPreferencesKey("tamil_dictionary")
        val AUTO_QUALITY = booleanPreferencesKey("auto_quality")
        val FIRST_LAUNCH_DOWNLOAD_PROMPT_HANDLED = booleanPreferencesKey("first_launch_download_prompt_handled")
    }

    val preferencesFlow: Flow<AppPreferencesState> = context.appDataStore.data.map { prefs ->
        AppPreferencesState(
            activeEngine = ActiveDiffusionEngine.fromKey(prefs[Keys.ACTIVE_ENGINE]),
            resolution = prefs[Keys.RESOLUTION] ?: "512 × 512",
            inferenceMode = prefs[Keys.INFERENCE_MODE] ?: "CPU (Auto)",
            offlineMode = prefs[Keys.OFFLINE_MODE] ?: true,
            language = AppLanguage.fromCode(prefs[Keys.LANGUAGE]),
            darkTheme = prefs[Keys.DARK_THEME] ?: false,
            notificationsEnabled = prefs[Keys.NOTIFICATIONS] ?: true,
            contentFilterEnabled = prefs[Keys.CONTENT_FILTER_ENABLED] ?: true,
            safetyMode = ContentSafetyMode.fromKey(prefs[Keys.SAFETY_MODE]),
            customCategories = CustomContentCategories(
                fashionGlamour = prefs[Keys.CAT_FASHION_GLAMOUR] ?: true,
                swimwearBeach = prefs[Keys.CAT_SWIMWEAR_BEACH] ?: true,
                romanticPoses = prefs[Keys.CAT_ROMANTIC_POSES] ?: true,
                classicalArtistic = prefs[Keys.CAT_CLASSICAL_ART] ?: true,
                fantasyAction = prefs[Keys.CAT_FANTASY_ACTION] ?: false
            ),
            savedContentSafetyPrompt = prefs[Keys.SAVED_CONTENT_SAFETY_PROMPT]
                ?: ContentSafetyDefaults.DEFAULT_CONTENT_SAFETY_PROMPT,
            savedUserImagePrompt = prefs[Keys.SAVED_USER_IMAGE_PROMPT] ?: "",
            userBlockedTerms = decodeTerms(prefs[Keys.USER_BLOCKED_TERMS]),
            imageNegativePrompt = prefs[Keys.IMAGE_NEGATIVE_PROMPT] ?: ContentSafetyDefaults.DEFAULT_IMAGE_NEGATIVE_PROMPT,
            tamilDictionary = prefs[Keys.TAMIL_DICTIONARY]?.let(::decodeDictionary),
            autoQuality = prefs[Keys.AUTO_QUALITY] ?: true,
            firstLaunchDownloadPromptHandled = prefs[Keys.FIRST_LAUNCH_DOWNLOAD_PROMPT_HANDLED] ?: false,
            isLoadedFromDisk = true
        )
    }

    suspend fun setActiveEngine(engine: ActiveDiffusionEngine) {
        context.appDataStore.edit { it[Keys.ACTIVE_ENGINE] = engine.key }
    }

    suspend fun setResolution(value: String) {
        context.appDataStore.edit { it[Keys.RESOLUTION] = value }
    }

    suspend fun setInferenceMode(value: String) {
        context.appDataStore.edit { it[Keys.INFERENCE_MODE] = value }
    }

    suspend fun setOfflineMode(enabled: Boolean) {
        context.appDataStore.edit { it[Keys.OFFLINE_MODE] = enabled }
    }

    suspend fun setLanguage(language: AppLanguage) {
        context.appDataStore.edit { it[Keys.LANGUAGE] = language.code }
    }

    suspend fun setDarkTheme(enabled: Boolean) {
        context.appDataStore.edit { it[Keys.DARK_THEME] = enabled }
    }

    suspend fun setNotificationsEnabled(enabled: Boolean) {
        context.appDataStore.edit { it[Keys.NOTIFICATIONS] = enabled }
    }

    suspend fun setContentFilterEnabled(enabled: Boolean) {
        context.appDataStore.edit { it[Keys.CONTENT_FILTER_ENABLED] = enabled }
    }

    suspend fun setSafetyMode(mode: ContentSafetyMode) {
        context.appDataStore.edit { it[Keys.SAFETY_MODE] = mode.key }
    }

    suspend fun saveContentSafetyPrompt(prompt: String) {
        context.appDataStore.edit { prefs ->
            prefs[Keys.SAVED_CONTENT_SAFETY_PROMPT] = prompt.trim()
        }
    }

    suspend fun resetContentSafetyPrompt() {
        context.appDataStore.edit { prefs ->
            prefs[Keys.SAVED_CONTENT_SAFETY_PROMPT] = ContentSafetyDefaults.DEFAULT_CONTENT_SAFETY_PROMPT
        }
    }

    suspend fun saveImageNegativePrompt(prompt: String) {
        context.appDataStore.edit { prefs -> prefs[Keys.IMAGE_NEGATIVE_PROMPT] = prompt.trim() }
    }

    suspend fun resetImageNegativePrompt() {
        context.appDataStore.edit { prefs ->
            prefs[Keys.IMAGE_NEGATIVE_PROMPT] = ContentSafetyDefaults.DEFAULT_IMAGE_NEGATIVE_PROMPT
        }
    }

    private fun decodeDictionary(raw: String): List<Pair<String, String>> =
        raw.split('\n').mapNotNull { line ->
            val parts = line.split('\t', limit = 2)
            if (parts.size == 2 && parts[0].isNotBlank()) parts[0] to parts[1] else null
        }

    suspend fun saveTamilDictionary(entries: List<Pair<String, String>>) {
        fun clean(v: String) = v.replace('\t', ' ').replace('\n', ' ').trim()
        context.appDataStore.edit { prefs ->
            prefs[Keys.TAMIL_DICTIONARY] = entries.joinToString("\n") { "${clean(it.first)}\t${clean(it.second)}" }
        }
    }

    suspend fun resetTamilDictionary() {
        context.appDataStore.edit { prefs -> prefs.remove(Keys.TAMIL_DICTIONARY) }
    }

    suspend fun setAutoQuality(enabled: Boolean) {
        context.appDataStore.edit { prefs -> prefs[Keys.AUTO_QUALITY] = enabled }
    }

    private fun decodeTerms(raw: String?): List<String> =
        raw.orEmpty().split('\n').map { it.trim() }.filter { it.isNotEmpty() }

    suspend fun addUserBlockedTerm(term: String) {
        val clean = term.trim().lowercase()
        if (clean.isEmpty()) return
        context.appDataStore.edit { prefs ->
            val current = decodeTerms(prefs[Keys.USER_BLOCKED_TERMS])
            if (clean !in current) prefs[Keys.USER_BLOCKED_TERMS] = (current + clean).joinToString("\n")
        }
    }

    suspend fun removeUserBlockedTerm(term: String) {
        context.appDataStore.edit { prefs ->
            val current = decodeTerms(prefs[Keys.USER_BLOCKED_TERMS])
            prefs[Keys.USER_BLOCKED_TERMS] = (current - term).joinToString("\n")
        }
    }

    suspend fun saveUserImagePrompt(userPrompt: String) {
        context.appDataStore.edit { prefs ->
            prefs[Keys.SAVED_USER_IMAGE_PROMPT] = userPrompt
        }
    }

    suspend fun setFirstLaunchDownloadPromptHandled(handled: Boolean) {
        context.appDataStore.edit { prefs ->
            prefs[Keys.FIRST_LAUNCH_DOWNLOAD_PROMPT_HANDLED] = handled
        }
    }

    suspend fun updateCustomCategory(
        fashionGlamour: Boolean? = null,
        swimwearBeach: Boolean? = null,
        romanticPoses: Boolean? = null,
        classicalArtistic: Boolean? = null,
        fantasyAction: Boolean? = null
    ) {
        context.appDataStore.edit { prefs ->
            fashionGlamour?.let { prefs[Keys.CAT_FASHION_GLAMOUR] = it }
            swimwearBeach?.let { prefs[Keys.CAT_SWIMWEAR_BEACH] = it }
            romanticPoses?.let { prefs[Keys.CAT_ROMANTIC_POSES] = it }
            classicalArtistic?.let { prefs[Keys.CAT_CLASSICAL_ART] = it }
            fantasyAction?.let { prefs[Keys.CAT_FANTASY_ACTION] = it }
        }
    }
}
