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
    val savedUserImagePrompt: String = ""
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
            savedUserImagePrompt = prefs[Keys.SAVED_USER_IMAGE_PROMPT] ?: ""
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

    suspend fun saveUserImagePrompt(userPrompt: String) {
        context.appDataStore.edit { prefs ->
            prefs[Keys.SAVED_USER_IMAGE_PROMPT] = userPrompt
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
