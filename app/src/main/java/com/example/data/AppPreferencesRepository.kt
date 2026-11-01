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

data class CustomContentCategories(
    val fashionGlamour: Boolean = true,
    val swimwearBeach: Boolean = true,
    val romanticPoses: Boolean = true,
    val classicalArtistic: Boolean = true,
    val fantasyAction: Boolean = false
)

data class AppPreferencesState(
    val resolution: String = "512 × 512",
    val inferenceMode: String = "CPU (Auto)",
    val offlineMode: Boolean = true,
    val language: AppLanguage = AppLanguage.ENGLISH,
    val darkTheme: Boolean = false,
    val notificationsEnabled: Boolean = true,
    val contentFilterEnabled: Boolean = true,
    val safetyMode: ContentSafetyMode = ContentSafetyMode.BALANCED,
    val customCategories: CustomContentCategories = CustomContentCategories()
) {
    val isTamil: Boolean
        get() = language == AppLanguage.TAMIL

    fun activeFilterSummary(): String {
        if (!contentFilterEnabled) {
            return if (isTamil) {
                "விருப்ப ஃபில்டர்: OFF • முழு சுதந்திர கலை உருவாக்கம்"
            } else {
                "Content Filter: OFF • Open Creative Studio"
            }
        }
        return when (safetyMode) {
            ContentSafetyMode.STRICT -> if (isTamil) {
                "Strict Mode: பொதுவான உருவப்படங்கள், இயற்கை & பாரம்பரிய உடைகள்"
            } else {
                "Strict Mode: General portraits, nature & classic outfits"
            }
            ContentSafetyMode.BALANCED -> if (isTamil) {
                "Balanced Mode: கலை, Fashion, Glamour & நாகரிகமான கலைப்படங்கள்"
            } else {
                "Balanced Mode: Art, Fashion, Glamour & tasteful creative portraits"
            }
            ContentSafetyMode.CUSTOM -> {
                val enabledList = buildList {
                    if (customCategories.fashionGlamour) add(if (isTamil) "Fashion & Glamour" else "Fashion & Glamour")
                    if (customCategories.swimwearBeach) add(if (isTamil) "Beachwear" else "Beachwear")
                    if (customCategories.romanticPoses) add(if (isTamil) "Romantic" else "Romantic")
                    if (customCategories.classicalArtistic) add(if (isTamil) "Fine Art" else "Fine Art")
                    if (customCategories.fantasyAction) add(if (isTamil) "Action" else "Action")
                }
                if (enabledList.isEmpty()) {
                    if (isTamil) "Custom Mode: பொதுவான படங்கள் மட்டும்" else "Custom Mode: Standard portraits only"
                } else {
                    val prefix = if (isTamil) "Custom அனுமதி: " else "Custom Allowed: "
                    prefix + enabledList.joinToString(", ")
                }
            }
        }
    }
}

class AppPreferencesRepository(private val context: Context) {

    private object Keys {
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
    }

    val preferencesFlow: Flow<AppPreferencesState> = context.appDataStore.data.map { prefs ->
        AppPreferencesState(
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
            )
        )
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
