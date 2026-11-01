package com.example.ui

import android.app.Application
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.example.ai.DiffusionModelSlot
import com.example.ai.OnnxModelManager
import com.example.ai.RealDeviceAndModelMetrics
import com.example.data.AppLanguage
import com.example.data.AppPreferencesRepository
import com.example.data.AppPreferencesState
import com.example.data.ContentSafetyMode
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

enum class BottomNavTab {
    HOME,
    FAVORITES,
    MY_CREATIONS,
    MODELS
}

enum class ActiveScreen {
    MAIN_TABS,
    SETTINGS,
    TEXT_TO_IMAGE,
    IMAGE_TO_IMAGE,
    CHARACTER_STUDIO,
    FACE_SWAP,
    IMAGE_EDITOR,
    MY_GALLERY
}

data class ImageGeneratorUiState(
    val activeScreen: ActiveScreen = ActiveScreen.MAIN_TABS,
    val selectedTab: BottomNavTab = BottomNavTab.HOME,
    val preferences: AppPreferencesState = AppPreferencesState(),
    val metrics: RealDeviceAndModelMetrics = RealDeviceAndModelMetrics(),
    val isImportingModels: Boolean = false,
    val importProgressText: String? = null,
    val statusToastMessage: String? = null,
    val promptDraft: String = "",
    val selectedInputPhotoUri: Uri? = null
)

private data class TransientUiState(
    val activeScreen: ActiveScreen = ActiveScreen.MAIN_TABS,
    val selectedTab: BottomNavTab = BottomNavTab.HOME,
    val metrics: RealDeviceAndModelMetrics = RealDeviceAndModelMetrics(),
    val isImportingModels: Boolean = false,
    val importProgressText: String? = null,
    val statusToastMessage: String? = null,
    val promptDraft: String = "",
    val selectedInputPhotoUri: Uri? = null
)

class AppViewModel(application: Application) : AndroidViewModel(application) {

    private val prefsRepo = AppPreferencesRepository(application)
    private val modelManager = OnnxModelManager(application)

    private val transientState = MutableStateFlow(
        TransientUiState(metrics = modelManager.inspectCurrentMetrics())
    )

    val uiState: StateFlow<ImageGeneratorUiState> = combine(
        prefsRepo.preferencesFlow,
        transientState
    ) { prefs, transient ->
        ImageGeneratorUiState(
            activeScreen = transient.activeScreen,
            selectedTab = transient.selectedTab,
            preferences = prefs,
            metrics = transient.metrics,
            isImportingModels = transient.isImportingModels,
            importProgressText = transient.importProgressText,
            statusToastMessage = transient.statusToastMessage,
            promptDraft = transient.promptDraft,
            selectedInputPhotoUri = transient.selectedInputPhotoUri
        )
    }.stateIn(
        scope = viewModelScope,
        started = SharingStarted.WhileSubscribed(5_000),
        initialValue = ImageGeneratorUiState(metrics = transientState.value.metrics)
    )

    fun refreshDeviceAndModelMetrics() {
        viewModelScope.launch {
            val fresh = withContext(Dispatchers.IO) {
                modelManager.inspectCurrentMetrics()
            }
            transientState.update { it.copy(metrics = fresh) }
        }
    }

    fun navigateToScreen(screen: ActiveScreen) {
        transientState.update { it.copy(activeScreen = screen) }
        refreshDeviceAndModelMetrics()
    }

    fun selectBottomTab(tab: BottomNavTab) {
        transientState.update {
            it.copy(
                activeScreen = ActiveScreen.MAIN_TABS,
                selectedTab = tab
            )
        }
        refreshDeviceAndModelMetrics()
    }

    fun navigateBackToHome() {
        transientState.update {
            if (it.activeScreen != ActiveScreen.MAIN_TABS) {
                it.copy(activeScreen = ActiveScreen.MAIN_TABS)
            } else {
                it.copy(selectedTab = BottomNavTab.HOME)
            }
        }
    }

    fun updatePromptDraft(text: String) {
        transientState.update { it.copy(promptDraft = text) }
    }

    fun updateSelectedInputPhoto(uri: Uri?) {
        transientState.update { it.copy(selectedInputPhotoUri = uri) }
    }

    fun clearToast() {
        transientState.update { it.copy(statusToastMessage = null) }
    }

    // Settings actions
    fun setResolution(resolution: String) = viewModelScope.launch {
        prefsRepo.setResolution(resolution)
    }

    fun setInferenceMode(mode: String) = viewModelScope.launch {
        prefsRepo.setInferenceMode(mode)
    }

    fun setOfflineMode(enabled: Boolean) = viewModelScope.launch {
        prefsRepo.setOfflineMode(enabled)
    }

    fun setLanguage(language: AppLanguage) = viewModelScope.launch {
        prefsRepo.setLanguage(language)
    }

    fun setDarkTheme(dark: Boolean) = viewModelScope.launch {
        prefsRepo.setDarkTheme(dark)
    }

    fun setNotificationsEnabled(enabled: Boolean) = viewModelScope.launch {
        prefsRepo.setNotificationsEnabled(enabled)
    }

    fun setContentFilterEnabled(enabled: Boolean) = viewModelScope.launch {
        prefsRepo.setContentFilterEnabled(enabled)
    }

    fun setSafetyMode(mode: ContentSafetyMode) = viewModelScope.launch {
        prefsRepo.setSafetyMode(mode)
    }

    fun updateCustomCategory(
        fashionGlamour: Boolean? = null,
        swimwearBeach: Boolean? = null,
        romanticPoses: Boolean? = null,
        classicalArtistic: Boolean? = null,
        fantasyAction: Boolean? = null
    ) = viewModelScope.launch {
        prefsRepo.updateCustomCategory(
            fashionGlamour = fashionGlamour,
            swimwearBeach = swimwearBeach,
            romanticPoses = romanticPoses,
            classicalArtistic = classicalArtistic,
            fantasyAction = fantasyAction
        )
    }

    // Model & Cache Management actions
    fun importAllModelsFromFolder(treeUri: Uri) {
        viewModelScope.launch {
            transientState.update {
                it.copy(
                    isImportingModels = true,
                    importProgressText = "Scanning selected folder...",
                    statusToastMessage = null
                )
            }
            val result = withContext(Dispatchers.IO) {
                modelManager.importAllModelsFromFolder(treeUri) { progress ->
                    transientState.update { s -> s.copy(importProgressText = progress) }
                }
            }
            val updatedMetrics = withContext(Dispatchers.IO) {
                modelManager.inspectCurrentMetrics()
            }
            result.fold(
                onSuccess = { imported ->
                    val msg = if (imported.isNotEmpty()) {
                        "Imported ${imported.size} ONNX model(s): ${imported.joinToString { it.canonicalFileName }}"
                    } else {
                        "No matching .onnx files (text_encoder, unet, vae_decoder, vae_encoder) found in that folder."
                    }
                    transientState.update {
                        it.copy(
                            metrics = updatedMetrics,
                            isImportingModels = false,
                            importProgressText = null,
                            statusToastMessage = msg
                        )
                    }
                },
                onFailure = { err ->
                    transientState.update {
                        it.copy(
                            metrics = updatedMetrics,
                            isImportingModels = false,
                            importProgressText = null,
                            statusToastMessage = "Folder import error: ${err.message}"
                        )
                    }
                }
            )
        }
    }

    fun importSingleSlot(uri: Uri, slot: DiffusionModelSlot) {
        viewModelScope.launch {
            transientState.update {
                it.copy(
                    isImportingModels = true,
                    importProgressText = "Importing ${slot.canonicalFileName}..."
                )
            }
            val result = withContext(Dispatchers.IO) {
                modelManager.importSingleSlotFromUri(uri, slot)
            }
            val updatedMetrics = withContext(Dispatchers.IO) {
                modelManager.inspectCurrentMetrics()
            }
            result.fold(
                onSuccess = { bytes ->
                    transientState.update {
                        it.copy(
                            metrics = updatedMetrics,
                            isImportingModels = false,
                            importProgressText = null,
                            statusToastMessage = "${slot.canonicalFileName} ready (${OnnxModelManager.formatBytes(bytes)})"
                        )
                    }
                },
                onFailure = { err ->
                    transientState.update {
                        it.copy(
                            metrics = updatedMetrics,
                            isImportingModels = false,
                            importProgressText = null,
                            statusToastMessage = "Import error: ${err.message}"
                        )
                    }
                }
            )
        }
    }

    fun deleteSlotModel(slot: DiffusionModelSlot) {
        viewModelScope.launch {
            withContext(Dispatchers.IO) { modelManager.deleteSlotModel(slot) }
            val updated = withContext(Dispatchers.IO) { modelManager.inspectCurrentMetrics() }
            transientState.update {
                it.copy(
                    metrics = updated,
                    statusToastMessage = "Removed ${slot.canonicalFileName}"
                )
            }
        }
    }

    fun clearCache() {
        viewModelScope.launch {
            val freed = withContext(Dispatchers.IO) { modelManager.clearAppCache() }
            val updated = withContext(Dispatchers.IO) { modelManager.inspectCurrentMetrics() }
            transientState.update {
                it.copy(
                    metrics = updated,
                    statusToastMessage = "Cleared ${OnnxModelManager.formatBytes(freed)} cache"
                )
            }
        }
    }

    fun unloadAiFromRam() {
        viewModelScope.launch {
            val released = withContext(Dispatchers.IO) { modelManager.unloadSessionsFromRam() }
            val updated = withContext(Dispatchers.IO) { modelManager.inspectCurrentMetrics() }
            transientState.update {
                it.copy(
                    metrics = updated,
                    statusToastMessage = if (released > 0) {
                        "Released $released ONNX session(s) from RAM"
                    } else {
                        "RAM is clean • Available RAM: ${OnnxModelManager.formatBytes(updated.deviceAvailableRamBytes)}"
                    }
                )
            }
        }
    }
}
