package com.example.ui

import android.app.Application
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.example.ai.DiffusionModelSlot
import com.example.ai.GenerationProgress
import com.example.ai.OnnxDiffusionEngine
import com.example.ai.OnnxModelManager
import com.example.ai.RealDeviceAndModelMetrics
import com.example.ai.SamplerType
import com.example.data.AppLanguage
import com.example.data.AppPreferencesRepository
import com.example.data.AppPreferencesState
import com.example.data.ContentSafetyMode
import com.example.data.CreationsRepository
import com.example.data.GeneratedCreation
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
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
    CONTENT_SAFETY_SETTINGS,
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
    val creations: List<GeneratedCreation> = emptyList(),
    val isImportingModels: Boolean = false,
    val importProgressText: String? = null,
    val onnxVerificationReport: String? = null,
    val isGenerating: Boolean = false,
    val generationProgress: GenerationProgress? = null,
    val generationErrorMessage: String? = null,
    val latestGeneratedCreation: GeneratedCreation? = null,
    val latestImg2ImgCreation: GeneratedCreation? = null,
    val latestImg2ImgSourcePath: String? = null,
    val statusToastMessage: String? = null,
    val userPromptInput: String = "",
    val selectedInputPhotoUri: Uri? = null
)

private data class TransientUiState(
    val activeScreen: ActiveScreen = ActiveScreen.MAIN_TABS,
    val selectedTab: BottomNavTab = BottomNavTab.HOME,
    val metrics: RealDeviceAndModelMetrics = RealDeviceAndModelMetrics(),
    val creations: List<GeneratedCreation> = emptyList(),
    val isImportingModels: Boolean = false,
    val importProgressText: String? = null,
    val onnxVerificationReport: String? = null,
    val isGenerating: Boolean = false,
    val generationProgress: GenerationProgress? = null,
    val generationErrorMessage: String? = null,
    val latestGeneratedCreation: GeneratedCreation? = null,
    val latestImg2ImgCreation: GeneratedCreation? = null,
    val latestImg2ImgSourcePath: String? = null,
    val statusToastMessage: String? = null,
    val userPromptInput: String? = null,
    val selectedInputPhotoUri: Uri? = null,
    val returnScreenForSafety: ActiveScreen = ActiveScreen.SETTINGS
)

class AppViewModel(application: Application) : AndroidViewModel(application) {

    private val prefsRepo = AppPreferencesRepository(application)
    private val modelManager = OnnxModelManager(application)
    private val creationsRepo = CreationsRepository(application)
    private val diffusionEngine = OnnxDiffusionEngine(application, modelManager)

    private var activeGenerationJob: Job? = null

    private val transientState = MutableStateFlow(
        TransientUiState(
            metrics = modelManager.inspectCurrentMetrics(),
            creations = creationsRepo.loadAllCreations()
        )
    )

    val uiState: StateFlow<ImageGeneratorUiState> = combine(
        prefsRepo.preferencesFlow,
        transientState
    ) { prefs, transient ->
        val effectiveUserPrompt = transient.userPromptInput ?: prefs.savedUserImagePrompt
        ImageGeneratorUiState(
            activeScreen = transient.activeScreen,
            selectedTab = transient.selectedTab,
            preferences = prefs,
            metrics = transient.metrics,
            creations = transient.creations,
            isImportingModels = transient.isImportingModels,
            importProgressText = transient.importProgressText,
            onnxVerificationReport = transient.onnxVerificationReport,
            isGenerating = transient.isGenerating,
            generationProgress = transient.generationProgress,
            generationErrorMessage = transient.generationErrorMessage,
            latestGeneratedCreation = transient.latestGeneratedCreation
                ?: transient.creations.firstOrNull(),
            latestImg2ImgCreation = transient.latestImg2ImgCreation,
            latestImg2ImgSourcePath = transient.latestImg2ImgSourcePath,
            statusToastMessage = transient.statusToastMessage,
            userPromptInput = effectiveUserPrompt,
            selectedInputPhotoUri = transient.selectedInputPhotoUri
        )
    }.stateIn(
        scope = viewModelScope,
        started = SharingStarted.WhileSubscribed(5_000),
        initialValue = ImageGeneratorUiState(
            metrics = transientState.value.metrics,
            creations = transientState.value.creations
        )
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
        transientState.update { current ->
            val nextReturn = if (screen == ActiveScreen.CONTENT_SAFETY_SETTINGS &&
                current.activeScreen != ActiveScreen.CONTENT_SAFETY_SETTINGS
            ) {
                current.activeScreen
            } else {
                current.returnScreenForSafety
            }
            current.copy(
                activeScreen = screen,
                returnScreenForSafety = nextReturn,
                generationErrorMessage = null
            )
        }
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

    fun navigateBackFromContentSafety() {
        transientState.update { it.copy(activeScreen = it.returnScreenForSafety) }
    }

    fun navigateBackToHome() {
        transientState.update {
            if (it.activeScreen == ActiveScreen.CONTENT_SAFETY_SETTINGS) {
                it.copy(activeScreen = it.returnScreenForSafety)
            } else if (it.activeScreen != ActiveScreen.MAIN_TABS) {
                it.copy(activeScreen = ActiveScreen.MAIN_TABS)
            } else {
                it.copy(selectedTab = BottomNavTab.HOME)
            }
        }
    }

    /**
     * Updates and saves ONLY the User Image Prompt (never overwrites Content Safety Prompt).
     */
    fun updateUserImagePrompt(text: String) {
        transientState.update { it.copy(userPromptInput = text, generationErrorMessage = null) }
        viewModelScope.launch {
            prefsRepo.saveUserImagePrompt(text)
        }
    }

    /**
     * Saves ONLY the Content Safety Prompt (never overwrites User Image Prompt).
     */
    fun saveContentSafetyPrompt(safetyPrompt: String) {
        viewModelScope.launch {
            prefsRepo.saveContentSafetyPrompt(safetyPrompt)
            transientState.update {
                it.copy(statusToastMessage = "Content Safety Prompt saved on device.")
            }
        }
    }

    fun resetContentSafetyPrompt() {
        viewModelScope.launch {
            prefsRepo.resetContentSafetyPrompt()
            transientState.update {
                it.copy(statusToastMessage = "Content Safety Prompt reset to default.")
            }
        }
    }

    fun selectSafetyModePreset(mode: ContentSafetyMode, templatePrompt: String) {
        viewModelScope.launch {
            prefsRepo.setSafetyMode(mode)
            if (mode != ContentSafetyMode.CUSTOM) {
                prefsRepo.saveContentSafetyPrompt(templatePrompt)
            }
        }
    }

    fun updateSelectedInputPhoto(uri: Uri?) {
        transientState.update {
            it.copy(
                selectedInputPhotoUri = uri,
                generationErrorMessage = null
            )
        }
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

    // Phase 3: Verify ONNX Runtime Sessions
    fun verifyOnnxPipeline() {
        viewModelScope.launch {
            transientState.update {
                it.copy(
                    isImportingModels = true,
                    importProgressText = "Verifying ONNX Runtime tensors..."
                )
            }
            val mode = uiState.value.preferences.inferenceMode
            val res = withContext(Dispatchers.Default) {
                diffusionEngine.verifyInstalledModels(mode)
            }
            transientState.update {
                it.copy(
                    isImportingModels = false,
                    importProgressText = null,
                    onnxVerificationReport = res.getOrElse { err -> "Verification error: ${err.message}" }
                )
            }
        }
    }

    // Phase 4: Offline Text-to-Image Execution
    fun generateOfflineTextToImage(
        sampler: SamplerType,
        steps: Int,
        guidanceScale: Float,
        seed: Long,
        negativePrompt: String
    ) {
        if (transientState.value.isGenerating) return

        val stateSnapshot = uiState.value
        val userPrompt = stateSnapshot.userPromptInput.trim()
        if (userPrompt.isEmpty()) {
            transientState.update {
                it.copy(generationErrorMessage = "Please enter an image prompt first.")
            }
            return
        }

        // Final Generation Input = User Image Prompt + Saved Content Safety Prompt
        val finalGenerationInput = stateSnapshot.preferences.buildFinalGenerationInput(userPrompt)

        activeGenerationJob = viewModelScope.launch {
            transientState.update {
                it.copy(
                    isGenerating = true,
                    generationErrorMessage = null,
                    generationProgress = GenerationProgress(
                        stageTitle = "Preparing Offline ONNX Pipeline...",
                        currentStep = 0,
                        totalSteps = steps,
                        percent = 2,
                        elapsedMs = 0L
                    )
                )
            }

            val result = withContext(Dispatchers.Default) {
                diffusionEngine.generateTextToImage(
                    userPrompt = userPrompt,
                    finalGenerationInput = finalGenerationInput,
                    negativePrompt = negativePrompt,
                    resolutionLabel = stateSnapshot.preferences.resolution,
                    inferenceMode = stateSnapshot.preferences.inferenceMode,
                    samplerType = sampler,
                    steps = steps,
                    guidanceScale = guidanceScale,
                    seed = seed,
                    onProgress = { prog ->
                        transientState.update { s -> s.copy(generationProgress = prog) }
                    }
                )
            }

            result.fold(
                onSuccess = { genOut ->
                    val savedCreation = withContext(Dispatchers.IO) {
                        creationsRepo.saveCreation(
                            bitmap = genOut.bitmap,
                            userPrompt = userPrompt,
                            finalGenerationInput = genOut.finalGenerationInput,
                            resolution = genOut.resolutionLabel,
                            steps = genOut.steps,
                            seed = genOut.seed,
                            durationMs = genOut.durationMs
                        )
                    }
                    val allCreations = withContext(Dispatchers.IO) {
                        creationsRepo.loadAllCreations()
                    }
                    val refreshedMetrics = withContext(Dispatchers.IO) {
                        modelManager.inspectCurrentMetrics()
                    }
                    transientState.update {
                        it.copy(
                            isGenerating = false,
                            generationProgress = null,
                            creations = allCreations,
                            latestGeneratedCreation = savedCreation,
                            metrics = refreshedMetrics,
                            statusToastMessage = "Image generated offline in ${genOut.durationMs / 1000}s!"
                        )
                    }
                },
                onFailure = { err ->
                    val cancelled = err is CancellationException ||
                        err.message?.contains("cancelled", ignoreCase = true) == true
                    val refreshedMetrics = withContext(Dispatchers.IO) {
                        modelManager.inspectCurrentMetrics()
                    }
                    transientState.update {
                        it.copy(
                            isGenerating = false,
                            generationProgress = null,
                            metrics = refreshedMetrics,
                            generationErrorMessage = if (cancelled) null else err.message,
                            statusToastMessage = if (cancelled) "Generation cancelled & RAM freed." else null
                        )
                    }
                }
            )
        }
    }

    // Phase 5: Offline Image-to-Image Execution (VAE Encoder -> Text Encoder -> UNet -> VAE Decoder)
    fun generateOfflineImageToImage(
        sampler: SamplerType,
        strength: Float,
        steps: Int,
        guidanceScale: Float,
        seed: Long,
        negativePrompt: String
    ) {
        if (transientState.value.isGenerating) return

        val stateSnapshot = uiState.value
        val sourceUri = stateSnapshot.selectedInputPhotoUri
        if (sourceUri == null) {
            transientState.update {
                it.copy(
                    generationErrorMessage = if (stateSnapshot.preferences.isTamil) {
                        "முதலில் மாற்ற வேண்டிய Reference புகைப்படத்தைத் தேர்வு செய்யவும்."
                    } else {
                        "Please select a Reference Photo from Gallery first."
                    }
                )
            }
            return
        }

        val userPrompt = stateSnapshot.userPromptInput.trim()
        if (userPrompt.isEmpty()) {
            transientState.update {
                it.copy(
                    generationErrorMessage = if (stateSnapshot.preferences.isTamil) {
                        "படத்தை எப்படி மாற்ற வேண்டும் என்ற Prompt-ஐ எழுதவும்."
                    } else {
                        "Please enter a transformation prompt first."
                    }
                )
            }
            return
        }

        // Final Generation Input = User Image Prompt + Saved Content Safety Prompt
        val finalGenerationInput = stateSnapshot.preferences.buildFinalGenerationInput(userPrompt)

        activeGenerationJob = viewModelScope.launch {
            transientState.update {
                it.copy(
                    isGenerating = true,
                    generationErrorMessage = null,
                    generationProgress = GenerationProgress(
                        stageTitle = "Preparing Offline Image-to-Image Pipeline...",
                        currentStep = 0,
                        totalSteps = steps,
                        percent = 2,
                        elapsedMs = 0L
                    )
                )
            }

            val result = withContext(Dispatchers.Default) {
                diffusionEngine.generateImageToImage(
                    sourceImageUri = sourceUri,
                    userPrompt = userPrompt,
                    finalGenerationInput = finalGenerationInput,
                    negativePrompt = negativePrompt,
                    strength = strength,
                    resolutionLabel = stateSnapshot.preferences.resolution,
                    inferenceMode = stateSnapshot.preferences.inferenceMode,
                    samplerType = sampler,
                    steps = steps,
                    guidanceScale = guidanceScale,
                    seed = seed,
                    onProgress = { prog ->
                        transientState.update { s -> s.copy(generationProgress = prog) }
                    }
                )
            }

            result.fold(
                onSuccess = { out ->
                    val savedCreation = withContext(Dispatchers.IO) {
                        creationsRepo.saveCreation(
                            bitmap = out.bitmap,
                            userPrompt = userPrompt,
                            finalGenerationInput = out.finalGenerationInput,
                            resolution = out.resolutionLabel,
                            steps = out.activeSteps,
                            seed = out.seed,
                            durationMs = out.durationMs
                        )
                    }
                    val allCreations = withContext(Dispatchers.IO) {
                        creationsRepo.loadAllCreations()
                    }
                    val refreshedMetrics = withContext(Dispatchers.IO) {
                        modelManager.inspectCurrentMetrics()
                    }
                    transientState.update {
                        it.copy(
                            isGenerating = false,
                            generationProgress = null,
                            creations = allCreations,
                            latestGeneratedCreation = savedCreation,
                            latestImg2ImgCreation = savedCreation,
                            latestImg2ImgSourcePath = out.croppedSourceFile.absolutePath,
                            metrics = refreshedMetrics,
                            statusToastMessage = "Image transformed offline in ${out.durationMs / 1000}s!"
                        )
                    }
                },
                onFailure = { err ->
                    val cancelled = err is CancellationException ||
                        err.message?.contains("cancelled", ignoreCase = true) == true
                    val refreshedMetrics = withContext(Dispatchers.IO) {
                        modelManager.inspectCurrentMetrics()
                    }
                    transientState.update {
                        it.copy(
                            isGenerating = false,
                            generationProgress = null,
                            metrics = refreshedMetrics,
                            generationErrorMessage = if (cancelled) null else err.message,
                            statusToastMessage = if (cancelled) "Transformation cancelled & RAM freed." else null
                        )
                    }
                }
            )
        }
    }

    fun cancelGeneration() {
        diffusionEngine.requestCancel()
        activeGenerationJob?.cancel()
        activeGenerationJob = null
        viewModelScope.launch {
            val updated = withContext(Dispatchers.IO) {
                modelManager.unloadSessionsFromRam()
                modelManager.inspectCurrentMetrics()
            }
            transientState.update {
                it.copy(
                    isGenerating = false,
                    generationProgress = null,
                    metrics = updated,
                    statusToastMessage = "Generation cancelled."
                )
            }
        }
    }

    // Gallery & Creations Actions
    fun exportCreationToGallery(creation: GeneratedCreation) {
        viewModelScope.launch {
            val res = withContext(Dispatchers.IO) {
                creationsRepo.exportToPhoneGallery(creation)
            }
            transientState.update {
                it.copy(
                    statusToastMessage = res.fold(
                        onSuccess = { path -> "Saved to Gallery: $path" },
                        onFailure = { e -> "Gallery save error: ${e.message}" }
                    )
                )
            }
        }
    }

    fun toggleFavoriteCreation(id: String) {
        viewModelScope.launch {
            val updated = withContext(Dispatchers.IO) { creationsRepo.toggleFavorite(id) }
            transientState.update { state ->
                state.copy(
                    creations = updated,
                    latestGeneratedCreation = state.latestGeneratedCreation?.let { latest ->
                        updated.firstOrNull { it.id == latest.id } ?: latest
                    },
                    latestImg2ImgCreation = state.latestImg2ImgCreation?.let { latest ->
                        updated.firstOrNull { it.id == latest.id } ?: latest
                    }
                )
            }
        }
    }

    fun deleteCreation(id: String) {
        viewModelScope.launch {
            val updated = withContext(Dispatchers.IO) { creationsRepo.deleteCreation(id) }
            transientState.update { state ->
                state.copy(
                    creations = updated,
                    latestGeneratedCreation = if (state.latestGeneratedCreation?.id == id) {
                        updated.firstOrNull()
                    } else {
                        state.latestGeneratedCreation
                    },
                    latestImg2ImgCreation = if (state.latestImg2ImgCreation?.id == id) {
                        null
                    } else {
                        state.latestImg2ImgCreation
                    },
                    statusToastMessage = "Deleted image."
                )
            }
        }
    }

    // Model & Cache Management actions
    fun importAllModelsFromFolder(treeUri: Uri) {
        viewModelScope.launch {
            transientState.update {
                it.copy(
                    isImportingModels = true,
                    importProgressText = "Scanning selected folder...",
                    statusToastMessage = null,
                    onnxVerificationReport = null
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
                    onnxVerificationReport = null,
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
