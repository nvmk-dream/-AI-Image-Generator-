package com.example.ui

import android.app.Application
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.example.ai.AutoDownloadProgressState
import com.example.ai.DedicatedModelSlot
import com.example.ai.DiffusionModelSlot
import com.example.ai.GenerationProgress
import com.example.ai.ImageEditorAdjustments
import com.example.ai.OfflineImageEditorEngine
import com.example.ai.OnnxAutoModelDownloader
import com.example.ai.OnnxDiffusionEngine
import com.example.ai.OnnxModelManager
import com.example.ai.PreferredUnetImportMode
import com.example.ai.RealDeviceAndModelMetrics
import com.example.ai.SamplerType
import com.example.ai.UnetVariant
import com.example.data.ActiveDiffusionEngine
import com.example.data.AppLanguage
import com.example.data.AppPreferencesRepository
import com.example.data.AppPreferencesState
import com.example.data.CharacterProfile
import com.example.data.CharacterStudioRepository
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
import java.io.File

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
    IMAGE_EDITOR,
    MY_GALLERY
}

data class ImageGeneratorUiState(
    val activeScreen: ActiveScreen = ActiveScreen.MAIN_TABS,
    val selectedTab: BottomNavTab = BottomNavTab.HOME,
    val preferences: AppPreferencesState = AppPreferencesState(),
    val metrics: RealDeviceAndModelMetrics = RealDeviceAndModelMetrics(),
    val creations: List<GeneratedCreation> = emptyList(),
    val characters: List<CharacterProfile> = emptyList(),
    val selectedCharacterId: String? = null,
    val pendingCharacterPhotoUri: Uri? = null,
    val isImportingModels: Boolean = false,
    val importProgressText: String? = null,
    val onnxVerificationReport: String? = null,
    val preferredUnetMode: PreferredUnetImportMode = PreferredUnetImportMode.AUTO_FAST_MOBILE,
    val lastModelFolderUri: Uri? = null,
    val dualUnetDetectedInFolder: Boolean = false,
    val isGenerating: Boolean = false,
    val generationProgress: GenerationProgress? = null,
    val generationErrorMessage: String? = null,
    val latestGeneratedCreation: GeneratedCreation? = null,
    val latestImg2ImgCreation: GeneratedCreation? = null,
    val latestImg2ImgSourcePath: String? = null,
    val latestCharacterCreation: GeneratedCreation? = null,
    val editorImageUri: Uri? = null,
    val useLatestCreationInEditor: Boolean = false,
    val latestEditedCreation: GeneratedCreation? = null,
    val isSavingEdit: Boolean = false,
    val statusToastMessage: String? = null,
    val userPromptInput: String = "",
    val selectedInputPhotoUri: Uri? = null,
    val showFirstLaunchDownloadDialog: Boolean = false,
    val autoDownloadProgress: AutoDownloadProgressState = AutoDownloadProgressState()
)

private data class TransientUiState(
    val activeScreen: ActiveScreen = ActiveScreen.MAIN_TABS,
    val selectedTab: BottomNavTab = BottomNavTab.HOME,
    val metrics: RealDeviceAndModelMetrics = RealDeviceAndModelMetrics(),
    val creations: List<GeneratedCreation> = emptyList(),
    val characters: List<CharacterProfile> = emptyList(),
    val selectedCharacterId: String? = null,
    val pendingCharacterPhotoUri: Uri? = null,
    val isImportingModels: Boolean = false,
    val importProgressText: String? = null,
    val onnxVerificationReport: String? = null,
    val preferredUnetMode: PreferredUnetImportMode = PreferredUnetImportMode.AUTO_FAST_MOBILE,
    val lastModelFolderUri: Uri? = null,
    val dualUnetDetectedInFolder: Boolean = false,
    val isGenerating: Boolean = false,
    val generationProgress: GenerationProgress? = null,
    val generationErrorMessage: String? = null,
    val latestGeneratedCreation: GeneratedCreation? = null,
    val latestImg2ImgCreation: GeneratedCreation? = null,
    val latestImg2ImgSourcePath: String? = null,
    val latestCharacterCreation: GeneratedCreation? = null,
    val editorImageUri: Uri? = null,
    val useLatestCreationInEditor: Boolean = false,
    val latestEditedCreation: GeneratedCreation? = null,
    val isSavingEdit: Boolean = false,
    val statusToastMessage: String? = null,
    val userPromptInput: String? = null,
    val selectedInputPhotoUri: Uri? = null,
    val returnScreenForSafety: ActiveScreen = ActiveScreen.SETTINGS,
    val manualShowAutoDownloadDialog: Boolean = false,
    val dismissedFirstLaunchDialogInSession: Boolean = false,
    val autoDownloadProgress: AutoDownloadProgressState = AutoDownloadProgressState()
)

class AppViewModel(application: Application) : AndroidViewModel(application) {

    private val prefsRepo = AppPreferencesRepository(application)
    private val modelManager = OnnxModelManager(application)
    private val autoDownloader = OnnxAutoModelDownloader(modelManager)
    private val creationsRepo = CreationsRepository(application)
    private val characterRepo = CharacterStudioRepository(application)
    private val diffusionEngine = OnnxDiffusionEngine(application, modelManager)

    private var activeGenerationJob: Job? = null
    private var activeAutoDownloadJob: Job? = null

    private val initialChars = characterRepo.loadCharacters()

    private val transientState = MutableStateFlow(
        TransientUiState(
            metrics = modelManager.inspectCurrentMetrics(),
            creations = creationsRepo.loadAllCreations(),
            characters = initialChars,
            selectedCharacterId = initialChars.firstOrNull()?.id
        )
    )

    val uiState: StateFlow<ImageGeneratorUiState> = combine(
        prefsRepo.preferencesFlow,
        transientState
    ) { prefs, transient ->
        val effectiveUserPrompt = transient.userPromptInput ?: prefs.savedUserImagePrompt
        val shouldAutoPromptFirstOpen = prefs.isLoadedFromDisk &&
            !prefs.firstLaunchDownloadPromptHandled &&
            !transient.dismissedFirstLaunchDialogInSession &&
            !transient.autoDownloadProgress.isDownloading &&
            transient.metrics.coreInstalledCount == 0 &&
            !transient.metrics.hasAnySlotFilesOnDisk

        ImageGeneratorUiState(
            activeScreen = transient.activeScreen,
            selectedTab = transient.selectedTab,
            preferences = prefs,
            metrics = transient.metrics,
            creations = transient.creations,
            characters = transient.characters,
            selectedCharacterId = transient.selectedCharacterId,
            pendingCharacterPhotoUri = transient.pendingCharacterPhotoUri,
            isImportingModels = transient.isImportingModels,
            importProgressText = transient.importProgressText,
            onnxVerificationReport = transient.onnxVerificationReport,
            preferredUnetMode = transient.preferredUnetMode,
            lastModelFolderUri = transient.lastModelFolderUri,
            dualUnetDetectedInFolder = transient.dualUnetDetectedInFolder,
            isGenerating = transient.isGenerating,
            generationProgress = transient.generationProgress,
            generationErrorMessage = transient.generationErrorMessage,
            latestGeneratedCreation = transient.latestGeneratedCreation
                ?: transient.creations.firstOrNull(),
            latestImg2ImgCreation = transient.latestImg2ImgCreation,
            latestImg2ImgSourcePath = transient.latestImg2ImgSourcePath,
            latestCharacterCreation = transient.latestCharacterCreation,
            editorImageUri = transient.editorImageUri,
            useLatestCreationInEditor = transient.useLatestCreationInEditor,
            latestEditedCreation = transient.latestEditedCreation,
            isSavingEdit = transient.isSavingEdit,
            statusToastMessage = transient.statusToastMessage,
            userPromptInput = effectiveUserPrompt,
            selectedInputPhotoUri = transient.selectedInputPhotoUri,
            showFirstLaunchDownloadDialog = transient.manualShowAutoDownloadDialog || shouldAutoPromptFirstOpen,
            autoDownloadProgress = transient.autoDownloadProgress
        )
    }.stateIn(
        scope = viewModelScope,
        started = SharingStarted.WhileSubscribed(5_000),
        initialValue = ImageGeneratorUiState(
            metrics = transientState.value.metrics,
            creations = transientState.value.creations,
            characters = transientState.value.characters,
            selectedCharacterId = transientState.value.selectedCharacterId
        )
    )

    fun refreshDeviceAndModelMetrics() {
        viewModelScope.launch {
            val fresh = withContext(Dispatchers.IO) { modelManager.inspectCurrentMetrics() }
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
        transientState.update { it.copy(activeScreen = ActiveScreen.MAIN_TABS, selectedTab = tab) }
        refreshDeviceAndModelMetrics()
    }

    fun navigateBackFromContentSafety() {
        transientState.update { it.copy(activeScreen = it.returnScreenForSafety) }
    }

    fun navigateBackToHome() {
        transientState.update {
            when {
                it.activeScreen == ActiveScreen.CONTENT_SAFETY_SETTINGS ->
                    it.copy(activeScreen = it.returnScreenForSafety)
                it.activeScreen != ActiveScreen.MAIN_TABS ->
                    it.copy(activeScreen = ActiveScreen.MAIN_TABS)
                else -> it.copy(selectedTab = BottomNavTab.HOME)
            }
        }
    }

    fun updateUserImagePrompt(text: String) {
        transientState.update { it.copy(userPromptInput = text, generationErrorMessage = null) }
        viewModelScope.launch { prefsRepo.saveUserImagePrompt(text) }
    }

    fun saveContentSafetyPrompt(safetyPrompt: String) {
        viewModelScope.launch {
            prefsRepo.saveContentSafetyPrompt(safetyPrompt)
            transientState.update { it.copy(statusToastMessage = "Content Safety Prompt saved on device.") }
        }
    }

    fun resetContentSafetyPrompt() {
        viewModelScope.launch {
            prefsRepo.resetContentSafetyPrompt()
            transientState.update { it.copy(statusToastMessage = "Content Safety Prompt reset to default.") }
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
        transientState.update { it.copy(selectedInputPhotoUri = uri, generationErrorMessage = null) }
    }

    fun clearToast() {
        transientState.update { it.copy(statusToastMessage = null) }
    }

    // Settings
    fun setResolution(resolution: String) = viewModelScope.launch { prefsRepo.setResolution(resolution) }
    fun setInferenceMode(mode: String) = viewModelScope.launch { prefsRepo.setInferenceMode(mode) }
    fun setOfflineMode(enabled: Boolean) = viewModelScope.launch { prefsRepo.setOfflineMode(enabled) }
    fun setLanguage(language: AppLanguage) = viewModelScope.launch { prefsRepo.setLanguage(language) }
    fun setDarkTheme(dark: Boolean) = viewModelScope.launch { prefsRepo.setDarkTheme(dark) }
    fun setNotificationsEnabled(enabled: Boolean) = viewModelScope.launch { prefsRepo.setNotificationsEnabled(enabled) }
    fun setContentFilterEnabled(enabled: Boolean) = viewModelScope.launch { prefsRepo.setContentFilterEnabled(enabled) }

    fun setActiveEngine(engine: ActiveDiffusionEngine) {
        viewModelScope.launch {
            prefsRepo.setActiveEngine(engine)
            val refreshed = withContext(Dispatchers.IO) {
                modelManager.unloadSessionsFromRam()
                modelManager.inspectCurrentMetrics()
            }
            val isTamil = uiState.value.preferences.isTamil
            val toast = when (engine) {
                ActiveDiffusionEngine.FAST_LCM ->
                    if (isTamil) "⚡ Fast LCM மாடல் தேர்வு செய்யப்பட்டது" else "⚡ Fast LCM Engine active"
                ActiveDiffusionEngine.STANDARD_SD ->
                    if (isTamil) "🎨 Standard SD மாடல் தேர்வு செய்யப்பட்டது" else "🎨 Standard SD Engine active"
            }
            transientState.update {
                it.copy(
                    metrics = refreshed,
                    generationErrorMessage = null,
                    statusToastMessage = toast
                )
            }
        }
    }

    // Phase 3: Verify ONNX Runtime Sessions
    fun verifyOnnxPipeline() {
        viewModelScope.launch {
            transientState.update {
                it.copy(isImportingModels = true, importProgressText = "Verifying ONNX Runtime tensors...")
            }
            val mode = uiState.value.preferences.inferenceMode
            val res = withContext(Dispatchers.Default) { diffusionEngine.verifyInstalledModels(mode) }
            transientState.update {
                it.copy(
                    isImportingModels = false,
                    importProgressText = null,
                    onnxVerificationReport = res.getOrElse { err -> "Verification error: ${err.message}" }
                )
            }
        }
    }

    // Phase 4: Offline Text-to-Image
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
            transientState.update { it.copy(generationErrorMessage = "Please enter an image prompt first.") }
            return
        }
        val finalGenerationInput = stateSnapshot.preferences.buildFinalGenerationInput(userPrompt)

        activeGenerationJob = viewModelScope.launch {
            transientState.update {
                it.copy(
                    isGenerating = true,
                    generationErrorMessage = null,
                    generationProgress = GenerationProgress("Preparing Offline ONNX Pipeline...", 0, steps, 2, 0L)
                )
            }
            val activeEngine = stateSnapshot.preferences.activeEngine
            val effectiveSampler = if (activeEngine == ActiveDiffusionEngine.STANDARD_SD && sampler == SamplerType.LCM_FAST) {
                SamplerType.EULER_STANDARD
            } else {
                sampler
            }
            val result = withContext(Dispatchers.Default) {
                diffusionEngine.generateTextToImage(
                    userPrompt = userPrompt,
                    finalGenerationInput = finalGenerationInput,
                    negativePrompt = negativePrompt,
                    resolutionLabel = stateSnapshot.preferences.resolution,
                    inferenceMode = stateSnapshot.preferences.inferenceMode,
                    samplerType = effectiveSampler,
                    steps = steps,
                    guidanceScale = guidanceScale,
                    seed = seed,
                    activeEngine = activeEngine,
                    onProgress = { prog -> transientState.update { s -> s.copy(generationProgress = prog) } }
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
                    val allCreations = withContext(Dispatchers.IO) { creationsRepo.loadAllCreations() }
                    val refreshedMetrics = withContext(Dispatchers.IO) { modelManager.inspectCurrentMetrics() }
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
                onFailure = { err -> handlePipelineFailure(err) }
            )
        }
    }

    // Phase 5: Offline Image-to-Image
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
        val finalGenerationInput = stateSnapshot.preferences.buildFinalGenerationInput(userPrompt)

        activeGenerationJob = viewModelScope.launch {
            transientState.update {
                it.copy(
                    isGenerating = true,
                    generationErrorMessage = null,
                    generationProgress = GenerationProgress("Preparing Offline Image-to-Image...", 0, steps, 2, 0L)
                )
            }
            val activeEngine = stateSnapshot.preferences.activeEngine
            val effectiveSampler = if (activeEngine == ActiveDiffusionEngine.STANDARD_SD && sampler == SamplerType.LCM_FAST) {
                SamplerType.EULER_STANDARD
            } else {
                sampler
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
                    samplerType = effectiveSampler,
                    steps = steps,
                    guidanceScale = guidanceScale,
                    seed = seed,
                    activeEngine = activeEngine,
                    onProgress = { prog -> transientState.update { s -> s.copy(generationProgress = prog) } }
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
                    val allCreations = withContext(Dispatchers.IO) { creationsRepo.loadAllCreations() }
                    val refreshedMetrics = withContext(Dispatchers.IO) { modelManager.inspectCurrentMetrics() }
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
                onFailure = { err -> handlePipelineFailure(err) }
            )
        }
    }

    // Phase 6A: Character Studio Actions
    fun selectCharacter(id: String) {
        transientState.update { it.copy(selectedCharacterId = id, generationErrorMessage = null) }
    }

    fun setPendingCharacterPhotoUri(uri: Uri?) {
        transientState.update { it.copy(pendingCharacterPhotoUri = uri) }
    }

    fun saveCharacterProfile(name: String, appearancePrompt: String, fixedSeed: Long) {
        val refUri = transientState.value.pendingCharacterPhotoUri
        viewModelScope.launch {
            val updated = withContext(Dispatchers.IO) {
                characterRepo.saveCharacter(name, appearancePrompt, fixedSeed, refUri)
            }
            transientState.update {
                it.copy(
                    characters = updated,
                    selectedCharacterId = updated.firstOrNull()?.id,
                    pendingCharacterPhotoUri = null,
                    statusToastMessage = "Character '${updated.firstOrNull()?.name}' saved!"
                )
            }
        }
    }

    fun deleteCharacterProfile(id: String) {
        viewModelScope.launch {
            val updated = withContext(Dispatchers.IO) { characterRepo.deleteCharacter(id) }
            transientState.update {
                it.copy(
                    characters = updated,
                    selectedCharacterId = if (it.selectedCharacterId == id) updated.firstOrNull()?.id else it.selectedCharacterId,
                    statusToastMessage = "Character removed."
                )
            }
        }
    }

    fun generateCharacterScene(
        character: CharacterProfile,
        scenePrompt: String,
        useReferenceImg2Img: Boolean,
        strength: Float,
        sampler: SamplerType,
        steps: Int,
        guidanceScale: Float,
        negativePrompt: String
    ) {
        if (transientState.value.isGenerating) return
        val stateSnapshot = uiState.value
        val fullUserPrompt = character.buildCharacterScenePrompt(scenePrompt)
        val finalGenerationInput = stateSnapshot.preferences.buildFinalGenerationInput(fullUserPrompt)

        activeGenerationJob = viewModelScope.launch {
            transientState.update {
                it.copy(
                    isGenerating = true,
                    generationErrorMessage = null,
                    generationProgress = GenerationProgress("Generating ${character.name}...", 0, steps, 3, 0L)
                )
            }

            val activeEngine = stateSnapshot.preferences.activeEngine
            val refPath = character.referencePhotoPath
            if (useReferenceImg2Img && !refPath.isNullOrBlank() && File(refPath).exists()) {
                val res = withContext(Dispatchers.Default) {
                    diffusionEngine.generateImageToImage(
                        sourceImageUri = Uri.fromFile(File(refPath)),
                        userPrompt = fullUserPrompt,
                        finalGenerationInput = finalGenerationInput,
                        negativePrompt = negativePrompt,
                        strength = strength,
                        resolutionLabel = stateSnapshot.preferences.resolution,
                        inferenceMode = stateSnapshot.preferences.inferenceMode,
                        samplerType = sampler,
                        steps = steps,
                        guidanceScale = guidanceScale,
                        seed = character.fixedSeed,
                        activeEngine = activeEngine,
                        onProgress = { prog -> transientState.update { s -> s.copy(generationProgress = prog) } }
                    )
                }
                res.fold(
                    onSuccess = { out ->
                        val saved = withContext(Dispatchers.IO) {
                            creationsRepo.saveCreation(
                                bitmap = out.bitmap,
                                userPrompt = "[${character.name}] $scenePrompt",
                                finalGenerationInput = out.finalGenerationInput,
                                resolution = out.resolutionLabel,
                                steps = out.activeSteps,
                                seed = out.seed,
                                durationMs = out.durationMs
                            )
                        }
                        val all = withContext(Dispatchers.IO) { creationsRepo.loadAllCreations() }
                        val m = withContext(Dispatchers.IO) { modelManager.inspectCurrentMetrics() }
                        transientState.update {
                            it.copy(
                                isGenerating = false,
                                generationProgress = null,
                                creations = all,
                                latestGeneratedCreation = saved,
                                latestCharacterCreation = saved,
                                metrics = m,
                                statusToastMessage = "${character.name} scene generated!"
                            )
                        }
                    },
                    onFailure = { err -> handlePipelineFailure(err) }
                )
            } else {
                val res = withContext(Dispatchers.Default) {
                    diffusionEngine.generateTextToImage(
                        userPrompt = fullUserPrompt,
                        finalGenerationInput = finalGenerationInput,
                        negativePrompt = negativePrompt,
                        resolutionLabel = stateSnapshot.preferences.resolution,
                        inferenceMode = stateSnapshot.preferences.inferenceMode,
                        samplerType = sampler,
                        steps = steps,
                        guidanceScale = guidanceScale,
                        seed = character.fixedSeed,
                        activeEngine = activeEngine,
                        onProgress = { prog -> transientState.update { s -> s.copy(generationProgress = prog) } }
                    )
                }
                res.fold(
                    onSuccess = { out ->
                        val saved = withContext(Dispatchers.IO) {
                            creationsRepo.saveCreation(
                                bitmap = out.bitmap,
                                userPrompt = "[${character.name}] $scenePrompt",
                                finalGenerationInput = out.finalGenerationInput,
                                resolution = out.resolutionLabel,
                                steps = out.steps,
                                seed = out.seed,
                                durationMs = out.durationMs
                            )
                        }
                        val all = withContext(Dispatchers.IO) { creationsRepo.loadAllCreations() }
                        val m = withContext(Dispatchers.IO) { modelManager.inspectCurrentMetrics() }
                        transientState.update {
                            it.copy(
                                isGenerating = false,
                                generationProgress = null,
                                creations = all,
                                latestGeneratedCreation = saved,
                                latestCharacterCreation = saved,
                                metrics = m,
                                statusToastMessage = "${character.name} scene generated!"
                            )
                        }
                    },
                    onFailure = { err -> handlePipelineFailure(err) }
                )
            }
        }
    }

    // Phase 6C: Offline Image Editor Actions
    fun setEditorImageUri(uri: Uri?) {
        transientState.update {
            it.copy(editorImageUri = uri, useLatestCreationInEditor = false)
        }
    }

    fun loadLatestCreationInEditor() {
        transientState.update { it.copy(useLatestCreationInEditor = true) }
    }

    fun saveEditedPhoto(adjustments: ImageEditorAdjustments, presetLabel: String) {
        if (transientState.value.isSavingEdit) return
        val snapshot = uiState.value
        val srcUri = if (snapshot.useLatestCreationInEditor) null else snapshot.editorImageUri
        val fallbackPath = snapshot.latestGeneratedCreation?.filePath

        viewModelScope.launch {
            transientState.update { it.copy(isSavingEdit = true) }
            val result = withContext(Dispatchers.Default) {
                runCatching {
                    val t0 = System.currentTimeMillis()
                    val bmp = OfflineImageEditorEngine.renderEditedBitmap(
                        context = getApplication(),
                        sourceUri = srcUri,
                        fallbackFilePath = fallbackPath,
                        adjustments = adjustments
                    )
                    val saved = creationsRepo.saveCreation(
                        bitmap = bmp,
                        userPrompt = "Edited Photo ($presetLabel)",
                        finalGenerationInput = "Offline Image Editor ($presetLabel)",
                        resolution = "${bmp.width} × ${bmp.height}",
                        steps = 1,
                        seed = 0L,
                        durationMs = System.currentTimeMillis() - t0
                    )
                    bmp.recycle()
                    saved
                }
            }
            val all = withContext(Dispatchers.IO) { creationsRepo.loadAllCreations() }
            result.fold(
                onSuccess = { saved ->
                    transientState.update {
                        it.copy(
                            isSavingEdit = false,
                            creations = all,
                            latestGeneratedCreation = saved,
                            latestEditedCreation = saved,
                            statusToastMessage = "Edited photo saved to My Creations!"
                        )
                    }
                },
                onFailure = { err ->
                    transientState.update {
                        it.copy(
                            isSavingEdit = false,
                            statusToastMessage = "Edit error: ${err.message}"
                        )
                    }
                }
            )
        }
    }

    private suspend fun handlePipelineFailure(err: Throwable) {
        val cancelled = err is CancellationException ||
            err.message?.contains("cancelled", ignoreCase = true) == true
        val refreshedMetrics = withContext(Dispatchers.IO) { modelManager.inspectCurrentMetrics() }
        transientState.update {
            it.copy(
                isGenerating = false,
                generationProgress = null,
                metrics = refreshedMetrics,
                generationErrorMessage = if (cancelled) null else err.message,
                statusToastMessage = if (cancelled) "Cancelled & RAM freed." else null
            )
        }
    }

    fun cancelGeneration() {
        diffusionEngine.requestCancel()
        val jobToCancel = activeGenerationJob
        activeGenerationJob = null
        transientState.update {
            it.copy(
                isGenerating = false,
                generationProgress = null,
                generationErrorMessage = null,
                statusToastMessage = if (uiState.value.preferences.isTamil) {
                    "உடனடியாக நிறுத்தப்பட்டு RAM விடுவிக்கப்பட்டது."
                } else {
                    "Cancelled & RAM freed."
                }
            )
        }
        viewModelScope.launch {
            runCatching { jobToCancel?.cancel() }
            runCatching { jobToCancel?.join() }
            val updated = withContext(Dispatchers.IO) {
                modelManager.unloadSessionsFromRam()
                modelManager.inspectCurrentMetrics()
            }
            transientState.update { it.copy(metrics = updated) }
        }
    }

    // Gallery & Creations
    fun exportCreationToGallery(creation: GeneratedCreation) {
        viewModelScope.launch {
            val res = withContext(Dispatchers.IO) { creationsRepo.exportToPhoneGallery(creation) }
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
                    latestGeneratedCreation = state.latestGeneratedCreation?.let { l -> updated.firstOrNull { it.id == l.id } ?: l },
                    latestImg2ImgCreation = state.latestImg2ImgCreation?.let { l -> updated.firstOrNull { it.id == l.id } ?: l },
                    latestCharacterCreation = state.latestCharacterCreation?.let { l -> updated.firstOrNull { it.id == l.id } ?: l }
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
                    latestGeneratedCreation = if (state.latestGeneratedCreation?.id == id) updated.firstOrNull() else state.latestGeneratedCreation,
                    latestImg2ImgCreation = if (state.latestImg2ImgCreation?.id == id) null else state.latestImg2ImgCreation,
                    latestCharacterCreation = if (state.latestCharacterCreation?.id == id) null else state.latestCharacterCreation,
                    latestEditedCreation = if (state.latestEditedCreation?.id == id) null else state.latestEditedCreation,
                    statusToastMessage = "Deleted image."
                )
            }
        }
    }

    fun setPreferredUnetImportMode(mode: PreferredUnetImportMode) {
        transientState.update { it.copy(preferredUnetMode = mode) }
    }

    fun switchUnetFromSavedFolder(mode: PreferredUnetImportMode): Boolean {
        val savedUri = transientState.value.lastModelFolderUri ?: return false
        importAllModelsFromFolder(savedUri, modeOverride = mode)
        return true
    }

    // Model & Cache Management
    fun importAllModelsFromFolder(
        treeUri: Uri,
        modeOverride: PreferredUnetImportMode? = null
    ) {
        val chosenMode = modeOverride ?: transientState.value.preferredUnetMode
        viewModelScope.launch {
            transientState.update {
                it.copy(
                    preferredUnetMode = chosenMode,
                    lastModelFolderUri = treeUri,
                    isImportingModels = true,
                    importProgressText = "Scanning selected folder...",
                    statusToastMessage = null,
                    onnxVerificationReport = null
                )
            }
            val result = withContext(Dispatchers.IO) {
                modelManager.importAllModelsFromFolder(
                    treeUri = treeUri,
                    preferredUnetMode = chosenMode
                ) { progress ->
                    transientState.update { s -> s.copy(importProgressText = progress) }
                }
            }
            val updatedMetrics = withContext(Dispatchers.IO) { modelManager.inspectCurrentMetrics() }
            val isTamil = uiState.value.preferences.isTamil
            result.fold(
                onSuccess = { importRes ->
                    val dedCount = importRes.importedDedicatedSlots.size
                    if (importRes.fastLcmReady) {
                        prefsRepo.setActiveEngine(ActiveDiffusionEngine.FAST_LCM)
                    } else if (importRes.standardSdReady) {
                        prefsRepo.setActiveEngine(ActiveDiffusionEngine.STANDARD_SD)
                    }
                    val engineNote = when {
                        importRes.fastLcmReady && importRes.standardSdReady ->
                            if (isTamil) " ⚡ Fast LCM & 🎨 Standard SD இரண்டும் தயார்!"
                            else " ⚡ Fast LCM & 🎨 Standard SD both linked & ready!"
                        importRes.fastLcmReady ->
                            if (isTamil) " ⚡ Fast LCM இணைக்கப்பட்டது!" else " ⚡ Fast LCM linked & ready!"
                        importRes.standardSdReady ->
                            if (isTamil) " 🎨 Standard SD இணைக்கப்பட்டது!" else " 🎨 Standard SD linked & ready!"
                        else -> ""
                    }
                    val spaceNote = if (importRes.skippedLowSpaceNote.isNotBlank()) {
                        " (${importRes.skippedLowSpaceNote})"
                    } else ""
                    val msg = if (dedCount > 0) {
                        if (isTamil) {
                            "✅ $dedCount ஸ்லாட்டுகள் வெற்றிகரமாக நிரப்பப்பட்டன!$engineNote$spaceNote"
                        } else {
                            "✅ Auto-filled $dedCount dedicated slot(s)!$engineNote$spaceNote"
                        }
                    } else {
                        if (isTamil) "அந்த Folder-ல் ONNX மாடல் ஃபைல்கள் எதுவும் இல்லை."
                        else "No matching ONNX or Tokenizer files found in that folder."
                    }
                    transientState.update {
                        it.copy(
                            metrics = updatedMetrics,
                            dualUnetDetectedInFolder = importRes.dualUnetDetected,
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

    fun importDedicatedSlot(uri: Uri, slot: DedicatedModelSlot, originalNameHint: String? = null) {
        viewModelScope.launch {
            val label = originalNameHint?.takeIf { it.isNotBlank() } ?: slot.expectedFileHint
            transientState.update {
                it.copy(
                    isImportingModels = true,
                    importProgressText = "Uploading $label...",
                    onnxVerificationReport = null
                )
            }
            val result = withContext(Dispatchers.IO) {
                modelManager.importDedicatedSlotFromUri(uri, slot, originalNameHint) { prog ->
                    transientState.update { s -> s.copy(importProgressText = prog) }
                }
            }
            val updatedMetrics = withContext(Dispatchers.IO) { modelManager.inspectCurrentMetrics() }
            val isTamil = uiState.value.preferences.isTamil
            result.fold(
                onSuccess = { (savedSlot, bytes) ->
                    if (savedSlot == DedicatedModelSlot.FAST_LCM_GRAPH || savedSlot == DedicatedModelSlot.FAST_LCM_WEIGHTS) {
                        if (updatedMetrics.isFastLcmEngineReady) {
                            prefsRepo.setActiveEngine(ActiveDiffusionEngine.FAST_LCM)
                        }
                    } else if (savedSlot == DedicatedModelSlot.STANDARD_SD_GRAPH || savedSlot == DedicatedModelSlot.STANDARD_SD_WEIGHTS) {
                        if (updatedMetrics.isStandardSdEngineReady && !updatedMetrics.isFastLcmEngineReady) {
                            prefsRepo.setActiveEngine(ActiveDiffusionEngine.STANDARD_SD)
                        }
                    }
                    val slotStatus = updatedMetrics.dedicatedSlots.firstOrNull { it.slot == savedSlot }
                    val detail = if (isTamil) slotStatus?.statusDetailTa else slotStatus?.statusDetailEn
                    val msg = detail?.takeIf { it.isNotBlank() }
                        ?: "✅ ${savedSlot.name} (${OnnxModelManager.formatBytes(bytes)}) ready!"
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
                            statusToastMessage = "Upload error: ${err.message}"
                        )
                    }
                }
            )
        }
    }

    fun importTokenizerFiles(uris: List<Uri>, nameHints: List<String?>) {
        if (uris.isEmpty()) return
        viewModelScope.launch {
            transientState.update {
                it.copy(
                    isImportingModels = true,
                    importProgressText = "Uploading Tokenizer (vocab.json & merges.txt)...",
                    onnxVerificationReport = null
                )
            }
            val result = withContext(Dispatchers.IO) {
                modelManager.importTokenizerUris(uris, nameHints) { prog ->
                    transientState.update { s -> s.copy(importProgressText = prog) }
                }
            }
            val updatedMetrics = withContext(Dispatchers.IO) { modelManager.inspectCurrentMetrics() }
            val isTamil = uiState.value.preferences.isTamil
            result.fold(
                onSuccess = {
                    val slotStatus = updatedMetrics.dedicatedSlots.firstOrNull { it.slot == DedicatedModelSlot.TOKENIZER }
                    val msg = if (isTamil) {
                        slotStatus?.statusDetailTa ?: "✅ Slot 6 Tokenizer ஏற்றப்பட்டது!"
                    } else {
                        slotStatus?.statusDetailEn ?: "✅ Slot 6 Tokenizer uploaded!"
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
                            statusToastMessage = "Tokenizer upload error: ${err.message}"
                        )
                    }
                }
            )
        }
    }

    fun deleteDedicatedSlot(slot: DedicatedModelSlot) {
        viewModelScope.launch {
            withContext(Dispatchers.IO) { modelManager.deleteDedicatedSlot(slot) }
            val updated = withContext(Dispatchers.IO) { modelManager.inspectCurrentMetrics() }
            val isTamil = uiState.value.preferences.isTamil
            transientState.update {
                it.copy(
                    metrics = updated,
                    onnxVerificationReport = null,
                    statusToastMessage = if (isTamil) "${slot.titleTa} நீக்கப்பட்டது" else "Removed ${slot.titleEn}"
                )
            }
        }
    }

    fun importSingleSlot(uri: Uri, slot: DiffusionModelSlot, originalNameHint: String? = null) {
        viewModelScope.launch {
            transientState.update {
                it.copy(
                    isImportingModels = true,
                    importProgressText = "Importing ${originalNameHint ?: slot.canonicalFileName}..."
                )
            }
            val result = withContext(Dispatchers.IO) {
                modelManager.importSingleSlotFromUri(uri, slot, originalNameHint) { prog ->
                    transientState.update { s -> s.copy(importProgressText = prog) }
                }
            }
            val updatedMetrics = withContext(Dispatchers.IO) { modelManager.inspectCurrentMetrics() }
            result.fold(
                onSuccess = { bytes ->
                    transientState.update {
                        it.copy(
                            metrics = updatedMetrics,
                            isImportingModels = false,
                            importProgressText = null,
                            statusToastMessage = "${originalNameHint ?: slot.canonicalFileName} ready (${OnnxModelManager.formatBytes(bytes)})"
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

    fun importCompanionWeightForSlot(uri: Uri, slot: DiffusionModelSlot, originalNameHint: String? = null) {
        viewModelScope.launch {
            transientState.update {
                it.copy(
                    isImportingModels = true,
                    importProgressText = "Attaching weights for ${slot.canonicalFileName}..."
                )
            }
            val result = withContext(Dispatchers.IO) {
                modelManager.importCompanionWeightForSlot(uri, slot, originalNameHint) { prog ->
                    transientState.update { s -> s.copy(importProgressText = prog) }
                }
            }
            val updatedMetrics = withContext(Dispatchers.IO) { modelManager.inspectCurrentMetrics() }
            result.fold(
                onSuccess = { (savedName, bytes) ->
                    transientState.update {
                        it.copy(
                            metrics = updatedMetrics,
                            isImportingModels = false,
                            importProgressText = null,
                            onnxVerificationReport = null,
                            statusToastMessage = "$savedName attached (${OnnxModelManager.formatBytes(bytes)})!"
                        )
                    }
                },
                onFailure = { err ->
                    transientState.update {
                        it.copy(
                            metrics = updatedMetrics,
                            isImportingModels = false,
                            importProgressText = null,
                            statusToastMessage = "Weights attach error: ${err.message}"
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

    fun openAutoDownloadDialog() {
        transientState.update { it.copy(manualShowAutoDownloadDialog = true) }
    }

    fun dismissFirstLaunchDownloadDialog() {
        transientState.update {
            it.copy(
                manualShowAutoDownloadDialog = false,
                dismissedFirstLaunchDialogInSession = true
            )
        }
        viewModelScope.launch {
            prefsRepo.setFirstLaunchDownloadPromptHandled(true)
        }
    }

    fun clearAutoDownloadBanner() {
        if (!transientState.value.autoDownloadProgress.isDownloading) {
            transientState.update { it.copy(autoDownloadProgress = AutoDownloadProgressState()) }
        }
    }

    fun cancelAutoDownload() {
        autoDownloader.requestCancel()
        activeAutoDownloadJob?.cancel()
    }

    fun confirmAndStartAutoDownload(
        selectedSlots: Set<DedicatedModelSlot> = OnnxAutoModelDownloader.DEFAULT_SELECTED_CORE_SLOTS
    ) {
        if (transientState.value.autoDownloadProgress.isDownloading) return
        val coreOnly = selectedSlots.filter { it.isCoreSlot }.toSet()
        if (coreOnly.isEmpty()) {
            dismissFirstLaunchDownloadDialog()
            return
        }

        transientState.update {
            it.copy(
                manualShowAutoDownloadDialog = false,
                dismissedFirstLaunchDialogInSession = true,
                onnxVerificationReport = null,
                autoDownloadProgress = AutoDownloadProgressState(
                    isDownloading = true,
                    totalSelectedSlotsCount = coreOnly.size,
                    statusTextEn = "Preparing phone folder & starting download...",
                    statusTextTa = "மொபைல் ஃபோல்டர் தயார் செய்யப்பட்டு பதிவிறக்கம் தொடங்குகிறது..."
                )
            )
        }

        activeAutoDownloadJob = viewModelScope.launch {
            prefsRepo.setFirstLaunchDownloadPromptHandled(true)

            val result = withContext(Dispatchers.IO) {
                autoDownloader.downloadSelectedCoreSlots(
                    selectedSlots = coreOnly,
                    onProgress = { progressState ->
                        transientState.update { s -> s.copy(autoDownloadProgress = progressState) }
                    },
                    onSlotCompleted = { completedSlot ->
                        val refreshed = modelManager.inspectCurrentMetrics()
                        if ((completedSlot == DedicatedModelSlot.FAST_LCM_GRAPH ||
                                completedSlot == DedicatedModelSlot.FAST_LCM_WEIGHTS) &&
                            refreshed.isFastLcmEngineReady
                        ) {
                            prefsRepo.setActiveEngine(ActiveDiffusionEngine.FAST_LCM)
                        } else if ((completedSlot == DedicatedModelSlot.STANDARD_SD_GRAPH ||
                                completedSlot == DedicatedModelSlot.STANDARD_SD_WEIGHTS) &&
                            refreshed.isStandardSdEngineReady &&
                            !refreshed.isFastLcmEngineReady
                        ) {
                            prefsRepo.setActiveEngine(ActiveDiffusionEngine.STANDARD_SD)
                        }
                        transientState.update { s -> s.copy(metrics = refreshed) }
                    }
                )
            }

            val finalMetrics = withContext(Dispatchers.IO) { modelManager.inspectCurrentMetrics() }
            val isTamil = uiState.value.preferences.isTamil

            result.fold(
                onSuccess = { batch ->
                    val readyTotal = batch.downloadedSlots.size + batch.skippedAlreadyReadySlots.size
                    val lowSpaceNoteEn = if (batch.pausedLowStorageSlot != null) {
                        " (Slot 2B ~3.44 GB paused due to low phone storage — Fast LCM is 100% Ready!)"
                    } else ""
                    val lowSpaceNoteTa = if (batch.pausedLowStorageSlot != null) {
                        " (போனில் இடமில்லாததால் Slot 2B ~3.44 GB நிறுத்தப்பட்டது — Fast LCM 100% தயார்!)"
                    } else ""

                    val summaryEn = "✅ $readyTotal Core Slot(s) saved in ${batch.folderDisplayPath} & auto-linked!$lowSpaceNoteEn"
                    val summaryTa = "✅ $readyTotal Core ஸ்லாட்டுகள் ${batch.folderDisplayPath} ஃபோல்டரில் சேமிக்கப்பட்டு ஆப்பில் தானாக இணைக்கப்பட்டன!$lowSpaceNoteTa"

                    transientState.update {
                        it.copy(
                            metrics = finalMetrics,
                            autoDownloadProgress = AutoDownloadProgressState(
                                isDownloading = false,
                                completedSlotsCount = readyTotal,
                                totalSelectedSlotsCount = coreOnly.size,
                                finishedSummaryEn = summaryEn,
                                finishedSummaryTa = summaryTa
                            ),
                            statusToastMessage = if (isTamil) summaryTa else summaryEn
                        )
                    }
                },
                onFailure = { err ->
                    if (err is CancellationException) {
                        val cancelEn = "Download paused — partial progress is saved in IG_Models. Tap Resume anytime."
                        val cancelTa = "பதிவிறக்கம் இடைநிறுத்தப்பட்டது — இதுவரை இறங்கிய பகுதி பத்திரமாக உள்ளது. எப்போது வேண்டுமானாலும் தொடரலாம்."
                        transientState.update {
                            it.copy(
                                metrics = finalMetrics,
                                autoDownloadProgress = AutoDownloadProgressState(
                                    isDownloading = false,
                                    errorTextEn = cancelEn,
                                    errorTextTa = cancelTa
                                ),
                                statusToastMessage = if (isTamil) cancelTa else cancelEn
                            )
                        }
                    } else {
                        val errMsg = err.message ?: "Network error"
                        val failEn = "Download interrupted: $errMsg (Tap Auto-Download to resume from saved progress)"
                        val failTa = "பதிவிறக்கம் தடைபட்டது: $errMsg (மீண்டும் அழுத்தினால் விட்ட இடத்திலிருந்தே தொடரும்)"
                        transientState.update {
                            it.copy(
                                metrics = finalMetrics,
                                autoDownloadProgress = AutoDownloadProgressState(
                                    isDownloading = false,
                                    errorTextEn = failEn,
                                    errorTextTa = failTa
                                ),
                                statusToastMessage = if (isTamil) failTa else failEn
                            )
                        }
                    }
                }
            )
        }
    }
}
