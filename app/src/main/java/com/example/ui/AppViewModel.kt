package com.example.ui

import android.app.Application
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.example.ai.DiffusionModelSlot
import com.example.ai.GenerationProgress
import com.example.ai.ImageEditorAdjustments
import com.example.ai.OfflineFaceSwapEngine
import com.example.ai.OfflineImageEditorEngine
import com.example.ai.OnnxDiffusionEngine
import com.example.ai.OnnxModelManager
import com.example.ai.RealDeviceAndModelMetrics
import com.example.ai.SamplerType
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
    val characters: List<CharacterProfile> = emptyList(),
    val selectedCharacterId: String? = null,
    val pendingCharacterPhotoUri: Uri? = null,
    val isImportingModels: Boolean = false,
    val importProgressText: String? = null,
    val onnxVerificationReport: String? = null,
    val isGenerating: Boolean = false,
    val generationProgress: GenerationProgress? = null,
    val generationErrorMessage: String? = null,
    val latestGeneratedCreation: GeneratedCreation? = null,
    val latestImg2ImgCreation: GeneratedCreation? = null,
    val latestImg2ImgSourcePath: String? = null,
    val latestCharacterCreation: GeneratedCreation? = null,
    val faceSwapSourceUri: Uri? = null,
    val faceSwapTargetUri: Uri? = null,
    val useLatestCreationAsFaceSwapTarget: Boolean = false,
    val latestFaceSwapCreation: GeneratedCreation? = null,
    val latestFaceSwapTargetPreviewPath: String? = null,
    val editorImageUri: Uri? = null,
    val useLatestCreationInEditor: Boolean = false,
    val latestEditedCreation: GeneratedCreation? = null,
    val isSavingEdit: Boolean = false,
    val statusToastMessage: String? = null,
    val userPromptInput: String = "",
    val selectedInputPhotoUri: Uri? = null
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
    val isGenerating: Boolean = false,
    val generationProgress: GenerationProgress? = null,
    val generationErrorMessage: String? = null,
    val latestGeneratedCreation: GeneratedCreation? = null,
    val latestImg2ImgCreation: GeneratedCreation? = null,
    val latestImg2ImgSourcePath: String? = null,
    val latestCharacterCreation: GeneratedCreation? = null,
    val faceSwapSourceUri: Uri? = null,
    val faceSwapTargetUri: Uri? = null,
    val useLatestCreationAsFaceSwapTarget: Boolean = false,
    val latestFaceSwapCreation: GeneratedCreation? = null,
    val latestFaceSwapTargetPreviewPath: String? = null,
    val editorImageUri: Uri? = null,
    val useLatestCreationInEditor: Boolean = false,
    val latestEditedCreation: GeneratedCreation? = null,
    val isSavingEdit: Boolean = false,
    val statusToastMessage: String? = null,
    val userPromptInput: String? = null,
    val selectedInputPhotoUri: Uri? = null,
    val returnScreenForSafety: ActiveScreen = ActiveScreen.SETTINGS
)

class AppViewModel(application: Application) : AndroidViewModel(application) {

    private val prefsRepo = AppPreferencesRepository(application)
    private val modelManager = OnnxModelManager(application)
    private val creationsRepo = CreationsRepository(application)
    private val characterRepo = CharacterStudioRepository(application)
    private val diffusionEngine = OnnxDiffusionEngine(application, modelManager)
    private val faceSwapEngine = OfflineFaceSwapEngine(application, modelManager)

    private var activeGenerationJob: Job? = null

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
            isGenerating = transient.isGenerating,
            generationProgress = transient.generationProgress,
            generationErrorMessage = transient.generationErrorMessage,
            latestGeneratedCreation = transient.latestGeneratedCreation
                ?: transient.creations.firstOrNull(),
            latestImg2ImgCreation = transient.latestImg2ImgCreation,
            latestImg2ImgSourcePath = transient.latestImg2ImgSourcePath,
            latestCharacterCreation = transient.latestCharacterCreation,
            faceSwapSourceUri = transient.faceSwapSourceUri,
            faceSwapTargetUri = transient.faceSwapTargetUri,
            useLatestCreationAsFaceSwapTarget = transient.useLatestCreationAsFaceSwapTarget,
            latestFaceSwapCreation = transient.latestFaceSwapCreation,
            latestFaceSwapTargetPreviewPath = transient.latestFaceSwapTargetPreviewPath,
            editorImageUri = transient.editorImageUri,
            useLatestCreationInEditor = transient.useLatestCreationInEditor,
            latestEditedCreation = transient.latestEditedCreation,
            isSavingEdit = transient.isSavingEdit,
            statusToastMessage = transient.statusToastMessage,
            userPromptInput = effectiveUserPrompt,
            selectedInputPhotoUri = transient.selectedInputPhotoUri
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

    // Phase 6B: Offline Face Swap Actions
    fun setFaceSwapSourceUri(uri: Uri?) {
        transientState.update { it.copy(faceSwapSourceUri = uri, generationErrorMessage = null) }
    }

    fun setFaceSwapTargetUri(uri: Uri?) {
        transientState.update {
            it.copy(
                faceSwapTargetUri = uri,
                useLatestCreationAsFaceSwapTarget = false,
                generationErrorMessage = null
            )
        }
    }

    fun useLatestCreationAsFaceSwapTarget() {
        transientState.update {
            it.copy(useLatestCreationAsFaceSwapTarget = true, generationErrorMessage = null)
        }
    }

    fun runOfflineFaceSwap(enableSkinToneMatch: Boolean, blendStrength: Float) {
        if (transientState.value.isGenerating) return
        val snapshot = uiState.value
        val srcUri = snapshot.faceSwapSourceUri
        if (srcUri == null) {
            transientState.update {
                it.copy(generationErrorMessage = "Please select a Source Face photo first.")
            }
            return
        }
        val fallbackPath = if (snapshot.useLatestCreationAsFaceSwapTarget) {
            snapshot.latestGeneratedCreation?.filePath
        } else {
            null
        }
        if (snapshot.faceSwapTargetUri == null && fallbackPath.isNullOrBlank()) {
            transientState.update {
                it.copy(generationErrorMessage = "Please select a Target Photo first.")
            }
            return
        }

        activeGenerationJob = viewModelScope.launch {
            transientState.update {
                it.copy(
                    isGenerating = true,
                    generationErrorMessage = null,
                    generationProgress = GenerationProgress("Starting Offline Face Swap...", 1, 5, 5, 0L)
                )
            }

            val result = withContext(Dispatchers.Default) {
                faceSwapEngine.executeOfflineFaceSwap(
                    sourceFaceUri = srcUri,
                    targetImageUri = if (snapshot.useLatestCreationAsFaceSwapTarget) null else snapshot.faceSwapTargetUri,
                    targetFallbackFilePath = fallbackPath,
                    enableSkinToneMatch = enableSkinToneMatch,
                    blendStrength = blendStrength,
                    inferenceMode = snapshot.preferences.inferenceMode,
                    onProgress = { prog -> transientState.update { s -> s.copy(generationProgress = prog) } }
                )
            }

            result.fold(
                onSuccess = { swapOut ->
                    val saved = withContext(Dispatchers.IO) {
                        creationsRepo.saveCreation(
                            bitmap = swapOut.swappedBitmap,
                            userPrompt = "Offline Face Swap (${swapOut.modelSummary})",
                            finalGenerationInput = "Offline Face Swap",
                            resolution = "${swapOut.swappedBitmap.width} × ${swapOut.swappedBitmap.height}",
                            steps = 5,
                            seed = 0L,
                            durationMs = swapOut.durationMs
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
                            latestFaceSwapCreation = saved,
                            latestFaceSwapTargetPreviewPath = swapOut.targetPreviewFile.absolutePath,
                            metrics = m,
                            statusToastMessage = "Face swapped offline in ${swapOut.durationMs / 1000.0}s!"
                        )
                    }
                },
                onFailure = { err -> handlePipelineFailure(err) }
            )
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
                    statusToastMessage = "Cancelled & RAM freed."
                )
            }
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
                    latestCharacterCreation = state.latestCharacterCreation?.let { l -> updated.firstOrNull { it.id == l.id } ?: l },
                    latestFaceSwapCreation = state.latestFaceSwapCreation?.let { l -> updated.firstOrNull { it.id == l.id } ?: l }
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
                    latestFaceSwapCreation = if (state.latestFaceSwapCreation?.id == id) null else state.latestFaceSwapCreation,
                    latestEditedCreation = if (state.latestEditedCreation?.id == id) null else state.latestEditedCreation,
                    statusToastMessage = "Deleted image."
                )
            }
        }
    }

    // Model & Cache Management
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
            val updatedMetrics = withContext(Dispatchers.IO) { modelManager.inspectCurrentMetrics() }
            result.fold(
                onSuccess = { imported ->
                    val msg = if (imported.isNotEmpty()) {
                        "Imported ${imported.size} ONNX model(s): ${imported.joinToString { it.canonicalFileName }}"
                    } else {
                        "No matching .onnx files found in that folder."
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

    fun importSingleSlot(uri: Uri, slot: DiffusionModelSlot, originalNameHint: String? = null) {
        viewModelScope.launch {
            transientState.update {
                it.copy(
                    isImportingModels = true,
                    importProgressText = "Importing ${originalNameHint ?: slot.canonicalFileName}..."
                )
            }
            val result = withContext(Dispatchers.IO) {
                modelManager.importSingleSlotFromUri(uri, slot, originalNameHint)
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
