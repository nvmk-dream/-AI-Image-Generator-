package com.example

import android.content.Intent
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.Crossfade
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.statusBars
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.material3.Scaffold
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.documentfile.provider.DocumentFile
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.example.ai.DedicatedModelSlot
import com.example.ai.DiffusionModelSlot
import com.example.ai.PreferredUnetImportMode
import com.example.ui.ActiveScreen
import com.example.ui.AppViewModel
import com.example.ui.BottomNavTab
import com.example.ui.components.AutoDownloadLiveBanner
import com.example.ui.components.DreamyHeartBackground
import com.example.ui.components.FirstLaunchModelDownloadDialog
import com.example.ui.components.GlassBottomNavBar
import com.example.ui.screens.CharacterStudioScreen
import com.example.ui.screens.ContentSafetySettingsScreen
import com.example.ui.screens.CreationsCollectionScreen
import com.example.ui.screens.FaceSwapScreen
import com.example.ui.screens.HomeScreen
import com.example.ui.screens.ImageEditorScreen
import com.example.ui.screens.ImageToImageScreen
import com.example.ui.screens.ModelsTabScreen
import com.example.ui.screens.SettingsScreen
import com.example.ui.screens.TextToImageScreen
import com.example.ui.theme.MyApplicationTheme

private enum class PhotoPickTarget {
    IMG2IMG_REFERENCE,
    CHARACTER_REFERENCE,
    FACESWAP_SOURCE,
    FACESWAP_TARGET,
    IMAGE_EDITOR_SOURCE
}

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            ImageGeneratorRoot()
        }
    }
}

@Composable
fun ImageGeneratorRoot(
    appViewModel: AppViewModel = viewModel()
) {
    val uiState by appViewModel.uiState.collectAsStateWithLifecycle()
    val context = LocalContext.current
    val snackbarHostState = remember { SnackbarHostState() }
    var pendingDedicatedSlot by remember { mutableStateOf(DedicatedModelSlot.FAST_LCM_GRAPH) }
    var pendingSlotForSingleImport by remember { mutableStateOf(DiffusionModelSlot.TEXT_ENCODER) }
    var pendingSlotForCompanionImport by remember { mutableStateOf(DiffusionModelSlot.UNET) }
    var pendingFolderUnetMode by remember { mutableStateOf<PreferredUnetImportMode?>(null) }
    var photoPickTarget by remember { mutableStateOf(PhotoPickTarget.IMG2IMG_REFERENCE) }

    val folderPickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenDocumentTree()
    ) { treeUri ->
        val modeOverride = pendingFolderUnetMode
        pendingFolderUnetMode = null
        if (treeUri != null) {
            runCatching {
                context.contentResolver.takePersistableUriPermission(
                    treeUri,
                    Intent.FLAG_GRANT_READ_URI_PERMISSION
                )
            }
            appViewModel.importAllModelsFromFolder(treeUri, modeOverride = modeOverride)
        }
    }

    val dedicatedSlotPickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenDocument()
    ) { fileUri ->
        if (fileUri != null) {
            val fileName = runCatching {
                DocumentFile.fromSingleUri(context, fileUri)?.name
            }.getOrNull()
            appViewModel.importDedicatedSlot(fileUri, pendingDedicatedSlot, fileName)
        }
    }

    val tokenizerMultiPickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenMultipleDocuments()
    ) { uris ->
        if (uris.isNotEmpty()) {
            val hints = uris.map { u ->
                runCatching { DocumentFile.fromSingleUri(context, u)?.name }.getOrNull()
            }
            appViewModel.importTokenizerFiles(uris, hints)
        }
    }

    val singleModelPickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenDocument()
    ) { fileUri ->
        if (fileUri != null) {
            val fileName = runCatching {
                DocumentFile.fromSingleUri(context, fileUri)?.name
            }.getOrNull()
            appViewModel.importSingleSlot(fileUri, pendingSlotForSingleImport, fileName)
        }
    }

    val companionWeightPickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenDocument()
    ) { fileUri ->
        if (fileUri != null) {
            val fileName = runCatching {
                DocumentFile.fromSingleUri(context, fileUri)?.name
            }.getOrNull()
            appViewModel.importCompanionWeightForSlot(fileUri, pendingSlotForCompanionImport, fileName)
        }
    }

    val photoPickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.PickVisualMedia()
    ) { photoUri ->
        if (photoUri != null) {
            when (photoPickTarget) {
                PhotoPickTarget.IMG2IMG_REFERENCE -> appViewModel.updateSelectedInputPhoto(photoUri)
                PhotoPickTarget.CHARACTER_REFERENCE -> appViewModel.setPendingCharacterPhotoUri(photoUri)
                PhotoPickTarget.FACESWAP_SOURCE -> appViewModel.setFaceSwapSourceUri(photoUri)
                PhotoPickTarget.FACESWAP_TARGET -> appViewModel.setFaceSwapTargetUri(photoUri)
                PhotoPickTarget.IMAGE_EDITOR_SOURCE -> appViewModel.setEditorImageUri(photoUri)
            }
        }
    }

    fun launchPhotoPicker(target: PhotoPickTarget) {
        photoPickTarget = target
        photoPickerLauncher.launch(
            PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)
        )
    }

    LaunchedEffect(uiState.statusToastMessage) {
        val msg = uiState.statusToastMessage
        if (!msg.isNullOrBlank()) {
            snackbarHostState.showSnackbar(msg)
            appViewModel.clearToast()
        }
    }

    MyApplicationTheme(darkTheme = uiState.preferences.darkTheme) {
        DreamyHeartBackground(darkTheme = uiState.preferences.darkTheme) {
            Scaffold(
                containerColor = Color.Transparent,
                contentWindowInsets = WindowInsets(0, 0, 0, 0),
                snackbarHost = { SnackbarHost(hostState = snackbarHostState) },
                bottomBar = {
                    if (uiState.activeScreen == ActiveScreen.MAIN_TABS) {
                        GlassBottomNavBar(
                            selectedTab = uiState.selectedTab,
                            isTamil = uiState.preferences.isTamil,
                            onSelectTab = { tab -> appViewModel.selectBottomTab(tab) }
                        )
                    }
                }
            ) { innerPadding ->
                Box(
                    modifier = Modifier
                        .fillMaxSize()
                        .padding(innerPadding)
                ) {
                    Crossfade(
                        targetState = uiState.activeScreen to uiState.selectedTab,
                        label = "screen_transition"
                    ) { (screen, tab) ->
                        when (screen) {
                            ActiveScreen.MAIN_TABS -> {
                                when (tab) {
                                    BottomNavTab.HOME -> HomeScreen(
                                        preferences = uiState.preferences,
                                        metrics = uiState.metrics,
                                        onOpenSettings = {
                                            appViewModel.navigateToScreen(ActiveScreen.SETTINGS)
                                        },
                                        onSelectFeature = { feature ->
                                            appViewModel.navigateToScreen(feature)
                                        },
                                        onOpenModelsTab = {
                                            appViewModel.selectBottomTab(BottomNavTab.MODELS)
                                        }
                                    )

                                    BottomNavTab.FAVORITES -> CreationsCollectionScreen(
                                        title = if (uiState.preferences.isTamil) "Favorites (பிடித்தவை)" else "Favorites",
                                        emptySubtitle = if (uiState.preferences.isTamil) {
                                            "நீங்கள் நட்சத்திரமிட்ட படங்கள் இங்கே தோன்றும்."
                                        } else {
                                            "Starred offline creations will be saved right here."
                                        },
                                        creations = uiState.creations.filter { it.isFavorite },
                                        isFavoritesView = true,
                                        darkTheme = uiState.preferences.darkTheme,
                                        onStartCreating = {
                                            appViewModel.navigateToScreen(ActiveScreen.TEXT_TO_IMAGE)
                                        },
                                        onToggleFavorite = appViewModel::toggleFavoriteCreation,
                                        onSaveToGallery = appViewModel::exportCreationToGallery,
                                        onDeleteCreation = appViewModel::deleteCreation
                                    )

                                    BottomNavTab.MY_CREATIONS -> CreationsCollectionScreen(
                                        title = if (uiState.preferences.isTamil) "My Creations (என் படைப்புகள்)" else "My Creations",
                                        emptySubtitle = if (uiState.preferences.isTamil) {
                                            "உருவாக்கப்பட்ட அனைத்துப் படங்களும் இங்கே சேமிக்கப்படும்."
                                        } else {
                                            "All images generated offline on your device will appear here."
                                        },
                                        creations = uiState.creations,
                                        isFavoritesView = false,
                                        darkTheme = uiState.preferences.darkTheme,
                                        onStartCreating = {
                                            appViewModel.navigateToScreen(ActiveScreen.TEXT_TO_IMAGE)
                                        },
                                        onToggleFavorite = appViewModel::toggleFavoriteCreation,
                                        onSaveToGallery = appViewModel::exportCreationToGallery,
                                        onDeleteCreation = appViewModel::deleteCreation
                                    )

                                    BottomNavTab.MODELS -> ModelsTabScreen(
                                        preferences = uiState.preferences,
                                        metrics = uiState.metrics,
                                        isImporting = uiState.isImportingModels,
                                        importProgressText = uiState.importProgressText,
                                        verificationReport = uiState.onnxVerificationReport,
                                        preferredUnetMode = uiState.preferredUnetMode,
                                        autoDownloadProgress = uiState.autoDownloadProgress,
                                        onOpenAutoDownloadDialog = appViewModel::openAutoDownloadDialog,
                                        onCancelAutoDownload = appViewModel::cancelAutoDownload,
                                        onClearAutoDownloadBanner = appViewModel::clearAutoDownloadBanner,
                                        onSelectEngine = appViewModel::setActiveEngine,
                                        onSelectPreferredUnetMode = appViewModel::setPreferredUnetImportMode,
                                        onSwitchUnetFromFolder = { mode ->
                                            if (!appViewModel.switchUnetFromSavedFolder(mode)) {
                                                pendingFolderUnetMode = mode
                                                folderPickerLauncher.launch(null)
                                            }
                                        },
                                        onOneTouchFolderUpload = {
                                            pendingFolderUnetMode = null
                                            folderPickerLauncher.launch(null)
                                        },
                                        onVerifyPipeline = appViewModel::verifyOnnxPipeline,
                                        onDedicatedSlotUpload = { slot ->
                                            pendingDedicatedSlot = slot
                                            dedicatedSlotPickerLauncher.launch(arrayOf("*/*"))
                                        },
                                        onTokenizerSlotUpload = {
                                            tokenizerMultiPickerLauncher.launch(arrayOf("*/*"))
                                        },
                                        onDeleteDedicatedSlot = appViewModel::deleteDedicatedSlot,
                                        onSingleSlotUpload = { slot ->
                                            pendingSlotForSingleImport = slot
                                            singleModelPickerLauncher.launch(arrayOf("*/*"))
                                        },
                                        onAttachCompanionWeight = { slot ->
                                            pendingSlotForCompanionImport = slot
                                            companionWeightPickerLauncher.launch(arrayOf("*/*"))
                                        },
                                        onOpenSettingsDownloads = {
                                            appViewModel.navigateToScreen(ActiveScreen.SETTINGS)
                                        },
                                        onDeleteSlot = { slot ->
                                            appViewModel.deleteSlotModel(slot)
                                        }
                                    )
                                }
                            }

                            ActiveScreen.SETTINGS -> SettingsScreen(
                                preferences = uiState.preferences,
                                metrics = uiState.metrics,
                                onBack = { appViewModel.navigateBackToHome() },
                                onOpenModelsManager = {
                                    appViewModel.selectBottomTab(BottomNavTab.MODELS)
                                },
                                onOpenContentSafetyPage = {
                                    appViewModel.navigateToScreen(ActiveScreen.CONTENT_SAFETY_SETTINGS)
                                },
                                onResolutionChange = appViewModel::setResolution,
                                onInferenceModeChange = appViewModel::setInferenceMode,
                                onOfflineModeChange = appViewModel::setOfflineMode,
                                onLanguageChange = appViewModel::setLanguage,
                                onDarkThemeChange = appViewModel::setDarkTheme,
                                onNotificationsChange = appViewModel::setNotificationsEnabled,
                                onContentFilterToggle = appViewModel::setContentFilterEnabled,
                                onClearCache = appViewModel::clearCache,
                                onUnloadRam = appViewModel::unloadAiFromRam
                            )

                            ActiveScreen.CONTENT_SAFETY_SETTINGS -> ContentSafetySettingsScreen(
                                preferences = uiState.preferences,
                                currentUserPromptDraft = uiState.userPromptInput,
                                onBack = { appViewModel.navigateBackFromContentSafety() },
                                onToggleContentFilter = appViewModel::setContentFilterEnabled,
                                onSelectModePreset = appViewModel::selectSafetyModePreset,
                                onSaveContentSafetyPrompt = appViewModel::saveContentSafetyPrompt,
                                onResetContentSafetyPrompt = appViewModel::resetContentSafetyPrompt
                            )

                            ActiveScreen.TEXT_TO_IMAGE -> TextToImageScreen(
                                preferences = uiState.preferences,
                                metrics = uiState.metrics,
                                userPrompt = uiState.userPromptInput,
                                isGenerating = uiState.isGenerating,
                                progress = uiState.generationProgress,
                                latestCreation = uiState.latestGeneratedCreation,
                                errorMessage = uiState.generationErrorMessage,
                                isImportingModels = uiState.isImportingModels,
                                importProgressText = uiState.importProgressText,
                                onQuickSwitchToFastLcmFolder = {
                                    if (!appViewModel.switchUnetFromSavedFolder(PreferredUnetImportMode.FORCE_FAST_LCM)) {
                                        pendingFolderUnetMode = PreferredUnetImportMode.FORCE_FAST_LCM
                                        folderPickerLauncher.launch(null)
                                    }
                                },
                                onSelectEngine = appViewModel::setActiveEngine,
                                onResolutionChange = appViewModel::setResolution,
                                onUserPromptChange = appViewModel::updateUserImagePrompt,
                                onGenerate = appViewModel::generateOfflineTextToImage,
                                onCancelGeneration = appViewModel::cancelGeneration,
                                onExportToGallery = appViewModel::exportCreationToGallery,
                                onToggleFavorite = appViewModel::toggleFavoriteCreation,
                                onOpenModelsTab = {
                                    appViewModel.selectBottomTab(BottomNavTab.MODELS)
                                },
                                onOpenContentSafetySettings = {
                                    appViewModel.navigateToScreen(ActiveScreen.CONTENT_SAFETY_SETTINGS)
                                },
                                onBack = { appViewModel.navigateBackToHome() }
                            )

                            ActiveScreen.IMAGE_TO_IMAGE -> ImageToImageScreen(
                                preferences = uiState.preferences,
                                metrics = uiState.metrics,
                                selectedPhotoUri = uiState.selectedInputPhotoUri,
                                userPrompt = uiState.userPromptInput,
                                isGenerating = uiState.isGenerating,
                                progress = uiState.generationProgress,
                                latestImg2ImgCreation = uiState.latestImg2ImgCreation,
                                latestSourcePreviewPath = uiState.latestImg2ImgSourcePath,
                                errorMessage = uiState.generationErrorMessage,
                                onSelectEngine = appViewModel::setActiveEngine,
                                onPickReferencePhoto = {
                                    launchPhotoPicker(PhotoPickTarget.IMG2IMG_REFERENCE)
                                },
                                onUserPromptChange = appViewModel::updateUserImagePrompt,
                                onGenerateImg2Img = appViewModel::generateOfflineImageToImage,
                                onCancelGeneration = appViewModel::cancelGeneration,
                                onExportToGallery = appViewModel::exportCreationToGallery,
                                onToggleFavorite = appViewModel::toggleFavoriteCreation,
                                onOpenModelsTab = {
                                    appViewModel.selectBottomTab(BottomNavTab.MODELS)
                                },
                                onOpenContentSafetySettings = {
                                    appViewModel.navigateToScreen(ActiveScreen.CONTENT_SAFETY_SETTINGS)
                                },
                                onBack = { appViewModel.navigateBackToHome() }
                            )

                            ActiveScreen.CHARACTER_STUDIO -> CharacterStudioScreen(
                                preferences = uiState.preferences,
                                metrics = uiState.metrics,
                                characters = uiState.characters,
                                selectedCharacterId = uiState.selectedCharacterId,
                                pendingCharacterPhotoUri = uiState.pendingCharacterPhotoUri,
                                isGenerating = uiState.isGenerating,
                                progress = uiState.generationProgress,
                                latestCharacterCreation = uiState.latestCharacterCreation,
                                errorMessage = uiState.generationErrorMessage,
                                onSelectEngine = appViewModel::setActiveEngine,
                                onSelectCharacter = appViewModel::selectCharacter,
                                onPickCharacterPhoto = {
                                    launchPhotoPicker(PhotoPickTarget.CHARACTER_REFERENCE)
                                },
                                onClearCharacterPhoto = {
                                    appViewModel.setPendingCharacterPhotoUri(null)
                                },
                                onSaveCharacter = appViewModel::saveCharacterProfile,
                                onDeleteCharacter = appViewModel::deleteCharacterProfile,
                                onGenerateCharacterScene = appViewModel::generateCharacterScene,
                                onCancelGeneration = appViewModel::cancelGeneration,
                                onExportToGallery = appViewModel::exportCreationToGallery,
                                onToggleFavorite = appViewModel::toggleFavoriteCreation,
                                onOpenModelsTab = {
                                    appViewModel.selectBottomTab(BottomNavTab.MODELS)
                                },
                                onOpenContentSafetySettings = {
                                    appViewModel.navigateToScreen(ActiveScreen.CONTENT_SAFETY_SETTINGS)
                                },
                                onBack = { appViewModel.navigateBackToHome() }
                            )

                            ActiveScreen.FACE_SWAP -> FaceSwapScreen(
                                preferences = uiState.preferences,
                                metrics = uiState.metrics,
                                sourceFaceUri = uiState.faceSwapSourceUri,
                                targetImageUri = uiState.faceSwapTargetUri,
                                targetFallbackCreation = uiState.latestGeneratedCreation,
                                useLatestCreationAsTarget = uiState.useLatestCreationAsFaceSwapTarget,
                                isSwapping = uiState.isGenerating,
                                progress = uiState.generationProgress,
                                latestFaceSwapCreation = uiState.latestFaceSwapCreation,
                                latestFaceSwapTargetPreviewPath = uiState.latestFaceSwapTargetPreviewPath,
                                errorMessage = uiState.generationErrorMessage,
                                onPickSourceFace = {
                                    launchPhotoPicker(PhotoPickTarget.FACESWAP_SOURCE)
                                },
                                onPickTargetPhoto = {
                                    launchPhotoPicker(PhotoPickTarget.FACESWAP_TARGET)
                                },
                                onUseLatestCreationAsTarget = appViewModel::useLatestCreationAsFaceSwapTarget,
                                onRunFaceSwap = appViewModel::runOfflineFaceSwap,
                                onCancelSwap = appViewModel::cancelGeneration,
                                onExportToGallery = appViewModel::exportCreationToGallery,
                                onToggleFavorite = appViewModel::toggleFavoriteCreation,
                                onOpenModelsTab = {
                                    appViewModel.selectBottomTab(BottomNavTab.MODELS)
                                },
                                onBack = { appViewModel.navigateBackToHome() }
                            )

                            ActiveScreen.IMAGE_EDITOR -> ImageEditorScreen(
                                preferences = uiState.preferences,
                                editorImageUri = uiState.editorImageUri,
                                latestCreation = uiState.latestGeneratedCreation,
                                useLatestCreationInEditor = uiState.useLatestCreationInEditor,
                                latestEditedCreation = uiState.latestEditedCreation,
                                isSavingEdit = uiState.isSavingEdit,
                                onPickEditorPhoto = {
                                    launchPhotoPicker(PhotoPickTarget.IMAGE_EDITOR_SOURCE)
                                },
                                onLoadLatestCreation = appViewModel::loadLatestCreationInEditor,
                                onSaveEditedPhoto = appViewModel::saveEditedPhoto,
                                onExportToGallery = appViewModel::exportCreationToGallery,
                                onBack = { appViewModel.navigateBackToHome() }
                            )

                            ActiveScreen.MY_GALLERY -> CreationsCollectionScreen(
                                title = "My Gallery",
                                emptySubtitle = if (uiState.preferences.isTamil) {
                                    "நீங்கள் உருவாக்கும் படங்கள் இங்கே சேமிக்கப்படும்."
                                } else {
                                    "Your offline generated photos will appear here."
                                },
                                creations = uiState.creations,
                                isFavoritesView = false,
                                darkTheme = uiState.preferences.darkTheme,
                                showBackButton = true,
                                onBack = { appViewModel.navigateBackToHome() },
                                onStartCreating = {
                                    appViewModel.navigateToScreen(ActiveScreen.TEXT_TO_IMAGE)
                                },
                                onToggleFavorite = appViewModel::toggleFavoriteCreation,
                                onSaveToGallery = appViewModel::exportCreationToGallery,
                                onDeleteCreation = appViewModel::deleteCreation
                            )
                        }
                    }

                    // Show floating live download banner when user navigates outside the Models tab while downloading
                    val showOverlayBanner = !(uiState.activeScreen == ActiveScreen.MAIN_TABS &&
                        uiState.selectedTab == BottomNavTab.MODELS)
                    if (showOverlayBanner) {
                        AutoDownloadLiveBanner(
                            progress = uiState.autoDownloadProgress,
                            isTamil = uiState.preferences.isTamil,
                            darkTheme = uiState.preferences.darkTheme,
                            onCancelDownload = appViewModel::cancelAutoDownload,
                            onResumeOrRetry = appViewModel::openAutoDownloadDialog,
                            onDismissBanner = appViewModel::clearAutoDownloadBanner,
                            modifier = Modifier
                                .align(Alignment.TopCenter)
                                .windowInsetsPadding(WindowInsets.statusBars)
                                .padding(horizontal = 14.dp, vertical = 6.dp)
                        )
                    }

                    // First-Launch Permission & Auto-Download Dialog (1A, 1B, 2A, 2B, 3, 4, 5, 6)
                    if (uiState.showFirstLaunchDownloadDialog) {
                        FirstLaunchModelDownloadDialog(
                            metrics = uiState.metrics,
                            isTamil = uiState.preferences.isTamil,
                            darkTheme = uiState.preferences.darkTheme,
                            onAllowAndDownload = appViewModel::confirmAndStartAutoDownload,
                            onDismissLater = appViewModel::dismissFirstLaunchDownloadDialog
                        )
                    }
                }
            }
        }
    }
}
