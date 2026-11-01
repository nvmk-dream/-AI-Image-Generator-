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
import androidx.compose.material3.Scaffold
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.example.ai.DiffusionModelSlot
import com.example.ui.ActiveScreen
import com.example.ui.AppViewModel
import com.example.ui.BottomNavTab
import com.example.ui.components.DreamyHeartBackground
import com.example.ui.components.GlassBottomNavBar
import com.example.ui.screens.ContentSafetySettingsScreen
import com.example.ui.screens.CreationsCollectionScreen
import com.example.ui.screens.HomeScreen
import com.example.ui.screens.ImageToImageScreen
import com.example.ui.screens.ModelsTabScreen
import com.example.ui.screens.SettingsScreen
import com.example.ui.screens.StudioWorkspaceScreen
import com.example.ui.screens.TextToImageScreen
import com.example.ui.theme.MyApplicationTheme

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
    var pendingSlotForSingleImport by remember { mutableStateOf(DiffusionModelSlot.TEXT_ENCODER) }

    val folderPickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenDocumentTree()
    ) { treeUri ->
        if (treeUri != null) {
            runCatching {
                context.contentResolver.takePersistableUriPermission(
                    treeUri,
                    Intent.FLAG_GRANT_READ_URI_PERMISSION
                )
            }
            appViewModel.importAllModelsFromFolder(treeUri)
        }
    }

    val singleModelPickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenDocument()
    ) { fileUri ->
        if (fileUri != null) {
            appViewModel.importSingleSlot(fileUri, pendingSlotForSingleImport)
        }
    }

    val photoPickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.PickVisualMedia()
    ) { photoUri ->
        if (photoUri != null) {
            appViewModel.updateSelectedInputPhoto(photoUri)
        }
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
                                        onOneTouchFolderUpload = {
                                            folderPickerLauncher.launch(null)
                                        },
                                        onVerifyPipeline = appViewModel::verifyOnnxPipeline,
                                        onSingleSlotUpload = { slot ->
                                            pendingSlotForSingleImport = slot
                                            singleModelPickerLauncher.launch(arrayOf("*/*"))
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
                                onPickReferencePhoto = {
                                    photoPickerLauncher.launch(
                                        PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)
                                    )
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

                            else -> StudioWorkspaceScreen(
                                screen = screen,
                                preferences = uiState.preferences,
                                metrics = uiState.metrics,
                                promptDraft = uiState.userPromptInput,
                                selectedPhotoUri = uiState.selectedInputPhotoUri,
                                onPromptChange = appViewModel::updateUserImagePrompt,
                                onPickPhoto = {
                                    photoPickerLauncher.launch(
                                        PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)
                                    )
                                },
                                onOpenModelsTab = {
                                    appViewModel.selectBottomTab(BottomNavTab.MODELS)
                                },
                                onBack = { appViewModel.navigateBackToHome() }
                            )
                        }
                    }
                }
            }
        }
    }
}
