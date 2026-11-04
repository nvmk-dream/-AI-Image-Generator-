package com.example.ui.screens

import android.graphics.Bitmap
import android.net.Uri
import androidx.activity.compose.BackHandler
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBars
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBars
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.AutoAwesome
import androidx.compose.material.icons.filled.CameraAlt
import androidx.compose.material.icons.filled.Cancel
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.Checkroom
import androidx.compose.material.icons.filled.CloudDownload
import androidx.compose.material.icons.filled.Compare
import androidx.compose.material.icons.filled.ContentCut
import androidx.compose.material.icons.filled.Download
import androidx.compose.material.icons.filled.Elderly
import androidx.compose.material.icons.filled.FitnessCenter
import androidx.compose.material.icons.filled.FolderSpecial
import androidx.compose.material.icons.filled.Lock
import androidx.compose.material.icons.filled.Memory
import androidx.compose.material.icons.filled.Security
import androidx.compose.material.icons.filled.Shuffle
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.minimumInteractiveComponentSize
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import coil.compose.AsyncImage
import com.example.ai.AvatarCustomTab
import com.example.ai.AvatarCustomizationCatalog
import com.example.ai.AvatarCustomizationParams
import com.example.ai.AvatarVisionAnalysis
import com.example.ai.GenerationProgress
import com.example.ai.HairShadeOption
import com.example.ai.MaskBrushStroke
import com.example.ai.OnnxModelManager
import com.example.ai.OutfitFabricOption
import com.example.ai.RealDeviceAndModelMetrics
import com.example.ai.RealisticAvatarPreset
import com.example.data.ActiveDiffusionEngine
import com.example.data.AppPreferencesState
import com.example.data.GeneratedCreation
import com.example.ui.components.ActiveEngineSwitchCard
import com.example.ui.components.AvatarInteractivePreviewCard
import com.example.ui.components.AvatarLabeledSlider
import com.example.ui.components.CircularGlowIconButton
import com.example.ui.components.FrostedGlassCard
import com.example.ui.components.RealisticPhotoPresetRow
import com.example.ui.theme.DeepPlumText
import com.example.ui.theme.NeonReadyGreen
import com.example.ui.theme.PinkGlowEnd
import com.example.ui.theme.PinkGlowStart
import com.example.ui.theme.PurpleActionEnd
import com.example.ui.theme.PurpleActionStart
import com.example.ui.theme.SoftAmberStatus
import com.example.ui.theme.VibrantPinkPill
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import java.io.File
import kotlin.math.roundToInt
import kotlin.random.Random

@Composable
fun AvatarCustomizationScreen(
    preferences: AppPreferencesState,
    metrics: RealDeviceAndModelMetrics,
    sourcePhotoUri: Uri?,
    hasLatestAppCreation: Boolean,
    params: AvatarCustomizationParams,
    userBrushStrokes: List<MaskBrushStroke>,
    visionAnalysis: AvatarVisionAnalysis?,
    maskOverlayBitmap: Bitmap?,
    latestAvatarCreation: GeneratedCreation?,
    latestAvatarSourcePath: String?,
    isGenerating: Boolean,
    progress: GenerationProgress?,
    errorMessage: String?,
    isDownloadingMediaPipe: Boolean,
    mediaPipeDownloadStatus: String?,
    onSelectEngine: (ActiveDiffusionEngine) -> Unit,
    onPickSourcePhoto: () -> Unit,
    onUseLatestAppCreation: () -> Unit,
    onParamsChange: (AvatarCustomizationParams) -> Unit,
    onAddBrushStroke: (MaskBrushStroke) -> Unit,
    onUndoBrushStroke: () -> Unit,
    onResetBrushStrokes: () -> Unit,
    onDownloadMediaPipeTasks: () -> Unit,
    onGenerateAvatar: () -> Unit,
    onCancelGeneration: () -> Unit,
    onUseResultAsNextSource: () -> Unit,
    onToggleFavorite: (String) -> Unit,
    onExportToGallery: (GeneratedCreation) -> Unit,
    onOpenModelsTab: () -> Unit,
    onOpenContentSafetySettings: () -> Unit,
    onBack: () -> Unit,
    modifier: Modifier = Modifier
) {
    BackHandler { onBack() }

    val isTamil = preferences.isTamil
    val darkTheme = preferences.darkTheme
    val textColor = if (darkTheme) Color.White else DeepPlumText
    val subColor = if (darkTheme) Color(0xFFE1C9F7) else Color(0xFF4F376E)

    var liveElapsedSec by remember { mutableIntStateOf(0) }
    LaunchedEffect(isGenerating) {
        if (isGenerating) {
            val started = System.currentTimeMillis()
            liveElapsedSec = 0
            while (true) {
                delay(1000L)
                liveElapsedSec = ((System.currentTimeMillis() - started) / 1000L).toInt()
            }
        } else {
            liveElapsedSec = 0
        }
    }

    val mainScrollState = rememberScrollState()
    val coroutineScope = rememberCoroutineScope()
    var lastAutoScrolledCreationId by remember { mutableStateOf(latestAvatarCreation?.id) }

    LaunchedEffect(latestAvatarCreation?.id) {
        val currentId = latestAvatarCreation?.id
        if (currentId != null && currentId != lastAutoScrolledCreationId) {
            lastAutoScrolledCreationId = currentId
            mainScrollState.animateScrollTo(0)
        }
    }

    val (_, strippedForbidden) = remember(params.customRealisticNote) {
        AvatarCustomizationCatalog.sanitizeCustomRealisticNote(params.customRealisticNote)
    }

    Column(
        horizontalAlignment = Alignment.CenterHorizontally,
        modifier = modifier
            .fillMaxSize()
            .windowInsetsPadding(WindowInsets.statusBars)
            .windowInsetsPadding(WindowInsets.navigationBars)
            .verticalScroll(mainScrollState)
            .padding(horizontal = 16.dp)
            .testTag("avatar_customization_screen")
    ) {
        Column(
            verticalArrangement = Arrangement.spacedBy(12.dp),
            modifier = Modifier
                .fillMaxWidth()
                .widthIn(max = 600.dp)
                .padding(vertical = 10.dp)
        ) {
            // Top Bar
            Row(
                verticalAlignment = Alignment.CenterVertically,
                modifier = Modifier.fillMaxWidth()
            ) {
                CircularGlowIconButton(
                    onClick = onBack,
                    contentDescription = "Back",
                    testTag = "avatar_back_button"
                ) {
                    Icon(
                        imageVector = Icons.AutoMirrored.Filled.ArrowBack,
                        contentDescription = "Back",
                        tint = Color.White
                    )
                }
                Spacer(modifier = Modifier.width(12.dp))
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        text = if (isTamil) "Avatar Customization" else "Avatar Customization",
                        color = textColor,
                        style = MaterialTheme.typography.headlineSmall.copy(fontWeight = FontWeight.ExtraBold)
                    )
                    Text(
                        text = if (isTamil) {
                            "100% இயற்கை புகைப்படம் (DSLR) • ஆஃப்லைன் AI"
                        } else {
                            "100% Natural Realistic DSLR Photography • Offline AI"
                        },
                        color = subColor,
                        style = MaterialTheme.typography.bodySmall.copy(fontWeight = FontWeight.SemiBold)
                    )
                }
                CircularGlowIconButton(
                    onClick = onOpenContentSafetySettings,
                    contentDescription = "Content Safety Settings",
                    testTag = "avatar_open_safety_button"
                ) {
                    Icon(
                        imageVector = Icons.Filled.Security,
                        contentDescription = "Safety",
                        tint = Color.White
                    )
                }
            }

            // Strict Camera Realism Banner + Hardware / MediaPipe / Slot 7 Status Card
            FrostedGlassCard(darkTheme = darkTheme) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Icon(
                        imageVector = Icons.Filled.CameraAlt,
                        contentDescription = null,
                        tint = NeonReadyGreen,
                        modifier = Modifier.size(20.dp)
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(
                        text = if (isTamil) {
                            "Realistic DSLR Photo Mode (Painting / 3D / Animation தடை செய்யப்பட்டுள்ளது)"
                        } else {
                            "Strict Realistic DSLR Photo Mode (No Painting / 3D / Animation)"
                        },
                        color = textColor,
                        style = MaterialTheme.typography.labelLarge.copy(fontWeight = FontWeight.ExtraBold),
                        modifier = Modifier.weight(1f)
                    )
                }

                Spacer(modifier = Modifier.height(6.dp))

                val mpStatus = metrics.mediaPipeTasksStatus
                val ramAvail = OnnxModelManager.formatBytes(metrics.deviceAvailableRamBytes)
                val slot7Badge = if (metrics.isAgingSlotInstalled) {
                    if (isTamil) "Slot 7 Aging: தயார் ✓" else "Slot 7 Aging: Ready ✓"
                } else {
                    if (isTamil) "Slot 7: Core 1A–6 Mode" else "Slot 7: Core 1A–6 Mode"
                }

                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Icon(
                        imageVector = Icons.Filled.Memory,
                        contentDescription = null,
                        tint = VibrantPinkPill,
                        modifier = Modifier.size(16.dp)
                    )
                    Spacer(modifier = Modifier.width(6.dp))
                    Text(
                        text = "${preferences.inferenceMode} • RAM Free: $ramAvail • $slot7Badge",
                        color = subColor,
                        style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.SemiBold)
                    )
                }

                Spacer(modifier = Modifier.height(8.dp))

                // MediaPipe Tasks Vision Status / 1-Tap Setup Row
                if (mpStatus.allTasksReady) {
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        modifier = Modifier
                            .fillMaxWidth()
                            .clip(RoundedCornerShape(12.dp))
                            .background(Color.White.copy(alpha = if (darkTheme) 0.10f else 0.65f))
                            .border(1.dp, NeonReadyGreen.copy(alpha = 0.75f), RoundedCornerShape(12.dp))
                            .padding(horizontal = 10.dp, vertical = 7.dp)
                    ) {
                        Icon(
                            imageVector = Icons.Filled.CheckCircle,
                            contentDescription = null,
                            tint = NeonReadyGreen,
                            modifier = Modifier.size(17.dp)
                        )
                        Spacer(modifier = Modifier.width(6.dp))
                        Text(
                            text = if (isTamil) {
                                "MediaPipe Vision (Face 478 + Pose 33 + Hair/Clothes Segmenter) 3/3 தயார் ✓"
                            } else {
                                "MediaPipe Vision (Face 478 + Pose 33 + Multiclass Segmenter) 3/3 Ready ✓"
                            },
                            color = textColor,
                            style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.Bold)
                        )
                    }
                } else {
                    Column(
                        modifier = Modifier
                            .fillMaxWidth()
                            .clip(RoundedCornerShape(14.dp))
                            .background(Color.White.copy(alpha = if (darkTheme) 0.10f else 0.65f))
                            .border(1.dp, SoftAmberStatus.copy(alpha = 0.75f), RoundedCornerShape(14.dp))
                            .padding(10.dp)
                    ) {
                        Text(
                            text = if (isTamil) {
                                "MediaPipe Tasks Vision (${mpStatus.installedCount}/3): துல்லியமான முடி/உடல்/ஆடை பிரிப்புக்கு 1-Tap பதிவிறக்கம் (~25 MB • ஆஃப்லைன் FaceDetector உடனடியாக செயலில் உள்ளது)"
                            } else {
                                "MediaPipe Tasks Vision (${mpStatus.installedCount}/3): Download ~25 MB Face/Pose/Segmenter tasks once for pixel-exact Hair & Clothes masks (Offline FaceDetector active now)"
                            },
                            color = textColor,
                            style = MaterialTheme.typography.labelSmall
                        )
                        if (!mediaPipeDownloadStatus.isNullOrBlank()) {
                            Spacer(modifier = Modifier.height(4.dp))
                            Text(
                                text = mediaPipeDownloadStatus,
                                color = VibrantPinkPill,
                                style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.Bold)
                            )
                        }
                        Spacer(modifier = Modifier.height(6.dp))
                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.Center,
                            modifier = Modifier
                                .fillMaxWidth()
                                .minimumInteractiveComponentSize()
                                .clip(RoundedCornerShape(12.dp))
                                .background(Brush.horizontalGradient(listOf(PurpleActionStart, VibrantPinkPill)))
                                .clickable(enabled = !isDownloadingMediaPipe && !isGenerating, onClick = onDownloadMediaPipeTasks)
                                .padding(vertical = 8.dp, horizontal = 10.dp)
                                .testTag("avatar_download_mediapipe_button")
                        ) {
                            if (isDownloadingMediaPipe) {
                                CircularProgressIndicator(
                                    color = Color.White,
                                    strokeWidth = 2.dp,
                                    modifier = Modifier.size(16.dp)
                                )
                            } else {
                                Icon(
                                    imageVector = Icons.Filled.CloudDownload,
                                    contentDescription = null,
                                    tint = Color.White,
                                    modifier = Modifier.size(17.dp)
                                )
                            }
                            Spacer(modifier = Modifier.width(6.dp))
                            Text(
                                text = if (isDownloadingMediaPipe) {
                                    if (isTamil) "MediaPipe பதிவிறக்கம் நடைபெறுகிறது..." else "Downloading MediaPipe Tasks..."
                                } else {
                                    if (isTamil) "MediaPipe Vision (3/3) பதிவிறக்கு" else "1-Tap Download MediaPipe Vision (~25 MB)"
                                },
                                color = Color.White,
                                style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.ExtraBold)
                            )
                        }
                    }
                }
            }

            // Active Engine Switch (Fast LCM vs Standard SD)
            ActiveEngineSwitchCard(
                activeEngine = preferences.activeEngine,
                metrics = metrics,
                isTamil = isTamil,
                darkTheme = darkTheme,
                enabled = !isGenerating,
                tagPrefix = "avatar",
                onSelectEngine = onSelectEngine
            )

            // 1. Interactive Realistic Photo & AI Mask Canvas
            AvatarInteractivePreviewCard(
                sourcePhotoUri = sourcePhotoUri,
                hasLatestAppCreation = hasLatestAppCreation,
                visionAnalysis = visionAnalysis,
                maskOverlayBitmap = maskOverlayBitmap,
                userStrokesCount = userBrushStrokes.size,
                latestAvatarCreation = latestAvatarCreation,
                latestAvatarSourcePath = latestAvatarSourcePath,
                isGenerating = isGenerating,
                isTamil = isTamil,
                darkTheme = darkTheme,
                onPickPhoto = onPickSourcePhoto,
                onUseLatestAppCreation = onUseLatestAppCreation,
                onAddBrushStroke = onAddBrushStroke,
                onUndoBrushStroke = onUndoBrushStroke,
                onResetBrushStrokes = onResetBrushStrokes,
                onUseResultAsNextSource = onUseResultAsNextSource,
                onToggleFavorite = onToggleFavorite,
                onExportToGallery = onExportToGallery
            )

            // 2. 4 Dedicated Module Tabs (Age | Body/Abs | Hairstyle | Outfit)
            FrostedGlassCard(darkTheme = darkTheme) {
                Text(
                    text = if (isTamil) {
                        "2. மாற்ற வேண்டிய பகுதி (4 Offline AI Modules)"
                    } else {
                        "2. Select Attribute Module (4 Offline AI Engines)"
                    },
                    color = textColor,
                    style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.ExtraBold)
                )
                Spacer(modifier = Modifier.height(8.dp))

                Row(
                    horizontalArrangement = Arrangement.spacedBy(6.dp),
                    modifier = Modifier.fillMaxWidth()
                ) {
                    AvatarModuleTabPill(
                        tab = AvatarCustomTab.AGE,
                        icon = Icons.Filled.Elderly,
                        isSelected = params.activeTab == AvatarCustomTab.AGE,
                        isTamil = isTamil,
                        darkTheme = darkTheme,
                        enabled = !isGenerating,
                        testTag = "avatar_tab_age",
                        onClick = { selectTabWithDefaultPreset(AvatarCustomTab.AGE, params, onParamsChange) },
                        modifier = Modifier.weight(1f)
                    )
                    AvatarModuleTabPill(
                        tab = AvatarCustomTab.BODY_ABS,
                        icon = Icons.Filled.FitnessCenter,
                        isSelected = params.activeTab == AvatarCustomTab.BODY_ABS,
                        isTamil = isTamil,
                        darkTheme = darkTheme,
                        enabled = !isGenerating,
                        testTag = "avatar_tab_body_abs",
                        onClick = { selectTabWithDefaultPreset(AvatarCustomTab.BODY_ABS, params, onParamsChange) },
                        modifier = Modifier.weight(1f)
                    )
                    AvatarModuleTabPill(
                        tab = AvatarCustomTab.HAIRSTYLE,
                        icon = Icons.Filled.ContentCut,
                        isSelected = params.activeTab == AvatarCustomTab.HAIRSTYLE,
                        isTamil = isTamil,
                        darkTheme = darkTheme,
                        enabled = !isGenerating,
                        testTag = "avatar_tab_hairstyle",
                        onClick = { selectTabWithDefaultPreset(AvatarCustomTab.HAIRSTYLE, params, onParamsChange) },
                        modifier = Modifier.weight(1f)
                    )
                    AvatarModuleTabPill(
                        tab = AvatarCustomTab.OUTFIT,
                        icon = Icons.Filled.Checkroom,
                        isSelected = params.activeTab == AvatarCustomTab.OUTFIT,
                        isTamil = isTamil,
                        darkTheme = darkTheme,
                        enabled = !isGenerating,
                        testTag = "avatar_tab_outfit",
                        onClick = { selectTabWithDefaultPreset(AvatarCustomTab.OUTFIT, params, onParamsChange) },
                        modifier = Modifier.weight(1f)
                    )
                }

                Spacer(modifier = Modifier.height(12.dp))

                // Realistic DSLR Photo Presets for Active Tab
                val tabPresets = remember(params.activeTab) {
                    AvatarCustomizationCatalog.presetsForTab(params.activeTab)
                }
                Text(
                    text = if (isTamil) {
                        "Realistic Photo Presets (${params.activeTab.titleTa})"
                    } else {
                        "Realistic DSLR Photo Presets (${params.activeTab.titleEn})"
                    },
                    color = textColor,
                    style = MaterialTheme.typography.titleSmall.copy(fontWeight = FontWeight.ExtraBold)
                )
                Spacer(modifier = Modifier.height(6.dp))

                RealisticPhotoPresetRow(
                    presets = tabPresets,
                    selectedPresetId = params.selectedPresetId,
                    isTamil = isTamil,
                    darkTheme = darkTheme,
                    enabled = !isGenerating,
                    onSelectPreset = { preset ->
                        onParamsChange(applyPresetDefaults(params, preset))
                    }
                )
            }

            // 3. Module-Specific User-Controlled Sliders Card
            FrostedGlassCard(darkTheme = darkTheme) {
                Text(
                    text = if (isTamil) params.activeTab.titleTa else params.activeTab.titleEn,
                    color = textColor,
                    style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.ExtraBold)
                )
                Spacer(modifier = Modifier.height(8.dp))

                when (params.activeTab) {
                    AvatarCustomTab.AGE -> {
                        AvatarLabeledSlider(
                            title = if (isTamil) "வயது (Age Slider: 16 – 80)" else "Target Age (16 to 80 Years)",
                            valueBadge = "${params.targetAge} Yrs",
                            value = params.targetAge.toFloat(),
                            valueRange = 16f..80f,
                            enabled = !isGenerating,
                            darkTheme = darkTheme,
                            testTag = "avatar_slider_target_age",
                            onValueChange = { ageVal ->
                                val newAge = ageVal.roundToInt().coerceIn(16, 80)
                                onParamsChange(params.copy(targetAge = newAge))
                            }
                        )
                        AvatarLabeledSlider(
                            title = if (isTamil) "தோல் சுருக்கம் & இயற்கை அமைப்பு" else "Skin Wrinkle & Pore Texture",
                            valueBadge = AvatarCustomizationCatalog.formatPercent(params.skinWrinkleIntensity),
                            value = params.skinWrinkleIntensity,
                            valueRange = 0f..1f,
                            enabled = !isGenerating,
                            darkTheme = darkTheme,
                            testTag = "avatar_slider_wrinkle_intensity",
                            onValueChange = { onParamsChange(params.copy(skinWrinkleIntensity = it)) }
                        )
                        AvatarLabeledSlider(
                            title = if (isTamil) "நரைமுடி அளவு (Silver / Grey Hair)" else "Natural Silver / Grey Hair Ratio",
                            valueBadge = AvatarCustomizationCatalog.formatPercent(params.silverHairRatio),
                            value = params.silverHairRatio,
                            valueRange = 0f..1f,
                            enabled = !isGenerating,
                            darkTheme = darkTheme,
                            testTag = "avatar_slider_silver_hair",
                            onValueChange = { onParamsChange(params.copy(silverHairRatio = it)) }
                        )
                        AvatarLabeledSlider(
                            title = if (isTamil) "முக எலும்பு & கண் அடையாளப் பாதுகாப்பு" else "Facial Bone & Eye Structure Lock",
                            valueBadge = AvatarCustomizationCatalog.formatPercent(params.facialBoneConsistency),
                            value = params.facialBoneConsistency,
                            valueRange = 0.40f..0.98f,
                            enabled = !isGenerating,
                            darkTheme = darkTheme,
                            testTag = "avatar_slider_bone_consistency",
                            onValueChange = { onParamsChange(params.copy(facialBoneConsistency = it)) }
                        )
                    }

                    AvatarCustomTab.BODY_ABS -> {
                        IdentityProtectionNotice(
                            text = if (isTamil) {
                                "🔒 உங்கள் அசல் முகம் மற்றும் பின்னணி 100% மாற்றமின்றி பாதுகாக்கப்படும்"
                            } else {
                                "🔒 Original Face, Head & Background are 100% Identity-Locked"
                            },
                            darkTheme = darkTheme
                        )
                        Spacer(modifier = Modifier.height(6.dp))
                        AvatarLabeledSlider(
                            title = if (isTamil) "சிக்ஸ்-பேக் & வயிற்றுத் தசை (Six-Pack Abs)" else "Six-Pack Abs & Core Definition",
                            valueBadge = AvatarCustomizationCatalog.formatPercent(params.absCoreDefinition),
                            value = params.absCoreDefinition,
                            valueRange = 0f..1f,
                            enabled = !isGenerating,
                            darkTheme = darkTheme,
                            testTag = "avatar_slider_abs_definition",
                            onValueChange = { onParamsChange(params.copy(absCoreDefinition = it)) }
                        )
                        AvatarLabeledSlider(
                            title = if (isTamil) "தோள்பட்டை & மார்பு தசை அளவு" else "Shoulder & Chest Muscle Mass",
                            valueBadge = AvatarCustomizationCatalog.formatPercent(params.muscleShoulderMass),
                            value = params.muscleShoulderMass,
                            valueRange = 0f..1f,
                            enabled = !isGenerating,
                            darkTheme = darkTheme,
                            testTag = "avatar_slider_muscle_mass",
                            onValueChange = { onParamsChange(params.copy(muscleShoulderMass = it)) }
                        )
                        AvatarLabeledSlider(
                            title = if (isTamil) "இடுப்பு & ஃபிட்னஸ் வடிவமைப்பு" else "Waistline & Athletic Posture Trim",
                            valueBadge = AvatarCustomizationCatalog.formatPercent(params.waistBodyTrim),
                            value = params.waistBodyTrim,
                            valueRange = 0f..1f,
                            enabled = !isGenerating,
                            darkTheme = darkTheme,
                            testTag = "avatar_slider_waist_trim",
                            onValueChange = { onParamsChange(params.copy(waistBodyTrim = it)) }
                        )
                    }

                    AvatarCustomTab.HAIRSTYLE -> {
                        IdentityProtectionNotice(
                            text = if (isTamil) {
                                "🔒 கண்கள், மூக்கு, உதடு கொண்ட முகப்பகுதி 100% பாதுகாக்கப்படும்"
                            } else {
                                "🔒 Inner Face Oval (Eyes, Nose, Lips) is 100% Identity-Locked"
                            },
                            darkTheme = darkTheme
                        )
                        Spacer(modifier = Modifier.height(8.dp))
                        Text(
                            text = if (isTamil) "இயற்கை முடி நிறம் (Natural Hair Shade)" else "Natural Human Hair Shade",
                            color = textColor,
                            style = MaterialTheme.typography.labelLarge.copy(fontWeight = FontWeight.Bold)
                        )
                        Spacer(modifier = Modifier.height(6.dp))
                        Row(
                            horizontalArrangement = Arrangement.spacedBy(6.dp),
                            modifier = Modifier
                                .fillMaxWidth()
                                .horizontalScroll(rememberScrollState())
                        ) {
                            HairShadeOption.entries.forEach { shade ->
                                SelectableOptionChip(
                                    label = if (isTamil) shade.labelTa else shade.labelEn,
                                    selected = params.hairShade == shade,
                                    darkTheme = darkTheme,
                                    enabled = !isGenerating,
                                    testTag = "avatar_hair_shade_${shade.id}",
                                    onClick = { onParamsChange(params.copy(hairShade = shade)) }
                                )
                            }
                        }
                        Spacer(modifier = Modifier.height(8.dp))
                        AvatarLabeledSlider(
                            title = if (isTamil) "முடி அடர்த்தி (Hair Volume & Density)" else "Hair Volume & Strand Density",
                            valueBadge = AvatarCustomizationCatalog.formatPercent(params.hairVolumeDensity),
                            value = params.hairVolumeDensity,
                            valueRange = 0f..1f,
                            enabled = !isGenerating,
                            darkTheme = darkTheme,
                            testTag = "avatar_slider_hair_volume",
                            onValueChange = { onParamsChange(params.copy(hairVolumeDensity = it)) }
                        )
                        AvatarLabeledSlider(
                            title = if (isTamil) "முடி நீளம் (Hair Length Scale)" else "Hair Length Scale",
                            valueBadge = AvatarCustomizationCatalog.formatPercent(params.hairLengthScale),
                            value = params.hairLengthScale,
                            valueRange = 0f..1f,
                            enabled = !isGenerating,
                            darkTheme = darkTheme,
                            testTag = "avatar_slider_hair_length",
                            onValueChange = { onParamsChange(params.copy(hairLengthScale = it)) }
                        )
                        AvatarLabeledSlider(
                            title = if (isTamil) "இயற்கை பளபளப்பு (Natural Hair Sheen)" else "Natural Daylight Hair Sheen",
                            valueBadge = AvatarCustomizationCatalog.formatPercent(params.hairNaturalShine),
                            value = params.hairNaturalShine,
                            valueRange = 0f..1f,
                            enabled = !isGenerating,
                            darkTheme = darkTheme,
                            testTag = "avatar_slider_hair_shine",
                            onValueChange = { onParamsChange(params.copy(hairNaturalShine = it)) }
                        )
                    }

                    AvatarCustomTab.OUTFIT -> {
                        IdentityProtectionNotice(
                            text = if (isTamil) {
                                "🔒 முகம் மற்றும் கைகள் 100% பாதுகாக்கப்பட்டு ஆடை மட்டும் மாற்றப்படும்"
                            } else {
                                "🔒 Face & Hands Protected — Realistic Cloth Folding Only"
                            },
                            darkTheme = darkTheme
                        )
                        Spacer(modifier = Modifier.height(8.dp))
                        Text(
                            text = if (isTamil) "துணி வகை (Authentic Fabric Texture)" else "Authentic Garment Fabric",
                            color = textColor,
                            style = MaterialTheme.typography.labelLarge.copy(fontWeight = FontWeight.Bold)
                        )
                        Spacer(modifier = Modifier.height(6.dp))
                        Row(
                            horizontalArrangement = Arrangement.spacedBy(6.dp),
                            modifier = Modifier
                                .fillMaxWidth()
                                .horizontalScroll(rememberScrollState())
                        ) {
                            OutfitFabricOption.entries.forEach { fabric ->
                                SelectableOptionChip(
                                    label = if (isTamil) fabric.labelTa else fabric.labelEn,
                                    selected = params.outfitFabric == fabric,
                                    darkTheme = darkTheme,
                                    enabled = !isGenerating,
                                    testTag = "avatar_fabric_${fabric.id}",
                                    onClick = { onParamsChange(params.copy(outfitFabric = fabric)) }
                                )
                            }
                        }
                        Spacer(modifier = Modifier.height(8.dp))
                        AvatarLabeledSlider(
                            title = if (isTamil) "இயற்கை துணி மடிப்பு (Cloth Draping Realism)" else "Pose-Aware Cloth Folding & Drape",
                            valueBadge = AvatarCustomizationCatalog.formatPercent(params.clothDrapingRealism),
                            value = params.clothDrapingRealism,
                            valueRange = 0f..1f,
                            enabled = !isGenerating,
                            darkTheme = darkTheme,
                            testTag = "avatar_slider_cloth_draping",
                            onValueChange = { onParamsChange(params.copy(clothDrapingRealism = it)) }
                        )
                        AvatarLabeledSlider(
                            title = if (isTamil) "ஆடை பொருத்தம் (Tailored Fit)" else "Garment Tailored Fit",
                            valueBadge = AvatarCustomizationCatalog.formatPercent(params.garmentTailoredFit),
                            value = params.garmentTailoredFit,
                            valueRange = 0f..1f,
                            enabled = !isGenerating,
                            darkTheme = darkTheme,
                            testTag = "avatar_slider_garment_fit",
                            onValueChange = { onParamsChange(params.copy(garmentTailoredFit = it)) }
                        )
                    }
                }
            }

            // 4. Shared Precision Mask, Identity Lock & Realism Sliders (Sliders 13–16)
            FrostedGlassCard(darkTheme = darkTheme) {
                Text(
                    text = if (isTamil) {
                        "3. துல்லிய Mask & Identity கட்டுப்பாடுகள் (Precision Controls)"
                    } else {
                        "3. Precision Mask, Identity & Realism Controls"
                    },
                    color = textColor,
                    style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.ExtraBold)
                )
                Spacer(modifier = Modifier.height(8.dp))

                AvatarLabeledSlider(
                    title = if (isTamil) "AI மாற்றத்தின் வலிமை (Transformation Strength)" else "AI Transformation Strength",
                    valueBadge = AvatarCustomizationCatalog.formatPercent(params.transformationStrength),
                    value = params.transformationStrength,
                    valueRange = 0.25f..0.82f,
                    enabled = !isGenerating,
                    darkTheme = darkTheme,
                    testTag = "avatar_slider_transform_strength",
                    onValueChange = { onParamsChange(params.copy(transformationStrength = it)) }
                )
                AvatarLabeledSlider(
                    title = if (isTamil) "அசல் முக அடையாளப் பூட்டு (Identity Lock)" else "Face & Identity Preservation Lock",
                    valueBadge = AvatarCustomizationCatalog.formatPercent(params.faceIdentityLock),
                    value = params.faceIdentityLock,
                    valueRange = 0.50f..1.00f,
                    enabled = !isGenerating,
                    darkTheme = darkTheme,
                    testTag = "avatar_slider_identity_lock",
                    onValueChange = { onParamsChange(params.copy(faceIdentityLock = it)) }
                )
                AvatarLabeledSlider(
                    title = if (isTamil) "விளிம்பு மென்மை (Mask Edge Feather)" else "Seamless Mask Edge Feathering",
                    valueBadge = "${params.maskFeatherPx.roundToInt()} px",
                    value = params.maskFeatherPx,
                    valueRange = 4f..32f,
                    enabled = !isGenerating,
                    darkTheme = darkTheme,
                    testTag = "avatar_slider_mask_feather",
                    onValueChange = { onParamsChange(params.copy(maskFeatherPx = it)) }
                )
                AvatarLabeledSlider(
                    title = if (isTamil) "Mask விரிவாக்கம் / சுருக்கம் (Mask Expand)" else "Mask Boundary Expand / Shrink",
                    valueBadge = "${params.maskExpandPx.roundToInt()} px",
                    value = params.maskExpandPx,
                    valueRange = -8f..24f,
                    enabled = !isGenerating,
                    darkTheme = darkTheme,
                    testTag = "avatar_slider_mask_expand",
                    onValueChange = { onParamsChange(params.copy(maskExpandPx = it)) }
                )
            }

            // 5. Optional Custom Realistic Camera Detail Note + Seed Lock
            FrostedGlassCard(darkTheme = darkTheme) {
                Text(
                    text = if (isTamil) {
                        "4. கூடுதல் இயற்கை விவரம் & Seed (விருப்பத்தேர்வு)"
                    } else {
                        "4. Extra Realistic Photo Detail & Seed (Optional)"
                    },
                    color = textColor,
                    style = MaterialTheme.typography.titleSmall.copy(fontWeight = FontWeight.ExtraBold)
                )
                Spacer(modifier = Modifier.height(6.dp))

                OutlinedTextField(
                    value = params.customRealisticNote,
                    onValueChange = { onParamsChange(params.copy(customRealisticNote = it)) },
                    enabled = !isGenerating,
                    placeholder = {
                        Text(
                            text = if (isTamil) {
                                "உதா: maroon border, temple daylight, silver wristwatch (Realistic Photo மட்டும்)"
                            } else {
                                "e.g., maroon silk border, soft outdoor sunlight, wristwatch (Realistic DSLR only)"
                            }
                        )
                    },
                    singleLine = false,
                    maxLines = 2,
                    modifier = Modifier
                        .fillMaxWidth()
                        .testTag("avatar_custom_note_input")
                )

                if (strippedForbidden) {
                    Spacer(modifier = Modifier.height(4.dp))
                    Text(
                        text = if (isTamil) {
                            "⚠️ Painting / 3D / Cartoon / Anime வார்த்தைகள் தானாக நீக்கப்பட்டு 100% Realistic புகைப்படமாக மட்டுமே உருவாக்கப்படும்."
                        } else {
                            "⚠️ Non-realistic terms (Painting / 3D / Cartoon / Anime) are automatically filtered — outputs are 100% realistic photographs only."
                        },
                        color = SoftAmberStatus,
                        style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.Bold)
                    )
                }

                Spacer(modifier = Modifier.height(8.dp))

                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text(
                        text = if (isTamil) "Seed பூட்டு (${params.seed})" else "Lock Seed (${params.seed})",
                        color = textColor,
                        style = MaterialTheme.typography.bodySmall.copy(fontWeight = FontWeight.Bold),
                        modifier = Modifier.weight(1f)
                    )
                    IconButton(
                        onClick = {
                            onParamsChange(params.copy(seed = Random.nextLong(100_000L, 999_999_999L)))
                        },
                        enabled = !isGenerating,
                        modifier = Modifier
                            .minimumInteractiveComponentSize()
                            .testTag("avatar_randomize_seed_button")
                    ) {
                        Icon(
                            imageVector = Icons.Filled.Shuffle,
                            contentDescription = "Randomize Seed",
                            tint = VibrantPinkPill
                        )
                    }
                    Switch(
                        checked = params.lockSeed,
                        onCheckedChange = { onParamsChange(params.copy(lockSeed = it)) },
                        enabled = !isGenerating,
                        modifier = Modifier.testTag("avatar_seed_lock_switch")
                    )
                }
            }

            // Missing Core Models Shortcut Card (if needed and Slot 7 isn't bypassing Age)
            val canBypassWithSlot7 = params.activeTab == AvatarCustomTab.AGE && metrics.isAgingSlotInstalled
            val modelsReady = metrics.isImg2ImgReadyFor(preferences.activeEngine) || canBypassWithSlot7
            if (!modelsReady) {
                FrostedGlassCard(darkTheme = darkTheme) {
                    Text(
                        text = metrics.missingSummaryFor(
                            engine = preferences.activeEngine,
                            requireImg2Img = true,
                            isTamil = isTamil
                        ),
                        color = SoftAmberStatus,
                        style = MaterialTheme.typography.bodyMedium.copy(fontWeight = FontWeight.Bold)
                    )
                    Spacer(modifier = Modifier.height(6.dp))
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.Center,
                        modifier = Modifier
                            .fillMaxWidth()
                            .minimumInteractiveComponentSize()
                            .clip(RoundedCornerShape(16.dp))
                            .background(Brush.horizontalGradient(listOf(PurpleActionStart, PurpleActionEnd)))
                            .clickable(onClick = onOpenModelsTab)
                            .padding(vertical = 10.dp)
                            .testTag("avatar_open_models_button")
                    ) {
                        Icon(
                            imageVector = Icons.Filled.FolderSpecial,
                            contentDescription = null,
                            tint = Color.White,
                            modifier = Modifier.size(18.dp)
                        )
                        Spacer(modifier = Modifier.width(6.dp))
                        Text(
                            text = if (isTamil) "Models பக்கத்திற்குச் செல் (1A–6 / Slot 7)" else "Open Models Tab (Slots 1A–6 & Slot 7)",
                            color = Color.White,
                            style = MaterialTheme.typography.labelLarge.copy(fontWeight = FontWeight.Bold)
                        )
                    }
                }
            }

            // Error Message Banner
            if (!errorMessage.isNullOrBlank()) {
                FrostedGlassCard(darkTheme = darkTheme) {
                    Text(
                        text = errorMessage,
                        color = Color(0xFFFF5252),
                        style = MaterialTheme.typography.bodyMedium.copy(fontWeight = FontWeight.Bold)
                    )
                }
            }

            // Live Progress Card during Offline Generation
            if (isGenerating && progress != null) {
                FrostedGlassCard(darkTheme = darkTheme) {
                    Text(
                        text = "${progress.stageTitle} (${liveElapsedSec}s)",
                        color = textColor,
                        style = MaterialTheme.typography.bodyMedium.copy(fontWeight = FontWeight.Bold)
                    )
                    Spacer(modifier = Modifier.height(6.dp))
                    LinearProgressIndicator(
                        progress = { (progress.percent / 100f).coerceIn(0.02f, 1f) },
                        color = VibrantPinkPill,
                        trackColor = Color.White.copy(alpha = 0.25f),
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(8.dp)
                            .clip(RoundedCornerShape(50))
                    )
                    Spacer(modifier = Modifier.height(8.dp))
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.Center,
                        modifier = Modifier
                            .fillMaxWidth()
                            .minimumInteractiveComponentSize()
                            .clip(RoundedCornerShape(14.dp))
                            .background(Color(0xFFD32F2F))
                            .clickable(onClick = onCancelGeneration)
                            .padding(vertical = 9.dp)
                            .testTag("avatar_cancel_button")
                    ) {
                        Icon(
                            imageVector = Icons.Filled.Cancel,
                            contentDescription = "Cancel",
                            tint = Color.White,
                            modifier = Modifier.size(18.dp)
                        )
                        Spacer(modifier = Modifier.width(6.dp))
                        Text(
                            text = if (isTamil) "நிறுத்து & RAM விடுவி" else "Cancel & Free RAM",
                            color = Color.White,
                            style = MaterialTheme.typography.labelLarge.copy(fontWeight = FontWeight.Bold)
                        )
                    }
                }
            }

            // Instant Bottom Generated Result Preview Card (visible right next to Generate button too!)
            if (!isGenerating && latestAvatarCreation != null && File(latestAvatarCreation.filePath).exists()) {
                FrostedGlassCard(
                    darkTheme = darkTheme,
                    modifier = Modifier.testTag("avatar_bottom_result_card")
                ) {
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(12.dp),
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        AsyncImage(
                            model = File(latestAvatarCreation.filePath),
                            contentDescription = "Latest Generated Realistic Avatar Photo",
                            contentScale = ContentScale.Crop,
                            modifier = Modifier
                                .size(78.dp)
                                .clip(RoundedCornerShape(14.dp))
                                .border(1.5.dp, NeonReadyGreen, RoundedCornerShape(14.dp))
                        )
                        Column(
                            verticalArrangement = Arrangement.spacedBy(6.dp),
                            modifier = Modifier.weight(1f)
                        ) {
                            Text(
                                text = if (isTamil) {
                                    "✅ புதிய Realistic புகைப்படம் தயார்!"
                                } else {
                                    "✅ Realistic DSLR Photo Ready!"
                                },
                                color = NeonReadyGreen,
                                style = MaterialTheme.typography.titleSmall.copy(fontWeight = FontWeight.ExtraBold)
                            )
                            Text(
                                text = "${latestAvatarCreation.resolution} • ${(latestAvatarCreation.durationMs / 1000L).coerceAtLeast(1L)}s",
                                color = subColor,
                                style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.SemiBold)
                            )
                            Row(
                                horizontalArrangement = Arrangement.spacedBy(8.dp),
                                modifier = Modifier.fillMaxWidth()
                            ) {
                                Row(
                                    verticalAlignment = Alignment.CenterVertically,
                                    horizontalArrangement = Arrangement.Center,
                                    modifier = Modifier
                                        .weight(1f)
                                        .minimumInteractiveComponentSize()
                                        .clip(RoundedCornerShape(12.dp))
                                        .background(Brush.horizontalGradient(listOf(PurpleActionStart, PurpleActionEnd)))
                                        .clickable {
                                            coroutineScope.launch { mainScrollState.animateScrollTo(0) }
                                        }
                                        .padding(vertical = 8.dp, horizontal = 8.dp)
                                        .testTag("avatar_bottom_scroll_compare_button")
                                ) {
                                    Icon(
                                        imageVector = Icons.Filled.Compare,
                                        contentDescription = null,
                                        tint = Color.White,
                                        modifier = Modifier.size(16.dp)
                                    )
                                    Spacer(modifier = Modifier.width(4.dp))
                                    Text(
                                        text = if (isTamil) "Before/After ↑" else "Before/After ↑",
                                        color = Color.White,
                                        style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.Bold),
                                        maxLines = 1
                                    )
                                }
                                Row(
                                    verticalAlignment = Alignment.CenterVertically,
                                    horizontalArrangement = Arrangement.Center,
                                    modifier = Modifier
                                        .minimumInteractiveComponentSize()
                                        .clip(RoundedCornerShape(12.dp))
                                        .background(Brush.horizontalGradient(listOf(PinkGlowStart, VibrantPinkPill)))
                                        .clickable { onExportToGallery(latestAvatarCreation) }
                                        .padding(vertical = 8.dp, horizontal = 10.dp)
                                        .testTag("avatar_bottom_save_gallery_button")
                                ) {
                                    Icon(
                                        imageVector = Icons.Filled.Download,
                                        contentDescription = "Save to Gallery",
                                        tint = Color.White,
                                        modifier = Modifier.size(16.dp)
                                    )
                                    Spacer(modifier = Modifier.width(4.dp))
                                    Text(
                                        text = if (isTamil) "Save" else "Save",
                                        color = Color.White,
                                        style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.Bold)
                                    )
                                }
                            }
                        }
                    }
                }
            }

            // Primary Generate Realistic Photograph Action Button
            val genBtnShape = RoundedCornerShape(24.dp)
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.Center,
                modifier = Modifier
                    .fillMaxWidth()
                    .minimumInteractiveComponentSize()
                    .shadow(10.dp, genBtnShape, spotColor = VibrantPinkPill)
                    .clip(genBtnShape)
                    .background(
                        Brush.horizontalGradient(
                            listOf(PinkGlowStart, VibrantPinkPill, PurpleActionEnd)
                        )
                    )
                    .border(1.5.dp, Color.White.copy(alpha = 0.9f), genBtnShape)
                    .clickable(enabled = !isGenerating, onClick = onGenerateAvatar)
                    .padding(vertical = 15.dp, horizontal = 18.dp)
                    .testTag("avatar_generate_button")
            ) {
                Icon(
                    imageVector = Icons.Filled.AutoAwesome,
                    contentDescription = null,
                    tint = Color.White,
                    modifier = Modifier.size(22.dp)
                )
                Spacer(modifier = Modifier.width(8.dp))
                Text(
                    text = if (isTamil) {
                        "உண்மையான புகைப்படத்தை உருவாக்கு (${params.activeTab.shortTitleTa})"
                    } else {
                        "Generate Realistic Photo (${params.activeTab.shortTitleEn})"
                    },
                    color = Color.White,
                    style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.ExtraBold)
                )
            }

            Spacer(modifier = Modifier.height(16.dp))
        }
    }
}

private fun selectTabWithDefaultPreset(
    tab: AvatarCustomTab,
    current: AvatarCustomizationParams,
    onParamsChange: (AvatarCustomizationParams) -> Unit
) {
    if (current.activeTab == tab) return
    val defaultPreset = AvatarCustomizationCatalog.presetsForTab(tab).first()
    onParamsChange(applyPresetDefaults(current.copy(activeTab = tab), defaultPreset))
}

private fun applyPresetDefaults(
    base: AvatarCustomizationParams,
    preset: RealisticAvatarPreset
): AvatarCustomizationParams {
    return base.copy(
        activeTab = preset.tab,
        selectedPresetId = preset.id,
        targetAge = preset.defaultTargetAge ?: base.targetAge,
        skinWrinkleIntensity = preset.defaultWrinkleIntensity ?: base.skinWrinkleIntensity,
        silverHairRatio = preset.defaultSilverHair ?: base.silverHairRatio,
        absCoreDefinition = preset.defaultAbsDefinition ?: base.absCoreDefinition,
        muscleShoulderMass = preset.defaultMuscleMass ?: base.muscleShoulderMass,
        waistBodyTrim = preset.defaultWaistTrim ?: base.waistBodyTrim,
        hairVolumeDensity = preset.defaultHairVolume ?: base.hairVolumeDensity,
        hairLengthScale = preset.defaultHairLength ?: base.hairLengthScale,
        clothDrapingRealism = preset.defaultClothDraping ?: base.clothDrapingRealism,
        transformationStrength = preset.defaultStrength
    )
}

@Composable
private fun AvatarModuleTabPill(
    tab: AvatarCustomTab,
    icon: ImageVector,
    isSelected: Boolean,
    isTamil: Boolean,
    darkTheme: Boolean,
    enabled: Boolean,
    testTag: String,
    onClick: () -> Unit,
    modifier: Modifier = Modifier
) {
    val shape = RoundedCornerShape(16.dp)
    val textColor = if (darkTheme) Color.White else DeepPlumText
    Column(
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center,
        modifier = modifier
            .minimumInteractiveComponentSize()
            .clip(shape)
            .background(
                if (isSelected) {
                    Brush.verticalGradient(listOf(PinkGlowStart, PurpleActionEnd))
                } else {
                    Brush.verticalGradient(
                        listOf(
                            Color.White.copy(alpha = if (darkTheme) 0.14f else 0.72f),
                            Color.White.copy(alpha = if (darkTheme) 0.08f else 0.56f)
                        )
                    )
                }
            )
            .border(
                width = if (isSelected) 1.6.dp else 1.dp,
                color = if (isSelected) Color.White else Color.White.copy(alpha = 0.65f),
                shape = shape
            )
            .clickable(enabled = enabled, onClick = onClick)
            .padding(vertical = 8.dp, horizontal = 4.dp)
            .testTag(testTag)
    ) {
        Icon(
            imageVector = icon,
            contentDescription = if (isTamil) tab.shortTitleTa else tab.shortTitleEn,
            tint = if (isSelected) Color.White else textColor,
            modifier = Modifier.size(20.dp)
        )
        Spacer(modifier = Modifier.height(3.dp))
        Text(
            text = if (isTamil) tab.shortTitleTa else tab.shortTitleEn,
            color = if (isSelected) Color.White else textColor,
            style = MaterialTheme.typography.labelSmall.copy(
                fontWeight = if (isSelected) FontWeight.ExtraBold else FontWeight.SemiBold,
                fontSize = 11.sp
            ),
            maxLines = 1,
            overflow = TextOverflow.Ellipsis
        )
    }
}

@Composable
private fun IdentityProtectionNotice(
    text: String,
    darkTheme: Boolean
) {
    Row(
        verticalAlignment = Alignment.CenterVertically,
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(12.dp))
            .background(Color.White.copy(alpha = if (darkTheme) 0.10f else 0.66f))
            .border(1.dp, NeonReadyGreen.copy(alpha = 0.7f), RoundedCornerShape(12.dp))
            .padding(horizontal = 10.dp, vertical = 6.dp)
    ) {
        Icon(
            imageVector = Icons.Filled.Lock,
            contentDescription = null,
            tint = NeonReadyGreen,
            modifier = Modifier.size(15.dp)
        )
        Spacer(modifier = Modifier.width(6.dp))
        Text(
            text = text,
            color = if (darkTheme) Color.White else DeepPlumText,
            style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.Bold)
        )
    }
}

@Composable
private fun SelectableOptionChip(
    label: String,
    selected: Boolean,
    darkTheme: Boolean,
    enabled: Boolean,
    testTag: String,
    onClick: () -> Unit
) {
    val shape = RoundedCornerShape(50)
    val textColor = if (darkTheme) Color.White else DeepPlumText
    Box(
        contentAlignment = Alignment.Center,
        modifier = Modifier
            .minimumInteractiveComponentSize()
            .clip(shape)
            .background(
                if (selected) {
                    Brush.horizontalGradient(listOf(PurpleActionStart, VibrantPinkPill))
                } else {
                    Brush.horizontalGradient(
                        listOf(
                            Color.White.copy(alpha = if (darkTheme) 0.14f else 0.72f),
                            Color.White.copy(alpha = if (darkTheme) 0.10f else 0.60f)
                        )
                    )
                }
            )
            .border(
                width = 1.dp,
                color = if (selected) Color.White else Color.White.copy(alpha = 0.65f),
                shape = shape
            )
            .clickable(enabled = enabled, onClick = onClick)
            .padding(horizontal = 12.dp, vertical = 7.dp)
            .testTag(testTag)
    ) {
        Text(
            text = label,
            color = if (selected) Color.White else textColor,
            style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.Bold)
        )
    }
}
