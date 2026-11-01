package com.example.ui.screens

import android.net.Uri
import androidx.activity.compose.BackHandler
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
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
import androidx.compose.material.icons.filled.AddPhotoAlternate
import androidx.compose.material.icons.filled.AutoAwesome
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.CloudDownload
import androidx.compose.material.icons.filled.DeleteOutline
import androidx.compose.material.icons.filled.FavoriteBorder
import androidx.compose.material.icons.filled.FileOpen
import androidx.compose.material.icons.filled.FolderSpecial
import androidx.compose.material.icons.filled.PhotoLibrary
import androidx.compose.material.icons.filled.StarBorder
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.material3.minimumInteractiveComponentSize
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import coil.compose.AsyncImage
import com.example.ai.DiffusionModelSlot
import com.example.ai.OnnxModelManager
import com.example.ai.RealDeviceAndModelMetrics
import com.example.data.AppPreferencesState
import com.example.ui.ActiveScreen
import com.example.ui.components.CircularGlowIconButton
import com.example.ui.components.FrostedGlassCard
import com.example.ui.theme.DeepPlumText
import com.example.ui.theme.NeonReadyGreen
import com.example.ui.theme.PinkGlowEnd
import com.example.ui.theme.PinkGlowStart
import com.example.ui.theme.PurpleActionEnd
import com.example.ui.theme.PurpleActionStart
import com.example.ui.theme.SoftAmberStatus
import com.example.ui.theme.VibrantPinkPill

@Composable
fun ModelsTabScreen(
    preferences: AppPreferencesState,
    metrics: RealDeviceAndModelMetrics,
    isImporting: Boolean,
    importProgressText: String?,
    onOneTouchFolderUpload: () -> Unit,
    onSingleSlotUpload: (DiffusionModelSlot) -> Unit,
    onDeleteSlot: (DiffusionModelSlot) -> Unit,
    modifier: Modifier = Modifier
) {
    val isTamil = preferences.isTamil
    val darkTheme = preferences.darkTheme
    val textColor = if (darkTheme) Color.White else DeepPlumText
    val subColor = if (darkTheme) Color(0xFFE1C9F7) else Color(0xFF4F376E)

    Column(
        horizontalAlignment = Alignment.CenterHorizontally,
        modifier = modifier
            .fillMaxSize()
            .windowInsetsPadding(WindowInsets.statusBars)
            .verticalScroll(rememberScrollState())
            .padding(horizontal = 16.dp)
    ) {
        Column(
            verticalArrangement = Arrangement.spacedBy(12.dp),
            modifier = Modifier
                .fillMaxWidth()
                .widthIn(max = 600.dp)
                .padding(vertical = 10.dp)
        ) {
            Text(
                text = if (isTamil) "Offline ONNX மாடல்கள்" else "Offline ONNX Models",
                color = textColor,
                style = MaterialTheme.typography.headlineMedium.copy(fontWeight = FontWeight.ExtraBold)
            )

            // One-Touch Folder Upload Hero Glass Card (matching face-swap convenience)
            FrostedGlassCard(darkTheme = darkTheme) {
                Text(
                    text = if (isTamil) {
                        "One-Touch Folder Upload (ஒரே கிளிக்கில் ஏற்றவும்)"
                    } else {
                        "One-Touch Folder Upload"
                    },
                    color = textColor,
                    style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold)
                )
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    text = if (isTamil) {
                        "மொபைலில் ஒரே Folder-ல் சேமித்த 4 .onnx மாடல்களையும் (text_encoder, unet, vae_decoder, vae_encoder) ஒரே கிளிக்கில் தானாக அடையாளம் கண்டு ஏற்றும்."
                    } else {
                        "Select the folder containing your downloaded .onnx models (text_encoder, unet, vae_decoder, vae_encoder) to import all slots automatically."
                    },
                    color = subColor,
                    style = MaterialTheme.typography.bodyMedium
                )
                Spacer(modifier = Modifier.height(12.dp))

                val btnShape = RoundedCornerShape(22.dp)
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.Center,
                    modifier = Modifier
                        .fillMaxWidth()
                        .minimumInteractiveComponentSize()
                        .shadow(8.dp, btnShape, spotColor = PurpleActionEnd)
                        .clip(btnShape)
                        .background(Brush.horizontalGradient(listOf(PinkGlowStart, PurpleActionEnd)))
                        .border(1.3.dp, Color.White.copy(alpha = 0.85f), btnShape)
                        .clickable(enabled = !isImporting, onClick = onOneTouchFolderUpload)
                        .padding(vertical = 13.dp, horizontal = 16.dp)
                        .testTag("one_touch_folder_upload_button")
                ) {
                    if (isImporting) {
                        CircularProgressIndicator(
                            color = Color.White,
                            strokeWidth = 2.5.dp,
                            modifier = Modifier.size(20.dp)
                        )
                    } else {
                        Icon(
                            imageVector = Icons.Filled.FolderSpecial,
                            contentDescription = null,
                            tint = Color.White,
                            modifier = Modifier.size(22.dp)
                        )
                    }
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(
                        text = if (isImporting) {
                            importProgressText ?: "Importing..."
                        } else if (isTamil) {
                            "Folder தேர்வு செய்து ஏற்றுக"
                        } else {
                            "Select Folder & Auto-Import"
                        },
                        color = Color.White,
                        style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold)
                    )
                }

                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    text = "Installed: ${metrics.installedCount}/4 • Disk: ${OnnxModelManager.formatBytes(metrics.totalModelBytes)} • Free Space: ${OnnxModelManager.formatBytes(metrics.deviceFreeBytes)}",
                    color = subColor,
                    style = MaterialTheme.typography.labelLarge
                )
            }

            // 4 Individual Model Slots
            metrics.slots.forEach { slotStatus ->
                val slot = slotStatus.slot
                FrostedGlassCard(darkTheme = darkTheme) {
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Icon(
                            imageVector = if (slotStatus.isInstalled) Icons.Filled.CheckCircle else Icons.Filled.CloudDownload,
                            contentDescription = null,
                            tint = if (slotStatus.isInstalled) NeonReadyGreen else SoftAmberStatus,
                            modifier = Modifier.size(24.dp)
                        )
                        Spacer(modifier = Modifier.width(10.dp))
                        Column(modifier = Modifier.weight(1f)) {
                            Text(
                                text = if (isTamil) slot.titleTa else slot.titleEn,
                                color = textColor,
                                style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold)
                            )
                            Text(
                                text = "${slot.canonicalFileName} • ${
                                    if (slotStatus.isInstalled) {
                                        OnnxModelManager.formatBytes(slotStatus.totalBytes)
                                    } else if (isTamil) {
                                        "இன்னும் சேர்க்கப்படவில்லை"
                                    } else {
                                        "Not uploaded yet"
                                    }
                                }",
                                color = subColor,
                                style = MaterialTheme.typography.bodyMedium
                            )
                            Text(
                                text = if (isTamil) slot.descriptionTa else slot.descriptionEn,
                                color = subColor.copy(alpha = 0.85f),
                                style = MaterialTheme.typography.labelSmall
                            )
                        }

                        IconButton(
                            onClick = { onSingleSlotUpload(slot) },
                            modifier = Modifier.testTag("upload_slot_${slot.folderHint}")
                        ) {
                            Icon(
                                imageVector = Icons.Filled.FileOpen,
                                contentDescription = "Select ${slot.canonicalFileName}",
                                tint = PurpleActionEnd
                            )
                        }

                        if (slotStatus.isInstalled) {
                            IconButton(
                                onClick = { onDeleteSlot(slot) },
                                modifier = Modifier.testTag("delete_slot_${slot.folderHint}")
                            ) {
                                Icon(
                                    imageVector = Icons.Filled.DeleteOutline,
                                    contentDescription = "Delete ${slot.canonicalFileName}",
                                    tint = Color(0xFFE53935)
                                )
                            }
                        }
                    }
                }
            }

            Spacer(modifier = Modifier.height(16.dp))
        }
    }
}

@Composable
fun EmptyCollectionTabScreen(
    title: String,
    subtitle: String,
    isFavorites: Boolean,
    darkTheme: Boolean,
    onStartCreating: () -> Unit,
    modifier: Modifier = Modifier
) {
    val textColor = if (darkTheme) Color.White else DeepPlumText
    val subColor = if (darkTheme) Color(0xFFE1C9F7) else Color(0xFF4F376E)

    Box(
        contentAlignment = Alignment.Center,
        modifier = modifier
            .fillMaxSize()
            .windowInsetsPadding(WindowInsets.statusBars)
            .padding(20.dp)
    ) {
        FrostedGlassCard(
            darkTheme = darkTheme,
            modifier = Modifier
                .fillMaxWidth()
                .widthIn(max = 500.dp)
        ) {
            Column(
                horizontalAlignment = Alignment.CenterHorizontally,
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(vertical = 18.dp, horizontal = 8.dp)
            ) {
                Icon(
                    imageVector = if (isFavorites) Icons.Filled.StarBorder else Icons.Filled.FavoriteBorder,
                    contentDescription = null,
                    tint = VibrantPinkPill,
                    modifier = Modifier.size(52.dp)
                )
                Spacer(modifier = Modifier.height(10.dp))
                Text(
                    text = title,
                    color = textColor,
                    style = MaterialTheme.typography.titleLarge.copy(fontWeight = FontWeight.Bold),
                    textAlign = TextAlign.Center
                )
                Spacer(modifier = Modifier.height(6.dp))
                Text(
                    text = subtitle,
                    color = subColor,
                    style = MaterialTheme.typography.bodyMedium,
                    textAlign = TextAlign.Center
                )
                Spacer(modifier = Modifier.height(16.dp))
                Box(
                    contentAlignment = Alignment.Center,
                    modifier = Modifier
                        .minimumInteractiveComponentSize()
                        .clip(RoundedCornerShape(50))
                        .background(Brush.horizontalGradient(listOf(PinkGlowStart, PinkGlowEnd)))
                        .border(1.dp, Color.White, RoundedCornerShape(50))
                        .clickable(onClick = onStartCreating)
                        .padding(horizontal = 22.dp, vertical = 10.dp)
                        .testTag("empty_collection_create_button")
                ) {
                    Text(
                        text = "Create Image",
                        color = Color.White,
                        style = MaterialTheme.typography.labelLarge.copy(fontWeight = FontWeight.Bold)
                    )
                }
            }
        }
    }
}

@Composable
fun StudioWorkspaceScreen(
    screen: ActiveScreen,
    preferences: AppPreferencesState,
    metrics: RealDeviceAndModelMetrics,
    promptDraft: String,
    selectedPhotoUri: Uri?,
    onPromptChange: (String) -> Unit,
    onPickPhoto: () -> Unit,
    onOpenModelsTab: () -> Unit,
    onBack: () -> Unit,
    modifier: Modifier = Modifier
) {
    BackHandler { onBack() }

    val isTamil = preferences.isTamil
    val darkTheme = preferences.darkTheme
    val textColor = if (darkTheme) Color.White else DeepPlumText
    val subColor = if (darkTheme) Color(0xFFE1C9F7) else Color(0xFF4F376E)

    val title = when (screen) {
        ActiveScreen.TEXT_TO_IMAGE -> "Text to Image"
        ActiveScreen.IMAGE_TO_IMAGE -> "Image to Image"
        ActiveScreen.CHARACTER_STUDIO -> "Character Studio"
        ActiveScreen.FACE_SWAP -> "Face Swap"
        ActiveScreen.IMAGE_EDITOR -> "Image Editor"
        ActiveScreen.MY_GALLERY -> "My Gallery"
        else -> "Studio"
    }

    val needsInputPhoto = screen in setOf(
        ActiveScreen.IMAGE_TO_IMAGE,
        ActiveScreen.CHARACTER_STUDIO,
        ActiveScreen.FACE_SWAP,
        ActiveScreen.IMAGE_EDITOR
    )

    Column(
        horizontalAlignment = Alignment.CenterHorizontally,
        modifier = modifier
            .fillMaxSize()
            .windowInsetsPadding(WindowInsets.statusBars)
            .verticalScroll(rememberScrollState())
            .padding(horizontal = 16.dp)
    ) {
        Column(
            verticalArrangement = Arrangement.spacedBy(12.dp),
            modifier = Modifier
                .fillMaxWidth()
                .widthIn(max = 600.dp)
                .padding(vertical = 10.dp)
        ) {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                modifier = Modifier.fillMaxWidth()
            ) {
                CircularGlowIconButton(
                    onClick = onBack,
                    contentDescription = "Back to Home",
                    testTag = "workspace_back_button"
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
                        text = title,
                        color = textColor,
                        style = MaterialTheme.typography.headlineMedium.copy(fontWeight = FontWeight.Bold)
                    )
                    Text(
                        text = "${preferences.resolution} • ${preferences.inferenceMode}",
                        color = subColor,
                        style = MaterialTheme.typography.bodyMedium
                    )
                }
            }

            if (screen == ActiveScreen.MY_GALLERY) {
                FrostedGlassCard(darkTheme = darkTheme) {
                    Column(
                        horizontalAlignment = Alignment.CenterHorizontally,
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(vertical = 22.dp)
                    ) {
                        Icon(
                            imageVector = Icons.Filled.PhotoLibrary,
                            contentDescription = null,
                            tint = PurpleActionEnd,
                            modifier = Modifier.size(48.dp)
                        )
                        Spacer(modifier = Modifier.height(8.dp))
                        Text(
                            text = if (isTamil) {
                                "நீங்கள் உருவாக்கும் படங்கள் இங்கே சேமிக்கப்படும்"
                            } else {
                                "Your offline generated photos will appear here"
                            },
                            color = textColor,
                            style = MaterialTheme.typography.titleMedium,
                            textAlign = TextAlign.Center
                        )
                    }
                }
            } else {
                if (needsInputPhoto) {
                    FrostedGlassCard(darkTheme = darkTheme) {
                        Text(
                            text = if (isTamil) "Reference புகைப்படம்" else "Reference Photo",
                            color = textColor,
                            style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold)
                        )
                        Spacer(modifier = Modifier.height(8.dp))
                        if (selectedPhotoUri != null) {
                            AsyncImage(
                                model = selectedPhotoUri,
                                contentDescription = "Selected photo",
                                contentScale = ContentScale.Crop,
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .height(190.dp)
                                    .clip(RoundedCornerShape(16.dp))
                            )
                            Spacer(modifier = Modifier.height(8.dp))
                        }
                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.Center,
                            modifier = Modifier
                                .fillMaxWidth()
                                .minimumInteractiveComponentSize()
                                .clip(RoundedCornerShape(16.dp))
                                .background(Brush.horizontalGradient(listOf(PurpleActionStart, PurpleActionEnd)))
                                .clickable(onClick = onPickPhoto)
                                .padding(vertical = 11.dp)
                                .testTag("pick_reference_photo_button")
                        ) {
                            Icon(
                                imageVector = Icons.Filled.AddPhotoAlternate,
                                contentDescription = null,
                                tint = Color.White
                            )
                            Spacer(modifier = Modifier.width(8.dp))
                            Text(
                                text = if (isTamil) "படம் தேர்வு செய்க" else "Choose Photo from Gallery",
                                color = Color.White,
                                style = MaterialTheme.typography.labelLarge.copy(fontWeight = FontWeight.Bold)
                            )
                        }
                    }
                }

                FrostedGlassCard(darkTheme = darkTheme) {
                    Text(
                        text = if (isTamil) "Prompt (தமிழ் / English)" else "Creative Prompt (Tamil / English)",
                        color = textColor,
                        style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold)
                    )
                    Spacer(modifier = Modifier.height(8.dp))
                    OutlinedTextField(
                        value = promptDraft,
                        onValueChange = onPromptChange,
                        placeholder = {
                            Text(
                                if (isTamil) {
                                    "உதாரணம்: அழகான பட்டுப்புடவை அணிந்த பெண், இயற்கை ஒளி..."
                                } else {
                                    "e.g., Cinematic portrait in silk saree, soft studio bokeh lighting..."
                                }
                            )
                        },
                        minLines = 3,
                        modifier = Modifier
                            .fillMaxWidth()
                            .testTag("prompt_input_field")
                    )
                    Spacer(modifier = Modifier.height(12.dp))

                    val modelsReady = metrics.isTxt2ImgReady
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.Center,
                        modifier = Modifier
                            .fillMaxWidth()
                            .minimumInteractiveComponentSize()
                            .clip(RoundedCornerShape(20.dp))
                            .background(Brush.horizontalGradient(listOf(PinkGlowStart, PinkGlowEnd)))
                            .border(1.2.dp, Color.White, RoundedCornerShape(20.dp))
                            .clickable {
                                if (!modelsReady) {
                                    onOpenModelsTab()
                                }
                            }
                            .padding(vertical = 13.dp, horizontal = 14.dp)
                            .testTag("workspace_primary_action_button")
                    ) {
                        Icon(
                            imageVector = if (modelsReady) Icons.Filled.AutoAwesome else Icons.Filled.FolderSpecial,
                            contentDescription = null,
                            tint = Color.White
                        )
                        Spacer(modifier = Modifier.width(8.dp))
                        Text(
                            text = if (modelsReady) {
                                if (isTamil) "Offline ONNX தயார் (Phase 3/4 Pipeline)" else "ONNX Ready • Next Phase Inference"
                            } else {
                                if (isTamil) "முதலில் ONNX Folder ஏற்றவும் (${metrics.installedCount}/4)" else "Upload ONNX Folder First (${metrics.installedCount}/4)"
                            },
                            color = Color.White,
                            style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold)
                        )
                    }
                }
            }
        }
    }
}
