package com.example.ui.screens

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
import androidx.compose.foundation.layout.aspectRatio
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
import androidx.compose.material.icons.filled.Bolt
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.CloudDownload
import androidx.compose.material.icons.filled.DeleteOutline
import androidx.compose.material.icons.filled.Download
import androidx.compose.material.icons.filled.FavoriteBorder
import androidx.compose.material.icons.filled.FileOpen
import androidx.compose.material.icons.filled.FolderSpecial
import androidx.compose.material.icons.filled.Star
import androidx.compose.material.icons.filled.StarBorder
import androidx.compose.material.icons.filled.WarningAmber
import androidx.compose.material.icons.outlined.StarOutline
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
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
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import coil.compose.AsyncImage
import com.example.ai.DiffusionModelSlot
import com.example.ai.ModelSlotStatus
import com.example.ai.OnnxModelManager
import com.example.ai.RealDeviceAndModelMetrics
import com.example.data.AppPreferencesState
import com.example.data.GeneratedCreation
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
import java.io.File

@Composable
fun ModelsTabScreen(
    preferences: AppPreferencesState,
    metrics: RealDeviceAndModelMetrics,
    isImporting: Boolean,
    importProgressText: String?,
    verificationReport: String?,
    onOneTouchFolderUpload: () -> Unit,
    onVerifyPipeline: () -> Unit,
    onSingleSlotUpload: (DiffusionModelSlot) -> Unit,
    onAttachCompanionWeight: (DiffusionModelSlot) -> Unit = {},
    onOpenSettingsDownloads: () -> Unit = {},
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

            // One-Touch Folder Upload + ONNX Verification Card
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
                        "ஒரே Folder-ல் சேமித்த Image Generation 4 .onnx மாடல்களையும் + Face Swap 3 .onnx மாடல்களையும் (det_10g, w600k_r50, inswapper_128 / hyperswap) தானாக அடையாளம் கண்டு ஏற்றும்."
                    } else {
                        "Select your folder containing Diffusion .onnx models (text_encoder, unet, vae_decoder, vae_encoder) and/or Face Swap .onnx models (det_10g, w600k_r50, inswapper_128 / hyperswap) to import automatically."
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

                if (metrics.hasAnySlotFilesOnDisk) {
                    Spacer(modifier = Modifier.height(8.dp))
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.Center,
                        modifier = Modifier
                            .fillMaxWidth()
                            .minimumInteractiveComponentSize()
                            .clip(RoundedCornerShape(16.dp))
                            .background(Brush.horizontalGradient(listOf(PurpleActionStart, PurpleActionEnd)))
                            .border(1.dp, Color.White.copy(alpha = 0.7f), RoundedCornerShape(16.dp))
                            .clickable(enabled = !isImporting, onClick = onVerifyPipeline)
                            .padding(vertical = 10.dp)
                            .testTag("verify_onnx_pipeline_button")
                    ) {
                        Icon(
                            imageVector = Icons.Filled.Bolt,
                            contentDescription = null,
                            tint = Color.White,
                            modifier = Modifier.size(18.dp)
                        )
                        Spacer(modifier = Modifier.width(6.dp))
                        Text(
                            text = if (isTamil) "ONNX மாடல்களைச் சரிபார் (Verify)" else "Verify ONNX Runtime Sessions",
                            color = Color.White,
                            style = MaterialTheme.typography.labelLarge.copy(fontWeight = FontWeight.Bold)
                        )
                    }
                }

                Spacer(modifier = Modifier.height(8.dp))
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.Center,
                    modifier = Modifier
                        .fillMaxWidth()
                        .minimumInteractiveComponentSize()
                        .clip(RoundedCornerShape(16.dp))
                        .background(Color.White.copy(alpha = if (darkTheme) 0.12f else 0.65f))
                        .border(1.dp, VibrantPinkPill.copy(alpha = 0.75f), RoundedCornerShape(16.dp))
                        .clickable(enabled = !isImporting, onClick = onOpenSettingsDownloads)
                        .padding(vertical = 9.dp, horizontal = 12.dp)
                        .testTag("models_open_settings_downloads_button")
                ) {
                    Icon(
                        imageVector = Icons.Filled.CloudDownload,
                        contentDescription = null,
                        tint = VibrantPinkPill,
                        modifier = Modifier.size(18.dp)
                    )
                    Spacer(modifier = Modifier.width(6.dp))
                    Text(
                        text = if (isTamil) {
                            "ONNX மாடல் Download Links (Settings)"
                        } else {
                            "Get ONNX Model Download Links (Settings)"
                        },
                        color = textColor,
                        style = MaterialTheme.typography.labelLarge.copy(fontWeight = FontWeight.Bold)
                    )
                }

                if (!verificationReport.isNullOrBlank()) {
                    Spacer(modifier = Modifier.height(8.dp))
                    Box(
                        modifier = Modifier
                            .fillMaxWidth()
                            .clip(RoundedCornerShape(12.dp))
                            .background(Color.White.copy(alpha = if (darkTheme) 0.08f else 0.58f))
                            .padding(10.dp)
                    ) {
                        Text(
                            text = verificationReport,
                            color = textColor,
                            style = MaterialTheme.typography.labelSmall
                        )
                    }
                }

                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    text = "Core AI: ${metrics.installedCount}/4 • Face Swap: ${metrics.faceSwapInstalledCount}/3 • Disk: ${OnnxModelManager.formatBytes(metrics.totalModelBytes)} • Free: ${OnnxModelManager.formatBytes(metrics.deviceFreeBytes)}",
                    color = subColor,
                    style = MaterialTheme.typography.labelLarge
                )
            }

            // Section 1: Core Diffusion Models (4)
            Text(
                text = if (isTamil) {
                    "Image Generation மாடல்கள் (${metrics.installedCount}/4)"
                } else {
                    "Core Image Generation Models (${metrics.installedCount}/4)"
                },
                color = textColor,
                style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.ExtraBold)
            )

            metrics.diffusionSlots.forEach { slotStatus ->
                ModelSlotItemCard(
                    slotStatus = slotStatus,
                    isTamil = isTamil,
                    darkTheme = darkTheme,
                    textColor = textColor,
                    subColor = subColor,
                    onSingleSlotUpload = onSingleSlotUpload,
                    onAttachCompanionWeight = onAttachCompanionWeight,
                    onDeleteSlot = onDeleteSlot
                )
            }

            // Section 2: Face Swap Add-On Models (3)
            Spacer(modifier = Modifier.height(4.dp))
            Text(
                text = if (isTamil) {
                    "Face Swap மாடல்கள் (${metrics.faceSwapInstalledCount}/3 • Optional)"
                } else {
                    "Face Swap Add-On Models (${metrics.faceSwapInstalledCount}/3 • Optional)"
                },
                color = textColor,
                style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.ExtraBold)
            )

            metrics.faceSwapSlots.forEach { slotStatus ->
                ModelSlotItemCard(
                    slotStatus = slotStatus,
                    isTamil = isTamil,
                    darkTheme = darkTheme,
                    textColor = textColor,
                    subColor = subColor,
                    onSingleSlotUpload = onSingleSlotUpload,
                    onAttachCompanionWeight = onAttachCompanionWeight,
                    onDeleteSlot = onDeleteSlot
                )
            }

            Spacer(modifier = Modifier.height(16.dp))
        }
    }
}

@Composable
private fun ModelSlotItemCard(
    slotStatus: ModelSlotStatus,
    isTamil: Boolean,
    darkTheme: Boolean,
    textColor: Color,
    subColor: Color,
    onSingleSlotUpload: (DiffusionModelSlot) -> Unit,
    onAttachCompanionWeight: (DiffusionModelSlot) -> Unit,
    onDeleteSlot: (DiffusionModelSlot) -> Unit
) {
    val slot = slotStatus.slot
    val isIncomplete = slotStatus.isIncompleteExternalGraph
    val displayFileName = if ((slotStatus.isInstalled || isIncomplete) && slotStatus.sourceFileLabel.isNotBlank()) {
        slotStatus.sourceFileLabel
    } else {
        slot.canonicalFileName
    }

    FrostedGlassCard(darkTheme = darkTheme) {
        Column(modifier = Modifier.fillMaxWidth()) {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                modifier = Modifier.fillMaxWidth()
            ) {
                Icon(
                    imageVector = when {
                        slotStatus.isInstalled -> Icons.Filled.CheckCircle
                        isIncomplete -> Icons.Filled.WarningAmber
                        else -> Icons.Filled.CloudDownload
                    },
                    contentDescription = null,
                    tint = when {
                        slotStatus.isInstalled -> NeonReadyGreen
                        isIncomplete -> Color(0xFFFF6D00)
                        else -> SoftAmberStatus
                    },
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
                        text = when {
                            slotStatus.isInstalled ->
                                "$displayFileName • ${OnnxModelManager.formatBytes(slotStatus.totalBytes)}"
                            isIncomplete -> if (isTamil) {
                                "$displayFileName (${OnnxModelManager.formatBytes(slotStatus.sizeBytes)}) • ⚠️ ${slotStatus.missingCompanionFiles.joinToString()} விடுபட்டுள்ளது"
                            } else {
                                "$displayFileName (${OnnxModelManager.formatBytes(slotStatus.sizeBytes)}) • ⚠️ Missing ${slotStatus.missingCompanionFiles.joinToString()}"
                            }
                            isTamil -> "$displayFileName • இன்னும் சேர்க்கப்படவில்லை"
                            else -> "$displayFileName • Not uploaded yet"
                        },
                        color = if (isIncomplete) Color(0xFFE65100) else subColor,
                        style = MaterialTheme.typography.bodyMedium.copy(
                            fontWeight = if (isIncomplete) FontWeight.Bold else FontWeight.Normal
                        )
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

                if (slotStatus.isInstalled || slotStatus.hasGraphFile) {
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

            if (isIncomplete) {
                val missingName = slotStatus.missingCompanionFiles.firstOrNull() ?: "weights.pb"
                Spacer(modifier = Modifier.height(8.dp))
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.Center,
                    modifier = Modifier
                        .fillMaxWidth()
                        .minimumInteractiveComponentSize()
                        .clip(RoundedCornerShape(14.dp))
                        .background(Brush.horizontalGradient(listOf(PinkGlowStart, VibrantPinkPill)))
                        .border(1.dp, Color.White.copy(alpha = 0.85f), RoundedCornerShape(14.dp))
                        .clickable { onAttachCompanionWeight(slot) }
                        .padding(vertical = 8.dp, horizontal = 12.dp)
                        .testTag("attach_companion_${slot.folderHint}")
                ) {
                    Icon(
                        imageVector = Icons.Filled.FileOpen,
                        contentDescription = null,
                        tint = Color.White,
                        modifier = Modifier.size(18.dp)
                    )
                    Spacer(modifier = Modifier.width(6.dp))
                    Text(
                        text = if (isTamil) {
                            "+ $missingName எடை ஃபைலை இணைக்கவும் (Attach)"
                        } else {
                            "+ Attach $missingName External Weights File"
                        },
                        color = Color.White,
                        style = MaterialTheme.typography.labelLarge.copy(fontWeight = FontWeight.Bold)
                    )
                }
            }
        }
    }
}

@Composable
fun CreationsCollectionScreen(
    title: String,
    emptySubtitle: String,
    creations: List<GeneratedCreation>,
    isFavoritesView: Boolean,
    darkTheme: Boolean,
    showBackButton: Boolean = false,
    onBack: () -> Unit = {},
    onStartCreating: () -> Unit,
    onToggleFavorite: (String) -> Unit,
    onSaveToGallery: (GeneratedCreation) -> Unit,
    onDeleteCreation: (String) -> Unit,
    modifier: Modifier = Modifier
) {
    if (showBackButton) {
        BackHandler { onBack() }
    }

    val textColor = if (darkTheme) Color.White else DeepPlumText
    val subColor = if (darkTheme) Color(0xFFE1C9F7) else Color(0xFF4F376E)

    if (creations.isEmpty()) {
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
                        imageVector = if (isFavoritesView) Icons.Filled.StarBorder else Icons.Filled.FavoriteBorder,
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
                        text = emptySubtitle,
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
        return
    }

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
                if (showBackButton) {
                    CircularGlowIconButton(
                        onClick = onBack,
                        contentDescription = "Back",
                        testTag = "gallery_back_button"
                    ) {
                        Icon(
                            imageVector = Icons.AutoMirrored.Filled.ArrowBack,
                            contentDescription = "Back",
                            tint = Color.White
                        )
                    }
                    Spacer(modifier = Modifier.width(12.dp))
                }
                Text(
                    text = "$title (${creations.size})",
                    color = textColor,
                    style = MaterialTheme.typography.headlineMedium.copy(fontWeight = FontWeight.ExtraBold)
                )
            }

            creations.forEach { item ->
                FrostedGlassCard(darkTheme = darkTheme) {
                    AsyncImage(
                        model = File(item.filePath),
                        contentDescription = item.userPrompt,
                        contentScale = ContentScale.Crop,
                        modifier = Modifier
                            .fillMaxWidth()
                            .aspectRatio(1f)
                            .clip(RoundedCornerShape(16.dp))
                            .border(1.2.dp, Color.White, RoundedCornerShape(16.dp))
                    )
                    Spacer(modifier = Modifier.height(8.dp))
                    Text(
                        text = item.userPrompt.ifBlank { "Generated Image" },
                        color = textColor,
                        style = MaterialTheme.typography.bodyLarge.copy(fontWeight = FontWeight.SemiBold),
                        maxLines = 2,
                        overflow = TextOverflow.Ellipsis
                    )
                    Text(
                        text = "${item.resolution} • ${item.steps} steps • Seed ${item.seed}",
                        color = subColor,
                        style = MaterialTheme.typography.labelSmall
                    )
                    Spacer(modifier = Modifier.height(8.dp))
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
                                .clip(RoundedCornerShape(14.dp))
                                .background(Brush.horizontalGradient(listOf(PurpleActionStart, PurpleActionEnd)))
                                .clickable { onSaveToGallery(item) }
                                .padding(vertical = 8.dp)
                        ) {
                            Icon(
                                imageVector = Icons.Filled.Download,
                                contentDescription = "Save to Gallery",
                                tint = Color.White,
                                modifier = Modifier.size(18.dp)
                            )
                            Spacer(modifier = Modifier.width(6.dp))
                            Text(
                                text = "Gallery",
                                color = Color.White,
                                style = MaterialTheme.typography.labelLarge.copy(fontWeight = FontWeight.Bold)
                            )
                        }

                        IconButton(onClick = { onToggleFavorite(item.id) }) {
                            Icon(
                                imageVector = if (item.isFavorite) Icons.Filled.Star else Icons.Outlined.StarOutline,
                                contentDescription = "Toggle Favorite",
                                tint = VibrantPinkPill
                            )
                        }

                        IconButton(onClick = { onDeleteCreation(item.id) }) {
                            Icon(
                                imageVector = Icons.Filled.DeleteOutline,
                                contentDescription = "Delete",
                                tint = Color(0xFFE53935)
                            )
                        }
                    }
                }
            }
            Spacer(modifier = Modifier.height(16.dp))
        }
    }
}
