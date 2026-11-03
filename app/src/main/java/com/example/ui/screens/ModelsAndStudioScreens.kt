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
import com.example.ai.PreferredUnetImportMode
import com.example.ai.RealDeviceAndModelMetrics
import com.example.ai.UnetVariant
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

import androidx.compose.material.icons.filled.Link
import androidx.compose.material.icons.filled.UploadFile
import com.example.ai.AutoDownloadProgressState
import com.example.ai.DedicatedModelSlot
import com.example.ai.DedicatedSlotStatus
import com.example.ai.EngineLinkState
import com.example.data.ActiveDiffusionEngine
import com.example.ui.components.ActiveEngineSwitchCard
import com.example.ui.components.AutoDownloadLiveBanner

@Composable
fun ModelsTabScreen(
    preferences: AppPreferencesState,
    metrics: RealDeviceAndModelMetrics,
    isImporting: Boolean,
    importProgressText: String?,
    verificationReport: String?,
    preferredUnetMode: PreferredUnetImportMode = PreferredUnetImportMode.AUTO_FAST_MOBILE,
    autoDownloadProgress: AutoDownloadProgressState = AutoDownloadProgressState(),
    onOpenAutoDownloadDialog: () -> Unit = {},
    onCancelAutoDownload: () -> Unit = {},
    onClearAutoDownloadBanner: () -> Unit = {},
    onSelectEngine: (ActiveDiffusionEngine) -> Unit = {},
    onSelectPreferredUnetMode: (PreferredUnetImportMode) -> Unit = {},
    onSwitchUnetFromFolder: (PreferredUnetImportMode) -> Unit = {},
    onOneTouchFolderUpload: () -> Unit,
    onVerifyPipeline: () -> Unit,
    onDedicatedSlotUpload: (DedicatedModelSlot) -> Unit = {},
    onTokenizerSlotUpload: () -> Unit = {},
    onDeleteDedicatedSlot: (DedicatedModelSlot) -> Unit = {},
    onSingleSlotUpload: (DiffusionModelSlot) -> Unit = {},
    onAttachCompanionWeight: (DiffusionModelSlot) -> Unit = {},
    onOpenSettingsDownloads: () -> Unit = {},
    onDeleteSlot: (DiffusionModelSlot) -> Unit = {},
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

            // Live Auto-Download Banner (when downloading or finished/paused)
            AutoDownloadLiveBanner(
                progress = autoDownloadProgress,
                isTamil = isTamil,
                darkTheme = darkTheme,
                onCancelDownload = onCancelAutoDownload,
                onResumeOrRetry = onOpenAutoDownloadDialog,
                onDismissBanner = onClearAutoDownloadBanner
            )

            // 1. Top 1-Touch Active Engine Switch (Fast LCM vs Standard SD)
            ActiveEngineSwitchCard(
                activeEngine = preferences.activeEngine,
                metrics = metrics,
                isTamil = isTamil,
                darkTheme = darkTheme,
                enabled = !isImporting && !autoDownloadProgress.isDownloading,
                tagPrefix = "models",
                onSelectEngine = onSelectEngine
            )

            // 2. Direct Auto-Download (1A–6) + Folder Auto-Import (Phone > IG) + Verify Card
            FrostedGlassCard(darkTheme = darkTheme) {
                Text(
                    text = if (isTamil) {
                        "நேரடி மாடல் பதிவிறக்கம் (1A–6) & Folder Auto-Import"
                    } else {
                        "Auto-Download (1A–6) & Folder Auto-Import"
                    },
                    color = textColor,
                    style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold)
                )
                Spacer(modifier = Modifier.height(4.dp))
                val folderPathLabel = metrics.phoneModelsFolderPath.ifBlank {
                    "Android/media/com.aistudio.imagegenerator.nvmkai/IG_Models"
                }
                Text(
                    text = if (isTamil) {
                        "போன் ஃபோல்டர்: $folderPathLabel\n1A, 1B, 2A, 2B, 3, 4, 5, 6 மாடல்களை நேரடியாக போன் ஃபோல்டரில் பதிவிறக்கி தானாக இணைக்கலாம் அல்லது ஏற்கனவே உள்ள IG ஃபோல்டரைத் தேர்ந்தெடுக்கலாம்."
                    } else {
                        "Phone Folder: $folderPathLabel\nDownload 1A, 1B, 2A, 2B, 3, 4, 5, 6 directly into your phone folder & auto-link, or import from an existing IG folder."
                    },
                    color = subColor,
                    style = MaterialTheme.typography.bodyMedium
                )
                Spacer(modifier = Modifier.height(10.dp))

                // Auto-Download Core Slots (1A–6) Button
                val autoBtnShape = RoundedCornerShape(22.dp)
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.Center,
                    modifier = Modifier
                        .fillMaxWidth()
                        .minimumInteractiveComponentSize()
                        .shadow(8.dp, autoBtnShape, spotColor = VibrantPinkPill)
                        .clip(autoBtnShape)
                        .background(Brush.horizontalGradient(listOf(PurpleActionStart, VibrantPinkPill)))
                        .border(1.3.dp, Color.White.copy(alpha = 0.88f), autoBtnShape)
                        .clickable(
                            enabled = !isImporting && !autoDownloadProgress.isDownloading,
                            onClick = onOpenAutoDownloadDialog
                        )
                        .padding(vertical = 12.dp, horizontal = 14.dp)
                        .testTag("models_auto_download_core_button")
                ) {
                    Icon(
                        imageVector = Icons.Filled.Download,
                        contentDescription = null,
                        tint = Color.White,
                        modifier = Modifier.size(21.dp)
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(
                        text = if (autoDownloadProgress.isDownloading) {
                            if (isTamil) "IG_Models-ல் பதிவிறக்கம் நடைபெறுகிறது (${autoDownloadProgress.overallPercent}%)..."
                            else "Auto-Downloading to IG_Models (${autoDownloadProgress.overallPercent}%)..."
                        } else if (isTamil) {
                            "1A–6 மாடல்களை தானாக Download & இணைக்கவும்"
                        } else {
                            "Auto-Download Core Slots (1A–6) to Phone"
                        },
                        color = Color.White,
                        style = MaterialTheme.typography.titleSmall.copy(fontWeight = FontWeight.ExtraBold)
                    )
                }

                Spacer(modifier = Modifier.height(8.dp))

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
                            "Folder தேர்வு செய்து 8 ஸ்லாட்களையும் நிரப்புக"
                        } else {
                            "Select Folder & Auto-Fill All Slots"
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
                    text = "Core Slots: ${metrics.coreInstalledCount}/8 • Face Swap: ${metrics.faceSwapInstalledCount}/3 • Disk: ${OnnxModelManager.formatBytes(metrics.totalModelBytes)} • Free: ${OnnxModelManager.formatBytes(metrics.deviceFreeBytes)}",
                    color = subColor,
                    style = MaterialTheme.typography.labelLarge
                )
            }

            // 3A. Fast LCM Engine (Slot 1A & Slot 1B)
            EngineGroupHeaderRow(
                title = if (isTamil) {
                    "⚡ Fast LCM Engine (பரிந்துரைக்கப்படும் வேகமான மாடல்)"
                } else {
                    "⚡ Fast LCM Engine (Recommended Fast Model)"
                },
                linkState = metrics.fastLcmLink,
                isTamil = isTamil,
                textColor = textColor
            )

            metrics.fastLcmDedicatedSlots.forEach { slotStatus ->
                DedicatedSlotItemCard(
                    slotStatus = slotStatus,
                    isTamil = isTamil,
                    darkTheme = darkTheme,
                    isImporting = isImporting,
                    textColor = textColor,
                    subColor = subColor,
                    onUploadClick = { onDedicatedSlotUpload(slotStatus.slot) },
                    onDeleteClick = { onDeleteDedicatedSlot(slotStatus.slot) }
                )
            }

            // 3B. Standard SD 1.5 Engine (Slot 2A & Slot 2B)
            Spacer(modifier = Modifier.height(4.dp))
            EngineGroupHeaderRow(
                title = if (isTamil) {
                    "🎨 Standard SD 1.5 Engine (ஹை-குவாலிட்டி மாடல்)"
                } else {
                    "🎨 Standard SD 1.5 Engine (High-Quality Model)"
                },
                linkState = metrics.standardSdLink,
                isTamil = isTamil,
                textColor = textColor
            )

            metrics.standardSdDedicatedSlots.forEach { slotStatus ->
                DedicatedSlotItemCard(
                    slotStatus = slotStatus,
                    isTamil = isTamil,
                    darkTheme = darkTheme,
                    isImporting = isImporting,
                    textColor = textColor,
                    subColor = subColor,
                    onUploadClick = { onDedicatedSlotUpload(slotStatus.slot) },
                    onDeleteClick = { onDeleteDedicatedSlot(slotStatus.slot) }
                )
            }

            // 3C. Other Essential Core Models (Slot 3, Slot 4, Slot 5, Slot 6)
            Spacer(modifier = Modifier.height(4.dp))
            val sharedReadyCount = metrics.essentialSharedDedicatedSlots.count { it.isInstalled }
            Text(
                text = if (isTamil) {
                    "மற்ற அத்தியாவசிய மாடல்கள் ($sharedReadyCount/4 • Slot 3–6)"
                } else {
                    "Other Essential Core Models ($sharedReadyCount/4 • Slot 3–6)"
                },
                color = textColor,
                style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.ExtraBold)
            )

            metrics.essentialSharedDedicatedSlots.forEach { slotStatus ->
                DedicatedSlotItemCard(
                    slotStatus = slotStatus,
                    isTamil = isTamil,
                    darkTheme = darkTheme,
                    isImporting = isImporting,
                    textColor = textColor,
                    subColor = subColor,
                    onUploadClick = {
                        if (slotStatus.slot == DedicatedModelSlot.TOKENIZER) {
                            onTokenizerSlotUpload()
                        } else {
                            onDedicatedSlotUpload(slotStatus.slot)
                        }
                    },
                    onDeleteClick = { onDeleteDedicatedSlot(slotStatus.slot) }
                )
            }

            // 4. Separate Optional Face Swap Models Section (3)
            Spacer(modifier = Modifier.height(6.dp))
            Text(
                text = if (isTamil) {
                    "Optional Face Swap மாடல்கள் (${metrics.faceSwapInstalledCount}/3)"
                } else {
                    "Optional Face Swap Models (${metrics.faceSwapInstalledCount}/3)"
                },
                color = textColor,
                style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.ExtraBold)
            )

            metrics.faceSwapDedicatedSlots.forEach { slotStatus ->
                DedicatedSlotItemCard(
                    slotStatus = slotStatus,
                    isTamil = isTamil,
                    darkTheme = darkTheme,
                    isImporting = isImporting,
                    textColor = textColor,
                    subColor = subColor,
                    onUploadClick = { onDedicatedSlotUpload(slotStatus.slot) },
                    onDeleteClick = { onDeleteDedicatedSlot(slotStatus.slot) }
                )
            }

            Spacer(modifier = Modifier.height(16.dp))
        }
    }
}

@Composable
private fun EngineGroupHeaderRow(
    title: String,
    linkState: EngineLinkState,
    isTamil: Boolean,
    textColor: Color
) {
    val partsUploaded = (if (linkState.isGraphInstalled) 1 else 0) + (if (linkState.isWeightsInstalled) 1 else 0)
    val badgeText = when {
        linkState.isPairLinked -> if (isTamil) "🔗 இணைக்கப்பட்டது (2/2)" else "🔗 Linked (2/2)"
        partsUploaded == 1 -> if (isTamil) "⚠️ பாதி (1/2)" else "⚠️ Partial (1/2)"
        else -> "0/2"
    }
    val badgeColor = when {
        linkState.isPairLinked -> NeonReadyGreen
        partsUploaded == 1 -> Color(0xFFFF9800)
        else -> SoftAmberStatus
    }

    Row(
        verticalAlignment = Alignment.CenterVertically,
        modifier = Modifier.fillMaxWidth()
    ) {
        Text(
            text = title,
            color = textColor,
            style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.ExtraBold),
            modifier = Modifier.weight(1f)
        )
        Spacer(modifier = Modifier.width(6.dp))
        Text(
            text = badgeText,
            color = badgeColor,
            style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.ExtraBold)
        )
    }
}

@Composable
private fun DedicatedSlotItemCard(
    slotStatus: DedicatedSlotStatus,
    isTamil: Boolean,
    darkTheme: Boolean,
    isImporting: Boolean,
    textColor: Color,
    subColor: Color,
    onUploadClick: () -> Unit,
    onDeleteClick: () -> Unit
) {
    val slot = slotStatus.slot
    val statusLine = if (isTamil) slotStatus.statusDetailTa else slotStatus.statusDetailEn
    val showDelete = if (slot == DedicatedModelSlot.TOKENIZER) {
        slotStatus.sizeBytes > 0L
    } else {
        slotStatus.isInstalled || slotStatus.isPartial || slotStatus.sizeBytes > 0L
    }

    FrostedGlassCard(darkTheme = darkTheme) {
        Column(modifier = Modifier.fillMaxWidth()) {
            Row(
                verticalAlignment = Alignment.Top,
                modifier = Modifier.fillMaxWidth()
            ) {
                Icon(
                    imageVector = when {
                        slotStatus.isLinkedWithPair -> Icons.Filled.Link
                        slotStatus.isInstalled -> Icons.Filled.CheckCircle
                        slotStatus.isPartial -> Icons.Filled.WarningAmber
                        else -> Icons.Filled.CloudDownload
                    },
                    contentDescription = null,
                    tint = when {
                        slotStatus.isInstalled -> NeonReadyGreen
                        slotStatus.isPartial -> Color(0xFFFF6D00)
                        else -> SoftAmberStatus
                    },
                    modifier = Modifier
                        .padding(top = 2.dp)
                        .size(23.dp)
                )
                Spacer(modifier = Modifier.width(10.dp))
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        text = if (isTamil) slot.titleTa else slot.titleEn,
                        color = textColor,
                        style = MaterialTheme.typography.titleSmall.copy(fontWeight = FontWeight.ExtraBold)
                    )
                    Spacer(modifier = Modifier.height(2.dp))
                    Text(
                        text = statusLine,
                        color = if (slotStatus.isInstalled) {
                            if (darkTheme) NeonReadyGreen else Color(0xFF1B5E20)
                        } else if (slotStatus.isPartial) {
                            Color(0xFFE65100)
                        } else {
                            subColor
                        },
                        style = MaterialTheme.typography.bodySmall.copy(
                            fontWeight = if (slotStatus.isInstalled || slotStatus.isPartial) FontWeight.Bold else FontWeight.Normal
                        )
                    )
                    Spacer(modifier = Modifier.height(2.dp))
                    Text(
                        text = if (isTamil) slot.descriptionTa else slot.descriptionEn,
                        color = subColor.copy(alpha = 0.85f),
                        style = MaterialTheme.typography.labelSmall
                    )
                }
            }

            Spacer(modifier = Modifier.height(10.dp))

            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                modifier = Modifier.fillMaxWidth()
            ) {
                val uploadBtnShape = RoundedCornerShape(14.dp)
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.Center,
                    modifier = Modifier
                        .weight(1f)
                        .minimumInteractiveComponentSize()
                        .clip(uploadBtnShape)
                        .background(
                            Brush.horizontalGradient(
                                if (slotStatus.isInstalled && (slot != DedicatedModelSlot.TOKENIZER || slotStatus.sizeBytes > 0L)) {
                                    listOf(PurpleActionStart, PurpleActionEnd)
                                } else {
                                    listOf(PinkGlowStart, VibrantPinkPill)
                                }
                            )
                        )
                        .border(1.dp, Color.White.copy(alpha = 0.85f), uploadBtnShape)
                        .clickable(enabled = !isImporting, onClick = onUploadClick)
                        .padding(vertical = 9.dp, horizontal = 12.dp)
                        .testTag("upload_slot_${slot.slotTag}")
                ) {
                    Icon(
                        imageVector = Icons.Filled.UploadFile,
                        contentDescription = "Upload ${slot.titleEn}",
                        tint = Color.White,
                        modifier = Modifier.size(18.dp)
                    )
                    Spacer(modifier = Modifier.width(6.dp))
                    Text(
                        text = when {
                            slot == DedicatedModelSlot.TOKENIZER && slotStatus.sizeBytes == 0L ->
                                if (isTamil) "Upload vocab.json & merges.txt" else "Upload vocab.json & merges.txt"
                            slotStatus.isInstalled ->
                                if (isTamil) "மாற்று (Replace File)" else "Replace File"
                            else ->
                                if (isTamil) "Upload (${slot.expectedFileHint.substringBefore(" (")})"
                                else "Upload File"
                        },
                        color = Color.White,
                        style = MaterialTheme.typography.labelLarge.copy(fontWeight = FontWeight.Bold),
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis
                    )
                }

                if (showDelete) {
                    IconButton(
                        onClick = onDeleteClick,
                        enabled = !isImporting,
                        modifier = Modifier
                            .minimumInteractiveComponentSize()
                            .clip(RoundedCornerShape(14.dp))
                            .background(Color.White.copy(alpha = if (darkTheme) 0.12f else 0.65f))
                            .border(1.dp, Color(0xFFE53935).copy(alpha = 0.55f), RoundedCornerShape(14.dp))
                            .testTag("delete_slot_${slot.slotTag}")
                    ) {
                        Icon(
                            imageVector = Icons.Filled.DeleteOutline,
                            contentDescription = "Delete ${slot.titleEn}",
                            tint = Color(0xFFE53935)
                        )
                    }
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
