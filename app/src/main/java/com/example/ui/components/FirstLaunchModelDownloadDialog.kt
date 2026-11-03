package com.example.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CheckBox
import androidx.compose.material.icons.filled.CheckBoxOutlineBlank
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.CloudDownload
import androidx.compose.material.icons.filled.Folder
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.minimumInteractiveComponentSize
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.Dialog
import androidx.compose.ui.window.DialogProperties
import com.example.ai.AutoDownloadProgressState
import com.example.ai.DedicatedModelSlot
import com.example.ai.OnnxAutoModelDownloader
import com.example.ai.OnnxModelManager
import com.example.ai.RealDeviceAndModelMetrics
import com.example.ui.theme.DeepPlumText
import com.example.ui.theme.NeonReadyGreen
import com.example.ui.theme.PinkGlowEnd
import com.example.ui.theme.PinkGlowStart
import com.example.ui.theme.PurpleActionEnd
import com.example.ui.theme.PurpleActionStart
import com.example.ui.theme.SoftAmberStatus
import com.example.ui.theme.VibrantPinkPill

@Composable
fun FirstLaunchModelDownloadDialog(
    metrics: RealDeviceAndModelMetrics,
    isTamil: Boolean,
    darkTheme: Boolean,
    onAllowAndDownload: (Set<DedicatedModelSlot>) -> Unit,
    onDismissLater: () -> Unit
) {
    val installedCoreSet = remember(metrics.dedicatedSlots) {
        metrics.coreDedicatedSlots.filter { it.isInstalled }.map { it.slot }.toSet()
    }
    var selectedSlots by remember {
        mutableStateOf(OnnxAutoModelDownloader.DEFAULT_SELECTED_CORE_SLOTS)
    }

    val dialogShape = RoundedCornerShape(26.dp)
    val bgBrush = if (darkTheme) {
        Brush.verticalGradient(listOf(Color(0xFF28173D), Color(0xFF1A0E29)))
    } else {
        Brush.verticalGradient(listOf(Color(0xFFFFF6FC), Color(0xFFF6E9FF)))
    }
    val textColor = if (darkTheme) Color.White else DeepPlumText
    val subColor = if (darkTheme) Color(0xFFE1C9F7) else Color(0xFF4F376E)

    val specsSortedForUi = remember {
        OnnxAutoModelDownloader.CORE_SLOT_SPECS.sortedBy { it.displayOrder }
    }
    val pendingDownloadBytes = specsSortedForUi
        .filter { it.slot in selectedSlots && it.slot !in installedCoreSet }
        .sumOf { it.totalExpectedBytes }

    val folderDisplay = metrics.phoneModelsFolderPath.ifBlank {
        "Android/media/com.aistudio.imagegenerator.nvmkai/IG_Models"
    }

    Dialog(
        onDismissRequest = onDismissLater,
        properties = DialogProperties(
            dismissOnBackPress = true,
            dismissOnClickOutside = false,
            usePlatformDefaultWidth = false
        )
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 16.dp, vertical = 20.dp)
                .widthIn(max = 520.dp)
                .shadow(18.dp, dialogShape, spotColor = PurpleActionEnd)
                .clip(dialogShape)
                .background(bgBrush)
                .border(1.5.dp, Color.White.copy(alpha = if (darkTheme) 0.35f else 0.9f), dialogShape)
                .padding(18.dp)
                .testTag("first_launch_model_download_dialog")
        ) {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                modifier = Modifier.fillMaxWidth()
            ) {
                Box(
                    contentAlignment = Alignment.Center,
                    modifier = Modifier
                        .size(42.dp)
                        .clip(RoundedCornerShape(14.dp))
                        .background(Brush.linearGradient(listOf(PinkGlowStart, PurpleActionEnd)))
                ) {
                    Icon(
                        imageVector = Icons.Filled.CloudDownload,
                        contentDescription = null,
                        tint = Color.White,
                        modifier = Modifier.size(24.dp)
                    )
                }
                Spacer(modifier = Modifier.width(10.dp))
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        text = if (isTamil) {
                            "Offline AI மாடல்களைப் பதிவிறக்கவா?"
                        } else {
                            "Download Core AI Models (1A–6)?"
                        },
                        color = textColor,
                        style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.ExtraBold)
                    )
                    Text(
                        text = if (isTamil) {
                            "1A, 1B, 2A, 2B, 3, 4, 5, 6 ஸ்லாட்டுகளை நேரடியாக போன் ஃபோல்டரில் சேமித்து தானாக இணைக்க அனுமதி கேட்கிறது"
                        } else {
                            "Allow permission to download 1A, 1B, 2A, 2B, 3, 4, 5, 6 into your phone folder & auto-activate"
                        },
                        color = subColor,
                        style = MaterialTheme.typography.labelSmall
                    )
                }
            }

            Spacer(modifier = Modifier.height(10.dp))

            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .weight(1f, fill = false)
                    .verticalScroll(rememberScrollState())
            ) {
                // Created Phone Folder Path Badge
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    modifier = Modifier
                        .fillMaxWidth()
                        .clip(RoundedCornerShape(12.dp))
                        .background(Color.White.copy(alpha = if (darkTheme) 0.10f else 0.72f))
                        .border(1.dp, NeonReadyGreen.copy(alpha = 0.7f), RoundedCornerShape(12.dp))
                        .padding(horizontal = 10.dp, vertical = 8.dp)
                        .testTag("first_launch_folder_path_badge")
                ) {
                    Icon(
                        imageVector = Icons.Filled.Folder,
                        contentDescription = null,
                        tint = NeonReadyGreen,
                        modifier = Modifier.size(18.dp)
                    )
                    Spacer(modifier = Modifier.width(6.dp))
                    Text(
                        text = if (isTamil) {
                            "போன் ஃபோல்டர் தயார் ✓ : $folderDisplay"
                        } else {
                            "Phone Folder Ready ✓ : $folderDisplay"
                        },
                        color = textColor,
                        style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.SemiBold)
                    )
                }

                Spacer(modifier = Modifier.height(8.dp))

                // Quick Selection Presets (All 8 vs Fast LCM without 3.44 GB 2B)
                Row(
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                    modifier = Modifier.fillMaxWidth()
                ) {
                    val allSelected = selectedSlots.size == OnnxAutoModelDownloader.DEFAULT_SELECTED_CORE_SLOTS.size
                    val fastOnlySet = OnnxAutoModelDownloader.DEFAULT_SELECTED_CORE_SLOTS - DedicatedModelSlot.STANDARD_SD_WEIGHTS
                    val isFastOnly = selectedSlots == fastOnlySet

                    PresetChip(
                        label = if (isTamil) "அனைத்தும் (1A–6)" else "All 8 (1A–6)",
                        active = allSelected,
                        darkTheme = darkTheme,
                        textColor = textColor,
                        testTag = "preset_select_all_core",
                        onClick = { selectedSlots = OnnxAutoModelDownloader.DEFAULT_SELECTED_CORE_SLOTS },
                        modifier = Modifier.weight(1f)
                    )
                    PresetChip(
                        label = if (isTamil) "2B (3.4GB) தவிர்த்து" else "Skip 2B (Fast 1.6GB)",
                        active = isFastOnly,
                        darkTheme = darkTheme,
                        textColor = textColor,
                        testTag = "preset_skip_2b_heavy",
                        onClick = { selectedSlots = fastOnlySet },
                        modifier = Modifier.weight(1f)
                    )
                }

                Spacer(modifier = Modifier.height(8.dp))

                // Checklist of ONLY the 8 Core Slots (1A, 1B, 2A, 2B, 3, 4, 5, 6)
                Column(
                    verticalArrangement = Arrangement.spacedBy(6.dp),
                    modifier = Modifier.fillMaxWidth()
                ) {
                    specsSortedForUi.forEach { spec ->
                        val alreadyReady = spec.slot in installedCoreSet
                        val checked = alreadyReady || spec.slot in selectedSlots
                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            modifier = Modifier
                                .fillMaxWidth()
                                .minimumInteractiveComponentSize()
                                .clip(RoundedCornerShape(12.dp))
                                .background(Color.White.copy(alpha = if (darkTheme) 0.08f else 0.65f))
                                .border(
                                    width = 1.dp,
                                    color = if (checked) VibrantPinkPill.copy(alpha = 0.65f) else Color.Transparent,
                                    shape = RoundedCornerShape(12.dp)
                                )
                                .clickable(enabled = !alreadyReady) {
                                    selectedSlots = if (spec.slot in selectedSlots) {
                                        selectedSlots - spec.slot
                                    } else {
                                        selectedSlots + spec.slot
                                    }
                                }
                                .padding(horizontal = 10.dp, vertical = 7.dp)
                                .testTag("download_slot_checkbox_${spec.shortCode.lowercase()}")
                        ) {
                            Icon(
                                imageVector = when {
                                    alreadyReady -> Icons.Filled.CheckCircle
                                    checked -> Icons.Filled.CheckBox
                                    else -> Icons.Filled.CheckBoxOutlineBlank
                                },
                                contentDescription = if (checked) "Selected ${spec.shortCode}" else "Unselected ${spec.shortCode}",
                                tint = when {
                                    alreadyReady -> NeonReadyGreen
                                    checked -> VibrantPinkPill
                                    else -> subColor
                                },
                                modifier = Modifier.size(20.dp)
                            )
                            Spacer(modifier = Modifier.width(8.dp))
                            Column(modifier = Modifier.weight(1f)) {
                                Text(
                                    text = if (isTamil) spec.titleTa else spec.titleEn,
                                    color = textColor,
                                    style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.Bold),
                                    maxLines = 1,
                                    overflow = TextOverflow.Ellipsis
                                )
                                Text(
                                    text = if (alreadyReady) {
                                        if (isTamil) "ஏற்கனவே போனில் தயார் ✓" else "Already installed on phone ✓"
                                    } else {
                                        OnnxModelManager.formatBytes(spec.totalExpectedBytes)
                                    },
                                    color = if (alreadyReady) NeonReadyGreen else subColor,
                                    style = MaterialTheme.typography.labelSmall
                                )
                            }
                        }
                    }
                }

                Spacer(modifier = Modifier.height(10.dp))

                Text(
                    text = if (isTamil) {
                        "தேர்வு: ${selectedSlots.size}/8 ஸ்லாட்டுகள் (${OnnxModelManager.formatBytes(pendingDownloadBytes)}) • போனில் காலியிடம்: ${OnnxModelManager.formatBytes(metrics.deviceFreeBytes)}\nகுறிப்பு: ஆப் Close/Open செய்தாலும், Update செய்தாலும் மாடல்கள் நீங்காது!"
                    } else {
                        "Selected: ${selectedSlots.size}/8 Slots (${OnnxModelManager.formatBytes(pendingDownloadBytes)}) • Phone Free: ${OnnxModelManager.formatBytes(metrics.deviceFreeBytes)}\nModels stay intact across app restarts and app updates."
                    },
                    color = subColor,
                    style = MaterialTheme.typography.labelSmall
                )
            }

            Spacer(modifier = Modifier.height(12.dp))

            Row(
                horizontalArrangement = Arrangement.spacedBy(10.dp),
                verticalAlignment = Alignment.CenterVertically,
                modifier = Modifier.fillMaxWidth()
            ) {
                Box(
                    contentAlignment = Alignment.Center,
                    modifier = Modifier
                        .weight(0.9f)
                        .minimumInteractiveComponentSize()
                        .clip(RoundedCornerShape(18.dp))
                        .background(Color.White.copy(alpha = if (darkTheme) 0.12f else 0.72f))
                        .border(1.dp, subColor.copy(alpha = 0.45f), RoundedCornerShape(18.dp))
                        .clickable(onClick = onDismissLater)
                        .padding(vertical = 12.dp, horizontal = 8.dp)
                        .testTag("first_launch_later_button")
                ) {
                    Text(
                        text = if (isTamil) "பிறகு (Later)" else "Later",
                        color = textColor,
                        style = MaterialTheme.typography.labelLarge.copy(fontWeight = FontWeight.Bold)
                    )
                }

                val allowEnabled = selectedSlots.isNotEmpty()
                Box(
                    contentAlignment = Alignment.Center,
                    modifier = Modifier
                        .weight(1.4f)
                        .minimumInteractiveComponentSize()
                        .shadow(8.dp, RoundedCornerShape(18.dp), spotColor = PurpleActionEnd)
                        .clip(RoundedCornerShape(18.dp))
                        .background(
                            if (allowEnabled) {
                                Brush.horizontalGradient(listOf(PinkGlowStart, PinkGlowEnd, PurpleActionEnd))
                            } else {
                                Brush.horizontalGradient(listOf(Color.Gray, Color.DarkGray))
                            }
                        )
                        .border(1.2.dp, Color.White.copy(alpha = 0.85f), RoundedCornerShape(18.dp))
                        .clickable(enabled = allowEnabled) { onAllowAndDownload(selectedSlots) }
                        .padding(vertical = 12.dp, horizontal = 10.dp)
                        .testTag("first_launch_allow_download_button")
                ) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Icon(
                            imageVector = Icons.Filled.CloudDownload,
                            contentDescription = null,
                            tint = Color.White,
                            modifier = Modifier.size(18.dp)
                        )
                        Spacer(modifier = Modifier.width(6.dp))
                        Text(
                            text = if (isTamil) "Allow & பதிவிறக்கு" else "Allow & Download",
                            color = Color.White,
                            style = MaterialTheme.typography.labelLarge.copy(fontWeight = FontWeight.ExtraBold)
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun PresetChip(
    label: String,
    active: Boolean,
    darkTheme: Boolean,
    textColor: Color,
    testTag: String,
    onClick: () -> Unit,
    modifier: Modifier = Modifier
) {
    val shape = RoundedCornerShape(12.dp)
    Box(
        contentAlignment = Alignment.Center,
        modifier = modifier
            .minimumInteractiveComponentSize()
            .clip(shape)
            .background(
                if (active) Brush.horizontalGradient(listOf(PurpleActionStart, PurpleActionEnd))
                else Brush.horizontalGradient(
                    listOf(
                        Color.White.copy(alpha = if (darkTheme) 0.12f else 0.65f),
                        Color.White.copy(alpha = if (darkTheme) 0.08f else 0.55f)
                    )
                )
            )
            .border(1.dp, if (active) Color.White else VibrantPinkPill.copy(alpha = 0.45f), shape)
            .clickable(onClick = onClick)
            .padding(vertical = 8.dp, horizontal = 8.dp)
            .testTag(testTag)
    ) {
        Text(
            text = label,
            color = if (active) Color.White else textColor,
            style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.Bold),
            maxLines = 1,
            overflow = TextOverflow.Ellipsis
        )
    }
}

@Composable
fun AutoDownloadLiveBanner(
    progress: AutoDownloadProgressState,
    isTamil: Boolean,
    darkTheme: Boolean,
    onCancelDownload: () -> Unit,
    onResumeOrRetry: () -> Unit,
    onDismissBanner: () -> Unit,
    modifier: Modifier = Modifier
) {
    val hasFinishedMsg = !progress.finishedSummaryEn.isNullOrBlank()
    val hasErrorMsg = !progress.errorTextEn.isNullOrBlank()
    if (!progress.isDownloading && !hasFinishedMsg && !hasErrorMsg) return

    val textColor = if (darkTheme) Color.White else DeepPlumText
    val subColor = if (darkTheme) Color(0xFFE1C9F7) else Color(0xFF4F376E)
    val cardShape = RoundedCornerShape(18.dp)

    Column(
        modifier = modifier
            .fillMaxWidth()
            .shadow(10.dp, cardShape, spotColor = PurpleActionEnd)
            .clip(cardShape)
            .background(
                if (darkTheme) Color(0xEE26153A) else Color(0xF5FFF5FC)
            )
            .border(
                width = 1.2.dp,
                color = when {
                    progress.isDownloading -> VibrantPinkPill
                    hasFinishedMsg -> NeonReadyGreen
                    else -> SoftAmberStatus
                },
                shape = cardShape
            )
            .padding(horizontal = 14.dp, vertical = 10.dp)
            .testTag("auto_download_live_banner")
    ) {
        if (progress.isDownloading) {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                modifier = Modifier.fillMaxWidth()
            ) {
                Icon(
                    imageVector = Icons.Filled.CloudDownload,
                    contentDescription = null,
                    tint = VibrantPinkPill,
                    modifier = Modifier.size(20.dp)
                )
                Spacer(modifier = Modifier.width(8.dp))
                Text(
                    text = if (isTamil) {
                        "IG_Models தானியங்கி பதிவிறக்கம் (${progress.overallPercent}%)"
                    } else {
                        "Auto-Downloading to IG_Models (${progress.overallPercent}%)"
                    },
                    color = textColor,
                    style = MaterialTheme.typography.labelLarge.copy(fontWeight = FontWeight.ExtraBold),
                    modifier = Modifier.weight(1f)
                )
                IconButton(
                    onClick = onCancelDownload,
                    modifier = Modifier
                        .size(48.dp)
                        .testTag("auto_download_cancel_button")
                ) {
                    Icon(
                        imageVector = Icons.Filled.Close,
                        contentDescription = if (isTamil) "பதிவிறக்கத்தை நிறுத்து" else "Pause Auto-Download",
                        tint = textColor
                    )
                }
            }
            Text(
                text = if (isTamil) progress.statusTextTa else progress.statusTextEn,
                color = subColor,
                style = MaterialTheme.typography.labelSmall
            )
            Spacer(modifier = Modifier.height(6.dp))
            LinearProgressIndicator(
                progress = { (progress.currentFilePercent / 100f).coerceIn(0f, 1f) },
                color = VibrantPinkPill,
                trackColor = VibrantPinkPill.copy(alpha = 0.2f),
                modifier = Modifier
                    .fillMaxWidth()
                    .height(6.dp)
                    .clip(RoundedCornerShape(4.dp))
            )
        } else if (hasFinishedMsg) {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                modifier = Modifier.fillMaxWidth()
            ) {
                Icon(
                    imageVector = Icons.Filled.CheckCircle,
                    contentDescription = null,
                    tint = NeonReadyGreen,
                    modifier = Modifier.size(20.dp)
                )
                Spacer(modifier = Modifier.width(8.dp))
                Text(
                    text = (if (isTamil) progress.finishedSummaryTa else progress.finishedSummaryEn).orEmpty(),
                    color = textColor,
                    style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.Bold),
                    modifier = Modifier.weight(1f)
                )
                IconButton(
                    onClick = onDismissBanner,
                    modifier = Modifier
                        .size(48.dp)
                        .testTag("auto_download_dismiss_banner_button")
                ) {
                    Icon(
                        imageVector = Icons.Filled.Close,
                        contentDescription = "Dismiss download summary",
                        tint = textColor
                    )
                }
            }
        } else if (hasErrorMsg) {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                modifier = Modifier.fillMaxWidth()
            ) {
                Text(
                    text = (if (isTamil) progress.errorTextTa else progress.errorTextEn).orEmpty(),
                    color = textColor,
                    style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.SemiBold),
                    modifier = Modifier.weight(1f)
                )
                IconButton(
                    onClick = onResumeOrRetry,
                    modifier = Modifier
                        .size(48.dp)
                        .testTag("auto_download_resume_button")
                ) {
                    Icon(
                        imageVector = Icons.Filled.Refresh,
                        contentDescription = if (isTamil) "மீண்டும் தொடர்" else "Resume Auto-Download",
                        tint = VibrantPinkPill
                    )
                }
                IconButton(
                    onClick = onDismissBanner,
                    modifier = Modifier
                        .size(48.dp)
                        .testTag("auto_download_dismiss_error_button")
                ) {
                    Icon(
                        imageVector = Icons.Filled.Close,
                        contentDescription = "Dismiss message",
                        tint = textColor
                    )
                }
            }
        }
    }
}
