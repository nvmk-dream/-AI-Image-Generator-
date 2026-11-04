package com.example.ui.screens

import android.Manifest
import android.os.Build
import androidx.activity.compose.BackHandler
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
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
import androidx.compose.material.icons.automirrored.filled.HelpOutline
import androidx.compose.material.icons.automirrored.filled.KeyboardArrowRight
import androidx.compose.material.icons.filled.Bolt
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.CleaningServices
import androidx.compose.material.icons.filled.DeleteSweep
import androidx.compose.material.icons.filled.EditNote
import androidx.compose.material.icons.filled.Favorite
import androidx.compose.material.icons.filled.FolderOpen
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.KeyboardArrowDown
import androidx.compose.material.icons.filled.Language
import androidx.compose.material.icons.filled.Memory
import androidx.compose.material.icons.filled.Notifications
import androidx.compose.material.icons.filled.Palette
import androidx.compose.material.icons.filled.PhotoSizeSelectLarge
import androidx.compose.material.icons.filled.Security
import androidx.compose.material.icons.filled.Storage
import androidx.compose.material.icons.filled.Tune
import androidx.compose.material.icons.filled.WifiOff
import androidx.compose.material.icons.outlined.FavoriteBorder
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Switch
import androidx.compose.material3.SwitchDefaults
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
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
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.BuildConfig
import com.example.ai.OfflineTamilPromptTranslator
import com.example.ai.OnnxModelManager
import com.example.ai.RealDeviceAndModelMetrics
import com.example.data.ActiveDiffusionEngine
import com.example.data.AppLanguage
import com.example.data.AppPreferencesState
import com.example.ui.components.CircularGlowIconButton
import com.example.ui.components.FrostedGlassCard
import com.example.ui.components.OnnxDownloadLinksCard
import com.example.ui.components.PaletteAppLogo
import com.example.ui.components.TamilDictionaryCard
import com.example.ui.theme.CyanVersionAccent
import com.example.ui.theme.DeepPlumText
import com.example.ui.theme.NeonReadyGreen
import com.example.ui.theme.PinkGlowStart
import com.example.ui.theme.PurpleActionEnd
import com.example.ui.theme.PurpleActionStart
import com.example.ui.theme.SoftAmberStatus
import com.example.ui.theme.VibrantPinkPill

@Composable
fun SettingsScreen(
    preferences: AppPreferencesState,
    metrics: RealDeviceAndModelMetrics,
    onBack: () -> Unit,
    onOpenModelsManager: () -> Unit,
    onOpenContentSafetyPage: () -> Unit,
    onResolutionChange: (String) -> Unit,
    onInferenceModeChange: (String) -> Unit,
    onOfflineModeChange: (Boolean) -> Unit,
    onLanguageChange: (AppLanguage) -> Unit,
    onDarkThemeChange: (Boolean) -> Unit,
    onNotificationsChange: (Boolean) -> Unit,
    onContentFilterToggle: (Boolean) -> Unit,
    onClearCache: () -> Unit,
    onUnloadRam: () -> Unit,
    onUpsertTamilWord: (String, String) -> Unit = { _, _ -> },
    onDeleteTamilWord: (String) -> Unit = {},
    onResetTamilDictionary: () -> Unit = {},
    modifier: Modifier = Modifier
) {
    BackHandler { onBack() }

    val isTamil = preferences.isTamil
    val darkTheme = preferences.darkTheme
    val primaryTextColor = if (darkTheme) Color.White else DeepPlumText
    val secondaryTextColor = if (darkTheme) Color(0xFFE1C9F7) else Color(0xFF4F376E)

    var showHelpDialog by remember { mutableStateOf(false) }
    var showPrivacyDialog by remember { mutableStateOf(false) }

    val notificationPermissionLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission()
    ) { granted ->
        onNotificationsChange(granted)
    }

    Column(
        horizontalAlignment = Alignment.CenterHorizontally,
        modifier = modifier
            .fillMaxSize()
            .windowInsetsPadding(WindowInsets.statusBars)
            .windowInsetsPadding(WindowInsets.navigationBars)
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
            // Top Settings Header
            Row(
                verticalAlignment = Alignment.CenterVertically,
                modifier = Modifier.fillMaxWidth()
            ) {
                CircularGlowIconButton(
                    onClick = onBack,
                    contentDescription = "Navigate Back",
                    testTag = "settings_back_button"
                ) {
                    Icon(
                        imageVector = Icons.AutoMirrored.Filled.ArrowBack,
                        contentDescription = "Back",
                        tint = Color.White,
                        modifier = Modifier.size(22.dp)
                    )
                }

                Text(
                    text = if (isTamil) "Settings (அமைப்புகள்)" else "Settings",
                    color = primaryTextColor,
                    style = MaterialTheme.typography.headlineMedium.copy(
                        fontWeight = FontWeight.ExtraBold,
                        fontSize = 23.sp
                    ),
                    modifier = Modifier
                        .weight(1f)
                        .padding(horizontal = 12.dp),
                    textAlign = androidx.compose.ui.text.style.TextAlign.Center
                )

                Box(
                    contentAlignment = Alignment.Center,
                    modifier = Modifier.size(48.dp)
                ) {
                    Icon(
                        imageVector = Icons.Outlined.FavoriteBorder,
                        contentDescription = null,
                        tint = VibrantPinkPill,
                        modifier = Modifier.size(30.dp)
                    )
                }
            }

            // App Identity Glass Card
            FrostedGlassCard(darkTheme = darkTheme) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    modifier = Modifier.fillMaxWidth()
                ) {
                    PaletteAppLogo(size = 62.dp)
                    Spacer(modifier = Modifier.width(14.dp))
                    Column(modifier = Modifier.weight(1f)) {
                        Text(
                            text = "Image Generator",
                            color = if (darkTheme) Color.White else Color(0xFF331B5E),
                            style = MaterialTheme.typography.titleLarge.copy(fontWeight = FontWeight.Bold)
                        )
                        Text(
                            text = "V ${BuildConfig.VERSION_NAME}",
                            color = if (darkTheme) CyanVersionAccent else Color(0xFF0097A7),
                            style = MaterialTheme.typography.bodyLarge.copy(fontWeight = FontWeight.Bold)
                        )
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Text(
                                text = "Create • Edit • Enjoy",
                                color = secondaryTextColor,
                                style = MaterialTheme.typography.bodyMedium
                            )
                            Spacer(modifier = Modifier.width(4.dp))
                            Icon(
                                imageVector = Icons.Filled.Favorite,
                                contentDescription = null,
                                tint = Color(0xFFFF6EC7),
                                modifier = Modifier.size(14.dp)
                            )
                        }
                    }
                }
            }

            // 1. AI Model Settings Card
            FrostedGlassCard(darkTheme = darkTheme) {
                SectionHeaderRow(
                    icon = Icons.Filled.Memory,
                    iconTint = PurpleActionEnd,
                    title = "AI Model Settings",
                    textColor = primaryTextColor
                )

                SettingsDivider(darkTheme)

                val isReady = metrics.ortRuntimeAvailable && metrics.isTxt2ImgReadyFor(preferences.activeEngine)
                val engineShort = if (preferences.activeEngine == ActiveDiffusionEngine.FAST_LCM) "Fast LCM" else "Standard SD"
                SettingsRowItem(
                    icon = Icons.Filled.CheckCircle,
                    iconTint = if (isReady) NeonReadyGreen else SoftAmberStatus,
                    label = "Model Status",
                    textColor = primaryTextColor
                ) {
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        modifier = Modifier
                            .clip(RoundedCornerShape(50))
                            .clickable(onClick = onOpenModelsManager)
                            .padding(horizontal = 6.dp, vertical = 4.dp)
                            .testTag("settings_model_status_badge")
                    ) {
                        Icon(
                            imageVector = Icons.Filled.CheckCircle,
                            contentDescription = null,
                            tint = if (isReady) NeonReadyGreen else SoftAmberStatus,
                            modifier = Modifier.size(18.dp)
                        )
                        Spacer(modifier = Modifier.width(5.dp))
                        Text(
                            text = if (isReady) {
                                "Ready ($engineShort)"
                            } else {
                                "${metrics.coreInstalledCount}/8 Slots (Tap)"
                            },
                            color = primaryTextColor,
                            style = MaterialTheme.typography.bodyMedium.copy(fontWeight = FontWeight.SemiBold)
                        )
                    }
                }

                SettingsDivider(darkTheme)

                SettingsRowItem(
                    icon = Icons.Filled.Storage,
                    iconTint = Color(0xFF2196F3),
                    label = "Model Size",
                    textColor = primaryTextColor
                ) {
                    Text(
                        text = OnnxModelManager.formatBytes(metrics.totalModelBytes),
                        color = primaryTextColor,
                        style = MaterialTheme.typography.bodyLarge.copy(fontWeight = FontWeight.Bold)
                    )
                }

                SettingsDivider(darkTheme)

                SettingsDropdownRow(
                    icon = Icons.Filled.PhotoSizeSelectLarge,
                    iconTint = Color(0xFF3F51B5),
                    label = "Resolution",
                    selectedValue = preferences.resolution,
                    options = listOf("384 × 384", "512 × 512", "448 × 640", "512 × 768", "640 × 640", "768 × 768"),
                    textColor = primaryTextColor,
                    darkTheme = darkTheme,
                    testTag = "dropdown_resolution",
                    onSelect = onResolutionChange
                )

                SettingsDivider(darkTheme)

                SettingsDropdownRow(
                    icon = Icons.Filled.Bolt,
                    iconTint = Color(0xFFFF9800),
                    label = "Inference Mode",
                    selectedValue = preferences.inferenceMode,
                    options = listOf("CPU (Auto)", "NNAPI", "Sequential Low-RAM"),
                    textColor = primaryTextColor,
                    darkTheme = darkTheme,
                    testTag = "dropdown_inference_mode",
                    onSelect = onInferenceModeChange
                )

                SettingsDivider(darkTheme)

                SettingsRowItem(
                    icon = Icons.Filled.WifiOff,
                    iconTint = Color(0xFFE91E63),
                    label = "Offline Mode",
                    textColor = primaryTextColor
                ) {
                    GreenGlassSwitch(
                        checked = preferences.offlineMode,
                        onCheckedChange = onOfflineModeChange,
                        testTag = "switch_offline_mode"
                    )
                }
            }

            // 1B. ONNX Model Download Links Card (Hugging Face Direct & Repo Links)
            OnnxDownloadLinksCard(
                isTamil = isTamil,
                darkTheme = darkTheme
            )

            // 2. Content Safety Settings Card -> Opens Dedicated Content Safety Settings Page!
            FrostedGlassCard(darkTheme = darkTheme) {
                SectionHeaderRow(
                    icon = Icons.Filled.Tune,
                    iconTint = VibrantPinkPill,
                    title = "Content Safety Settings",
                    textColor = primaryTextColor
                )

                SettingsDivider(darkTheme)

                SettingsRowItem(
                    icon = Icons.Filled.Security,
                    iconTint = PurpleActionEnd,
                    label = if (isTamil) "Apply Safety Prompt" else "Content Safety Prompt",
                    textColor = primaryTextColor
                ) {
                    GreenGlassSwitch(
                        checked = preferences.contentFilterEnabled,
                        onCheckedChange = onContentFilterToggle,
                        testTag = "switch_content_filter"
                    )
                }

                SettingsDivider(darkTheme)

                // Dedicated Page Entry Row (Edit / Save / Reset Safety Prompt)
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    modifier = Modifier
                        .fillMaxWidth()
                        .minimumInteractiveComponentSize()
                        .clip(RoundedCornerShape(14.dp))
                        .background(
                            if (darkTheme) Color.White.copy(alpha = 0.08f) else Color.White.copy(alpha = 0.55f)
                        )
                        .clickable(onClick = onOpenContentSafetyPage)
                        .padding(horizontal = 12.dp, vertical = 10.dp)
                        .testTag("open_content_safety_page_row")
                ) {
                    Icon(
                        imageVector = Icons.Filled.EditNote,
                        contentDescription = null,
                        tint = VibrantPinkPill,
                        modifier = Modifier.size(24.dp)
                    )
                    Spacer(modifier = Modifier.width(10.dp))
                    Column(modifier = Modifier.weight(1f)) {
                        Text(
                            text = if (isTamil) {
                                "Safety Prompt Editor (Edit / Save / Reset)"
                            } else {
                                "Edit Content Safety Prompt"
                            },
                            color = primaryTextColor,
                            style = MaterialTheme.typography.bodyLarge.copy(fontWeight = FontWeight.Bold)
                        )
                        Text(
                            text = preferences.savedContentSafetyPrompt,
                            color = secondaryTextColor,
                            style = MaterialTheme.typography.labelSmall,
                            maxLines = 2,
                            overflow = TextOverflow.Ellipsis
                        )
                    }
                    Spacer(modifier = Modifier.width(6.dp))
                    Icon(
                        imageVector = Icons.AutoMirrored.Filled.KeyboardArrowRight,
                        contentDescription = "Open Content Safety Settings Page",
                        tint = primaryTextColor
                    )
                }
            }

            // 2B. Editable Tamil → English dictionary
            TamilDictionaryCard(
                darkTheme = darkTheme,
                entries = preferences.tamilDictionary ?: OfflineTamilPromptTranslator.defaultPhraseMap,
                onUpsert = onUpsertTamilWord,
                onDelete = onDeleteTamilWord,
                onReset = onResetTamilDictionary
            )

            // 3. Storage & Cache Card
            FrostedGlassCard(darkTheme = darkTheme) {
                SectionHeaderRow(
                    icon = Icons.Filled.Storage,
                    iconTint = Color(0xFF1976D2),
                    title = "Storage & Cache",
                    textColor = primaryTextColor
                )

                SettingsDivider(darkTheme)

                val targetPipelineBytes = 5L * 1024L * 1024L * 1024L
                val progress = (metrics.totalModelBytes.toFloat() / targetPipelineBytes.toFloat())
                    .coerceIn(0f, 1f)

                Column(modifier = Modifier.fillMaxWidth()) {
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Text(
                            text = "Model Storage",
                            color = primaryTextColor,
                            style = MaterialTheme.typography.bodyLarge,
                            modifier = Modifier.weight(1f)
                        )
                        Text(
                            text = "${OnnxModelManager.formatBytes(metrics.totalModelBytes)} • Free: ${OnnxModelManager.formatBytes(metrics.deviceFreeBytes)}",
                            color = primaryTextColor,
                            style = MaterialTheme.typography.labelLarge.copy(fontWeight = FontWeight.Bold)
                        )
                    }
                    Spacer(modifier = Modifier.height(6.dp))
                    LinearProgressIndicator(
                        progress = { progress },
                        color = NeonReadyGreen,
                        trackColor = Color.White.copy(alpha = 0.45f),
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(7.dp)
                            .clip(RoundedCornerShape(50))
                    )
                }

                SettingsDivider(darkTheme)

                SettingsRowItem(
                    icon = Icons.Filled.FolderOpen,
                    iconTint = PurpleActionEnd,
                    label = if (isTamil) "Models (Upload / Delete)" else "Model Management",
                    textColor = primaryTextColor
                ) {
                    PinkActionPillButton(
                        text = "Manage",
                        testTag = "manage_models_button",
                        onClick = onOpenModelsManager
                    )
                }

                SettingsDivider(darkTheme)

                SettingsRowItem(
                    icon = Icons.Filled.DeleteSweep,
                    iconTint = Color(0xFFE53935),
                    label = "Clear Cache (${OnnxModelManager.formatBytes(metrics.cacheBytes)})",
                    textColor = primaryTextColor
                ) {
                    PinkActionPillButton(
                        text = "Clear",
                        testTag = "clear_cache_button",
                        onClick = onClearCache
                    )
                }
            }

            // 4. App Preferences Card
            FrostedGlassCard(darkTheme = darkTheme) {
                SectionHeaderRow(
                    icon = Icons.Filled.Language,
                    iconTint = Color(0xFF0288D1),
                    title = "App Preferences",
                    textColor = primaryTextColor
                )

                SettingsDivider(darkTheme)

                SettingsDropdownRow(
                    icon = Icons.Filled.Language,
                    iconTint = PurpleActionEnd,
                    label = if (isTamil) "Language (மொழி)" else "Language",
                    selectedValue = preferences.language.displayLabel,
                    options = AppLanguage.entries.map { it.displayLabel },
                    textColor = primaryTextColor,
                    darkTheme = darkTheme,
                    testTag = "dropdown_language",
                    onSelect = { label ->
                        val chosen = AppLanguage.entries.firstOrNull { it.displayLabel == label }
                            ?: AppLanguage.ENGLISH
                        onLanguageChange(chosen)
                    }
                )

                SettingsDivider(darkTheme)

                SettingsRowItem(
                    icon = Icons.Filled.Palette,
                    iconTint = Color(0xFF5E35B1),
                    label = "Theme",
                    textColor = primaryTextColor
                ) {
                    Row(
                        modifier = Modifier
                            .clip(RoundedCornerShape(50))
                            .background(Color.White.copy(alpha = if (darkTheme) 0.14f else 0.6f))
                            .border(1.dp, Color.White.copy(alpha = 0.7f), RoundedCornerShape(50))
                            .padding(2.dp)
                    ) {
                        ThemeSegmentChip(
                            label = "Light",
                            selected = !preferences.darkTheme,
                            darkTheme = darkTheme,
                            testTag = "theme_light_chip",
                            onClick = { onDarkThemeChange(false) }
                        )
                        ThemeSegmentChip(
                            label = "Dark",
                            selected = preferences.darkTheme,
                            darkTheme = darkTheme,
                            testTag = "theme_dark_chip",
                            onClick = { onDarkThemeChange(true) }
                        )
                    }
                }

                SettingsDivider(darkTheme)

                SettingsRowItem(
                    icon = Icons.Filled.Notifications,
                    iconTint = Color(0xFFFFA000),
                    label = "Notifications",
                    textColor = primaryTextColor
                ) {
                    GreenGlassSwitch(
                        checked = preferences.notificationsEnabled,
                        onCheckedChange = { wantEnabled ->
                            if (wantEnabled && Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                                notificationPermissionLauncher.launch(Manifest.permission.POST_NOTIFICATIONS)
                            } else {
                                onNotificationsChange(wantEnabled)
                            }
                        },
                        testTag = "switch_notifications"
                    )
                }
            }

            // 5. About & Help Card
            FrostedGlassCard(darkTheme = darkTheme) {
                SectionHeaderRow(
                    icon = Icons.Filled.Info,
                    iconTint = Color(0xFF1976D2),
                    title = "About & Help",
                    textColor = primaryTextColor
                )

                SettingsDivider(darkTheme)

                SettingsRowItem(
                    icon = Icons.Filled.Info,
                    iconTint = Color(0xFF3949AB),
                    label = "App Version",
                    textColor = primaryTextColor
                ) {
                    Text(
                        text = "V ${BuildConfig.VERSION_NAME}(${BuildConfig.VERSION_CODE})",
                        color = primaryTextColor,
                        style = MaterialTheme.typography.bodyLarge.copy(fontWeight = FontWeight.SemiBold)
                    )
                }

                SettingsDivider(darkTheme)

                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    modifier = Modifier
                        .fillMaxWidth()
                        .minimumInteractiveComponentSize()
                        .clickable { showHelpDialog = true }
                        .padding(vertical = 6.dp)
                        .testTag("help_support_row")
                ) {
                    Icon(
                        imageVector = Icons.AutoMirrored.Filled.HelpOutline,
                        contentDescription = null,
                        tint = PurpleActionEnd,
                        modifier = Modifier.size(22.dp)
                    )
                    Spacer(modifier = Modifier.width(10.dp))
                    Text(
                        text = if (isTamil) "Help & Support (உதவி)" else "Help & Support",
                        color = primaryTextColor,
                        style = MaterialTheme.typography.bodyLarge,
                        modifier = Modifier.weight(1f)
                    )
                    Icon(
                        imageVector = Icons.AutoMirrored.Filled.KeyboardArrowRight,
                        contentDescription = null,
                        tint = primaryTextColor
                    )
                }

                SettingsDivider(darkTheme)

                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    modifier = Modifier
                        .fillMaxWidth()
                        .minimumInteractiveComponentSize()
                        .clickable { showPrivacyDialog = true }
                        .padding(vertical = 6.dp)
                        .testTag("privacy_policy_row")
                ) {
                    Icon(
                        imageVector = Icons.Filled.Security,
                        contentDescription = null,
                        tint = Color(0xFFFFA000),
                        modifier = Modifier.size(22.dp)
                    )
                    Spacer(modifier = Modifier.width(10.dp))
                    Text(
                        text = "Privacy Policy",
                        color = primaryTextColor,
                        style = MaterialTheme.typography.bodyLarge,
                        modifier = Modifier.weight(1f)
                    )
                    Icon(
                        imageVector = Icons.AutoMirrored.Filled.KeyboardArrowRight,
                        contentDescription = null,
                        tint = primaryTextColor
                    )
                }
            }

            // Bottom Action Pill: Unload AI from RAM
            val bottomBtnShape = RoundedCornerShape(28.dp)
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.Center,
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 18.dp, vertical = 6.dp)
                    .minimumInteractiveComponentSize()
                    .shadow(10.dp, bottomBtnShape, spotColor = PurpleActionEnd)
                    .clip(bottomBtnShape)
                    .background(Brush.horizontalGradient(listOf(PurpleActionStart, PurpleActionEnd)))
                    .border(1.4.dp, Color.White.copy(alpha = 0.85f), bottomBtnShape)
                    .clickable(onClick = onUnloadRam)
                    .padding(vertical = 13.dp, horizontal = 18.dp)
                    .testTag("unload_ram_button")
            ) {
                Icon(
                    imageVector = Icons.Filled.CleaningServices,
                    contentDescription = null,
                    tint = Color.White,
                    modifier = Modifier.size(20.dp)
                )
                Spacer(modifier = Modifier.width(8.dp))
                Text(
                    text = if (isTamil) {
                        "Free Up RAM • (${OnnxModelManager.formatBytes(metrics.deviceAvailableRamBytes)} காலி)"
                    } else {
                        "Unload AI from RAM (${OnnxModelManager.formatBytes(metrics.deviceAvailableRamBytes)} Free)"
                    },
                    color = Color.White,
                    style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold)
                )
            }

            Spacer(modifier = Modifier.height(12.dp))
        }
    }

    if (showHelpDialog) {
        AlertDialog(
            onDismissRequest = { showHelpDialog = false },
            title = { Text(if (isTamil) "One-Touch ONNX மாடல் உதவி" else "Offline ONNX Setup Guide") },
            text = {
                Text(
                    if (isTamil) {
                        "உங்கள் மொபைலில் `Android/media/com.aistudio.imagegenerator.nvmkai/IG_Models/` ஃபோல்டரில் 8 Core ஸ்லாட் மாடல்கள் (1A–6) சேமிக்கப்படும்:\n\n" +
                            "1A & 1B: Fast LCM UNet (model.onnx + model_int8.onnx.data [~860 MB])\n" +
                            "2A & 2B: Standard SD UNet (model.onnx + weights.pb [~3.44 GB])\n" +
                            "3: Text Encoder CLIP (model.onnx)\n" +
                            "4: VAE Decoder (model.onnx)\n" +
                            "5: VAE Encoder (model.onnx)\n" +
                            "6: Tokenizer (vocab.json & merges.txt)\n\n" +
                            "Models பக்கத்தில் 'Auto-Download Core Slots (1A–6)' அல்லது 'Folder Auto-Import' மூலம் எளிதாக இணைக்கலாம்."
                    } else {
                        "Core ONNX models (Slots 1A–6) are stored directly inside `Android/media/com.aistudio.imagegenerator.nvmkai/IG_Models/`:\n\n" +
                            "1A & 1B: Fast LCM UNet (model.onnx + model_int8.onnx.data [~860 MB])\n" +
                            "2A & 2B: Standard SD UNet (model.onnx + weights.pb [~3.44 GB])\n" +
                            "3: Text Encoder CLIP (model.onnx)\n" +
                            "4: VAE Decoder (model.onnx)\n" +
                            "5: VAE Encoder (model.onnx)\n" +
                            "6: Tokenizer (vocab.json & merges.txt)\n\n" +
                            "Use 'Auto-Download Core Slots (1A–6)' or 'Folder Auto-Import' in the Models tab."
                    }
                )
            },
            confirmButton = {
                TextButton(onClick = { showHelpDialog = false }) { Text("OK") }
            }
        )
    }

    if (showPrivacyDialog) {
        AlertDialog(
            onDismissRequest = { showPrivacyDialog = false },
            title = { Text("Privacy Policy") },
            text = {
                Text(
                    if (isTamil) {
                        "இந்த Image Generator ஆப் 100% உங்கள் மொபைலிலேயே (Offline) இயங்கும் வகையில் உருவாக்கப்பட்டுள்ளது. உங்கள் Prompts, புகைப்படங்கள் மற்றும் அமைப்புகள் எதுவும் எந்த வெளி சர்வருக்கும் அனுப்பப்படாது."
                    } else {
                        "Image Generator runs 100% locally on your Android device. Your prompts, reference photos, generated creations, and preferences remain stored privately on your device with zero cloud uploads."
                    }
                )
            },
            confirmButton = {
                TextButton(onClick = { showPrivacyDialog = false }) { Text("OK") }
            }
        )
    }
}

@Composable
private fun SectionHeaderRow(
    icon: ImageVector,
    iconTint: Color,
    title: String,
    textColor: Color
) {
    Row(
        verticalAlignment = Alignment.CenterVertically,
        modifier = Modifier.fillMaxWidth()
    ) {
        Icon(
            imageVector = icon,
            contentDescription = null,
            tint = iconTint,
            modifier = Modifier.size(22.dp)
        )
        Spacer(modifier = Modifier.width(8.dp))
        Text(
            text = title,
            color = textColor,
            style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold),
            modifier = Modifier.weight(1f)
        )
        Icon(
            imageVector = Icons.Filled.Favorite,
            contentDescription = null,
            tint = Color.White.copy(alpha = 0.65f),
            modifier = Modifier.size(16.dp)
        )
    }
}

@Composable
private fun SettingsDivider(darkTheme: Boolean) {
    HorizontalDivider(
        modifier = Modifier.padding(vertical = 8.dp),
        thickness = 0.8.dp,
        color = if (darkTheme) Color.White.copy(alpha = 0.14f) else Color.White.copy(alpha = 0.55f)
    )
}

@Composable
private fun SettingsRowItem(
    icon: ImageVector,
    iconTint: Color,
    label: String,
    textColor: Color,
    trailing: @Composable () -> Unit
) {
    Row(
        verticalAlignment = Alignment.CenterVertically,
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 2.dp)
    ) {
        Icon(
            imageVector = icon,
            contentDescription = null,
            tint = iconTint,
            modifier = Modifier.size(21.dp)
        )
        Spacer(modifier = Modifier.width(10.dp))
        Text(
            text = label,
            color = textColor,
            style = MaterialTheme.typography.bodyLarge,
            modifier = Modifier.weight(1f)
        )
        trailing()
    }
}

@Composable
private fun SettingsDropdownRow(
    icon: ImageVector,
    iconTint: Color,
    label: String,
    selectedValue: String,
    options: List<String>,
    textColor: Color,
    darkTheme: Boolean,
    testTag: String,
    onSelect: (String) -> Unit
) {
    var expanded by remember { mutableStateOf(false) }

    SettingsRowItem(
        icon = icon,
        iconTint = iconTint,
        label = label,
        textColor = textColor
    ) {
        Box {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                modifier = Modifier
                    .minimumInteractiveComponentSize()
                    .clip(RoundedCornerShape(50))
                    .background(Color.White.copy(alpha = if (darkTheme) 0.16f else 0.72f))
                    .border(1.dp, Color.White.copy(alpha = 0.85f), RoundedCornerShape(50))
                    .clickable { expanded = true }
                    .padding(horizontal = 12.dp, vertical = 6.dp)
                    .testTag(testTag)
            ) {
                Text(
                    text = selectedValue,
                    color = textColor,
                    style = MaterialTheme.typography.labelLarge.copy(fontWeight = FontWeight.Bold)
                )
                Spacer(modifier = Modifier.width(4.dp))
                Icon(
                    imageVector = Icons.Filled.KeyboardArrowDown,
                    contentDescription = null,
                    tint = textColor,
                    modifier = Modifier.size(18.dp)
                )
            }
            DropdownMenu(
                expanded = expanded,
                onDismissRequest = { expanded = false }
            ) {
                options.forEach { option ->
                    DropdownMenuItem(
                        text = { Text(option) },
                        onClick = {
                            onSelect(option)
                            expanded = false
                        }
                    )
                }
            }
        }
    }
}

@Composable
private fun GreenGlassSwitch(
    checked: Boolean,
    onCheckedChange: (Boolean) -> Unit,
    testTag: String
) {
    Switch(
        checked = checked,
        onCheckedChange = onCheckedChange,
        colors = SwitchDefaults.colors(
            checkedThumbColor = Color.White,
            checkedTrackColor = NeonReadyGreen,
            uncheckedThumbColor = Color.White,
            uncheckedTrackColor = Color(0x669E9E9E)
        ),
        modifier = Modifier.testTag(testTag)
    )
}

@Composable
private fun PinkActionPillButton(
    text: String,
    testTag: String,
    onClick: () -> Unit
) {
    val shape = RoundedCornerShape(50)
    Box(
        contentAlignment = Alignment.Center,
        modifier = Modifier
            .minimumInteractiveComponentSize()
            .clip(shape)
            .background(Brush.horizontalGradient(listOf(PinkGlowStart, VibrantPinkPill)))
            .border(1.dp, Color.White.copy(alpha = 0.85f), shape)
            .clickable(onClick = onClick)
            .padding(horizontal = 18.dp, vertical = 6.dp)
            .testTag(testTag)
    ) {
        Text(
            text = text,
            color = Color.White,
            style = MaterialTheme.typography.labelLarge.copy(fontWeight = FontWeight.Bold)
        )
    }
}

@Composable
private fun ThemeSegmentChip(
    label: String,
    selected: Boolean,
    darkTheme: Boolean,
    testTag: String,
    onClick: () -> Unit
) {
    val shape = RoundedCornerShape(50)
    Box(
        contentAlignment = Alignment.Center,
        modifier = Modifier
            .clip(shape)
            .then(
                if (selected) {
                    Modifier.background(Brush.horizontalGradient(listOf(PinkGlowStart, VibrantPinkPill)))
                } else {
                    Modifier
                }
            )
            .clickable(onClick = onClick)
            .padding(horizontal = 14.dp, vertical = 5.dp)
            .testTag(testTag)
    ) {
        Text(
            text = label,
            color = if (selected) Color.White else if (darkTheme) Color.White.copy(alpha = 0.8f) else DeepPlumText,
            style = MaterialTheme.typography.labelLarge.copy(fontWeight = FontWeight.SemiBold)
        )
    }
}
