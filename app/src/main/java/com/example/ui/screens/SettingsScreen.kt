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
import androidx.compose.foundation.shape.CircleShape
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
import androidx.compose.material3.Checkbox
import androidx.compose.material3.CheckboxDefaults
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
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.BuildConfig
import com.example.ai.OnnxModelManager
import com.example.ai.RealDeviceAndModelMetrics
import com.example.data.AppLanguage
import com.example.data.AppPreferencesState
import com.example.data.ContentSafetyMode
import com.example.ui.components.CircularGlowIconButton
import com.example.ui.components.FrostedGlassCard
import com.example.ui.components.PaletteAppLogo
import com.example.ui.theme.CyanVersionAccent
import com.example.ui.theme.DeepPlumText
import com.example.ui.theme.NeonReadyGreen
import com.example.ui.theme.PinkGlowEnd
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
    onResolutionChange: (String) -> Unit,
    onInferenceModeChange: (String) -> Unit,
    onOfflineModeChange: (Boolean) -> Unit,
    onLanguageChange: (AppLanguage) -> Unit,
    onDarkThemeChange: (Boolean) -> Unit,
    onNotificationsChange: (Boolean) -> Unit,
    onContentFilterToggle: (Boolean) -> Unit,
    onSafetyModeChange: (ContentSafetyMode) -> Unit,
    onCustomCategoryChange: (
        fashionGlamour: Boolean?,
        swimwearBeach: Boolean?,
        romanticPoses: Boolean?,
        classicalArtistic: Boolean?,
        fantasyAction: Boolean?
    ) -> Unit,
    onClearCache: () -> Unit,
    onUnloadRam: () -> Unit,
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
                                text = if (isTamil) {
                                    "Offline AI • Create • Edit • Enjoy"
                                } else {
                                    "Create • Edit • Enjoy"
                                },
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
                    title = if (isTamil) "AI Model Settings" else "AI Model Settings",
                    textColor = primaryTextColor
                )

                SettingsDivider(darkTheme)

                // Real Model Status
                val isReady = metrics.ortRuntimeAvailable && metrics.isTxt2ImgReady
                SettingsRowItem(
                    icon = Icons.Filled.CheckCircle,
                    iconTint = if (isReady) NeonReadyGreen else SoftAmberStatus,
                    label = if (isTamil) "Model Status" else "Model Status",
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
                                "Ready (ONNX)"
                            } else {
                                "${metrics.installedCount}/4 Models (Tap)"
                            },
                            color = primaryTextColor,
                            style = MaterialTheme.typography.bodyMedium.copy(fontWeight = FontWeight.SemiBold)
                        )
                    }
                }

                SettingsDivider(darkTheme)

                // Real Model Size
                SettingsRowItem(
                    icon = Icons.Filled.Storage,
                    iconTint = Color(0xFF2196F3),
                    label = if (isTamil) "Model Size" else "Model Size",
                    textColor = primaryTextColor
                ) {
                    Text(
                        text = OnnxModelManager.formatBytes(metrics.totalModelBytes),
                        color = primaryTextColor,
                        style = MaterialTheme.typography.bodyLarge.copy(fontWeight = FontWeight.Bold)
                    )
                }

                SettingsDivider(darkTheme)

                // Resolution Dropdown
                SettingsDropdownRow(
                    icon = Icons.Filled.PhotoSizeSelectLarge,
                    iconTint = Color(0xFF3F51B5),
                    label = if (isTamil) "Resolution" else "Resolution",
                    selectedValue = preferences.resolution,
                    options = listOf("512 × 512", "640 × 640", "768 × 768"),
                    textColor = primaryTextColor,
                    darkTheme = darkTheme,
                    testTag = "dropdown_resolution",
                    onSelect = onResolutionChange
                )

                SettingsDivider(darkTheme)

                // Inference Mode Dropdown
                SettingsDropdownRow(
                    icon = Icons.Filled.Bolt,
                    iconTint = Color(0xFFFF9800),
                    label = if (isTamil) "Inference Mode" else "Inference Mode",
                    selectedValue = preferences.inferenceMode,
                    options = listOf("CPU (Auto)", "NNAPI", "Sequential Low-RAM"),
                    textColor = primaryTextColor,
                    darkTheme = darkTheme,
                    testTag = "dropdown_inference_mode",
                    onSelect = onInferenceModeChange
                )

                SettingsDivider(darkTheme)

                // Offline Mode Switch
                SettingsRowItem(
                    icon = Icons.Filled.WifiOff,
                    iconTint = Color(0xFFE91E63),
                    label = if (isTamil) "Offline Mode" else "Offline Mode",
                    textColor = primaryTextColor
                ) {
                    GreenGlassSwitch(
                        checked = preferences.offlineMode,
                        onCheckedChange = onOfflineModeChange,
                        testTag = "switch_offline_mode"
                    )
                }
            }

            // 2. Content Safety Settings Card (User-configurable 3 modes, local & offline)
            FrostedGlassCard(darkTheme = darkTheme) {
                SectionHeaderRow(
                    icon = Icons.Filled.Tune,
                    iconTint = VibrantPinkPill,
                    title = if (isTamil) "Content Safety Settings" else "Content Safety Settings",
                    textColor = primaryTextColor
                )

                SettingsDivider(darkTheme)

                SettingsRowItem(
                    icon = Icons.Filled.Security,
                    iconTint = PurpleActionEnd,
                    label = if (isTamil) "Content Filter" else "Content Filter",
                    textColor = primaryTextColor
                ) {
                    GreenGlassSwitch(
                        checked = preferences.contentFilterEnabled,
                        onCheckedChange = onContentFilterToggle,
                        testTag = "switch_content_filter"
                    )
                }

                Spacer(modifier = Modifier.height(8.dp))

                // 3-Mode Selector Pills: Strict | Balanced | Custom
                Row(
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                    modifier = Modifier.fillMaxWidth()
                ) {
                    val modes = listOf(
                        ContentSafetyMode.STRICT to (if (isTamil) "Strict" else "Strict"),
                        ContentSafetyMode.BALANCED to (if (isTamil) "Balanced" else "Balanced"),
                        ContentSafetyMode.CUSTOM to (if (isTamil) "Custom" else "Custom")
                    )
                    modes.forEach { (mode, modeTitle) ->
                        val isSelected = preferences.safetyMode == mode
                        val shape = RoundedCornerShape(16.dp)
                        Box(
                            contentAlignment = Alignment.Center,
                            modifier = Modifier
                                .weight(1f)
                                .minimumInteractiveComponentSize()
                                .clip(shape)
                                .background(
                                    if (isSelected) {
                                        Brush.horizontalGradient(listOf(PinkGlowStart, PinkGlowEnd))
                                    } else {
                                        Brush.horizontalGradient(
                                            listOf(
                                                Color.White.copy(alpha = if (darkTheme) 0.12f else 0.55f),
                                                Color.White.copy(alpha = if (darkTheme) 0.08f else 0.45f)
                                            )
                                        )
                                    }
                                )
                                .border(
                                    width = 1.dp,
                                    color = if (isSelected) Color.White else Color.White.copy(alpha = 0.5f),
                                    shape = shape
                                )
                                .clickable { onSafetyModeChange(mode) }
                                .padding(vertical = 9.dp, horizontal = 6.dp)
                                .testTag("safety_mode_${mode.key}")
                        ) {
                            Text(
                                text = modeTitle,
                                color = if (isSelected) Color.White else primaryTextColor,
                                style = MaterialTheme.typography.labelLarge.copy(fontWeight = FontWeight.Bold)
                            )
                        }
                    }
                }

                Spacer(modifier = Modifier.height(8.dp))

                // Active Mode Description Box
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clip(RoundedCornerShape(12.dp))
                        .background(
                            if (darkTheme) Color.White.copy(alpha = 0.08f) else Color.White.copy(alpha = 0.52f)
                        )
                        .padding(horizontal = 12.dp, vertical = 8.dp)
                ) {
                    Text(
                        text = preferences.activeFilterSummary(),
                        color = secondaryTextColor,
                        style = MaterialTheme.typography.bodyMedium.copy(fontWeight = FontWeight.Medium)
                    )
                }

                // Custom Mode Individual Category Switches
                if (preferences.safetyMode == ContentSafetyMode.CUSTOM) {
                    Spacer(modifier = Modifier.height(8.dp))
                    CustomCategoryCheckRow(
                        label = if (isTamil) "Fashion & Glamour உடைகள்" else "Fashion & Glamour Outfits",
                        checked = preferences.customCategories.fashionGlamour,
                        textColor = primaryTextColor,
                        testTag = "custom_cat_fashion",
                        onCheckedChange = {
                            onCustomCategoryChange(it, null, null, null, null)
                        }
                    )
                    CustomCategoryCheckRow(
                        label = if (isTamil) "Swimwear / கடற்கரை Fashion" else "Swimwear & Beach Fashion",
                        checked = preferences.customCategories.swimwearBeach,
                        textColor = primaryTextColor,
                        testTag = "custom_cat_swimwear",
                        onCheckedChange = {
                            onCustomCategoryChange(null, it, null, null, null)
                        }
                    )
                    CustomCategoryCheckRow(
                        label = if (isTamil) "Romantic & Couple போஸ்கள்" else "Romantic & Couple Poses",
                        checked = preferences.customCategories.romanticPoses,
                        textColor = primaryTextColor,
                        testTag = "custom_cat_romantic",
                        onCheckedChange = {
                            onCustomCategoryChange(null, null, it, null, null)
                        }
                    )
                    CustomCategoryCheckRow(
                        label = if (isTamil) "Classical / கலைநய ஓவியப் பாணி" else "Classical & Artistic Style",
                        checked = preferences.customCategories.classicalArtistic,
                        textColor = primaryTextColor,
                        testTag = "custom_cat_artistic",
                        onCheckedChange = {
                            onCustomCategoryChange(null, null, null, it, null)
                        }
                    )
                    CustomCategoryCheckRow(
                        label = if (isTamil) "Fantasy Action காட்சிகள்" else "Fantasy Action Scenes",
                        checked = preferences.customCategories.fantasyAction,
                        textColor = primaryTextColor,
                        testTag = "custom_cat_action",
                        onCheckedChange = {
                            onCustomCategoryChange(null, null, null, null, it)
                        }
                    )
                }
            }

            // 3. Storage & Cache Card (Real Device Metrics)
            FrostedGlassCard(darkTheme = darkTheme) {
                SectionHeaderRow(
                    icon = Icons.Filled.Storage,
                    iconTint = Color(0xFF1976D2),
                    title = if (isTamil) "Storage & Cache" else "Storage & Cache",
                    textColor = primaryTextColor
                )

                SettingsDivider(darkTheme)

                // Model Storage Progress Row (real model bytes + real free storage)
                val targetPipelineBytes = 2L * 1024L * 1024L * 1024L // 2.0 GB reference bar
                val progress = (metrics.totalModelBytes.toFloat() / targetPipelineBytes.toFloat())
                    .coerceIn(0f, 1f)

                Column(modifier = Modifier.fillMaxWidth()) {
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Text(
                            text = if (isTamil) "Model Storage" else "Model Storage",
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

                // Model Download / Upload / Delete Management Row
                SettingsRowItem(
                    icon = Icons.Filled.FolderOpen,
                    iconTint = PurpleActionEnd,
                    label = if (isTamil) "Models (Upload / Delete)" else "Model Management",
                    textColor = primaryTextColor
                ) {
                    PinkActionPillButton(
                        text = if (isTamil) "Manage" else "Manage",
                        testTag = "manage_models_button",
                        onClick = onOpenModelsManager
                    )
                }

                SettingsDivider(darkTheme)

                // Clear Cache Row
                SettingsRowItem(
                    icon = Icons.Filled.DeleteSweep,
                    iconTint = Color(0xFFE53935),
                    label = "${if (isTamil) "Clear Cache" else "Clear Cache"} (${OnnxModelManager.formatBytes(metrics.cacheBytes)})",
                    textColor = primaryTextColor
                ) {
                    PinkActionPillButton(
                        text = if (isTamil) "Clear" else "Clear",
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
                    title = if (isTamil) "App Preferences" else "App Preferences",
                    textColor = primaryTextColor
                )

                SettingsDivider(darkTheme)

                // Language Dropdown (English / தமிழ்)
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

                // Theme Segmented Toggle (Light | Dark)
                SettingsRowItem(
                    icon = Icons.Filled.Palette,
                    iconTint = Color(0xFF5E35B1),
                    label = if (isTamil) "Theme" else "Theme",
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

                // Notifications Toggle
                SettingsRowItem(
                    icon = Icons.Filled.Notifications,
                    iconTint = Color(0xFFFFA000),
                    label = if (isTamil) "Notifications" else "Notifications",
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
                    title = if (isTamil) "About & Help" else "About & Help",
                    textColor = primaryTextColor
                )

                SettingsDivider(darkTheme)

                SettingsRowItem(
                    icon = Icons.Filled.Info,
                    iconTint = Color(0xFF3949AB),
                    label = if (isTamil) "App Version" else "App Version",
                    textColor = primaryTextColor
                ) {
                    Text(
                        text = "V ${BuildConfig.VERSION_NAME}(${BuildConfig.VERSION_CODE})",
                        color = primaryTextColor,
                        style = MaterialTheme.typography.bodyLarge.copy(fontWeight = FontWeight.SemiBold)
                    )
                }

                SettingsDivider(darkTheme)

                // Help & Support Row
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

                // Privacy Policy Row
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
                        text = if (isTamil) "Privacy Policy" else "Privacy Policy",
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

            // Bottom Action Pill: Unload AI from RAM (replaces sample Logout button with useful offline action)
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
            title = {
                Text(if (isTamil) "One-Touch ONNX மாடல் உதவி" else "Offline ONNX Setup Guide")
            },
            text = {
                Text(
                    if (isTamil) {
                        "உங்கள் மொபைலில் ஒரு Folder-ல் (உதாரணம்: Download/AI_Models) கீழ்கண்ட 4 .onnx ஃபைல்களைச் சேமிக்கவும்:\n\n" +
                            "1. text_encoder.onnx\n" +
                            "2. unet.onnx\n" +
                            "3. vae_decoder.onnx\n" +
                            "4. vae_encoder.onnx\n\n" +
                            "பின்னர் Models பக்கத்தில் 'One-Touch Folder Upload' பட்டனைத் தொட்டால் 4 மாடல்களும் தானாகவே ஏற்றப்படும்."
                    } else {
                        "Save the 4 ONNX model files inside one folder on your phone (e.g. Download/AI_Models):\n\n" +
                            "1. text_encoder.onnx\n" +
                            "2. unet.onnx\n" +
                            "3. vae_decoder.onnx\n" +
                            "4. vae_encoder.onnx\n\n" +
                            "Then open the Models tab and tap 'One-Touch Folder Upload' to auto-import all models at once."
                    }
                )
            },
            confirmButton = {
                TextButton(onClick = { showHelpDialog = false }) {
                    Text("OK")
                }
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
                TextButton(onClick = { showPrivacyDialog = false }) {
                    Text("OK")
                }
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
            color = if (selected) {
                Color.White
            } else if (darkTheme) {
                Color.White.copy(alpha = 0.8f)
            } else {
                DeepPlumText
            },
            style = MaterialTheme.typography.labelLarge.copy(fontWeight = FontWeight.SemiBold)
        )
    }
}

@Composable
private fun CustomCategoryCheckRow(
    label: String,
    checked: Boolean,
    textColor: Color,
    testTag: String,
    onCheckedChange: (Boolean) -> Unit
) {
    Row(
        verticalAlignment = Alignment.CenterVertically,
        modifier = Modifier
            .fillMaxWidth()
            .clickable { onCheckedChange(!checked) }
            .padding(vertical = 2.dp)
            .testTag(testTag)
    ) {
        Checkbox(
            checked = checked,
            onCheckedChange = onCheckedChange,
            colors = CheckboxDefaults.colors(
                checkedColor = VibrantPinkPill,
                checkmarkColor = Color.White
            )
        )
        Text(
            text = label,
            color = textColor,
            style = MaterialTheme.typography.bodyMedium.copy(fontWeight = FontWeight.Medium)
        )
    }
}
