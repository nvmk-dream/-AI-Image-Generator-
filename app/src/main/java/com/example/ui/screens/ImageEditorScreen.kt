package com.example.ui.screens

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
import androidx.compose.foundation.layout.aspectRatio
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
import androidx.compose.material.icons.filled.AddPhotoAlternate
import androidx.compose.material.icons.filled.AutoAwesome
import androidx.compose.material.icons.filled.AutoFixHigh
import androidx.compose.material.icons.filled.Download
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Visibility
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.material3.minimumInteractiveComponentSize
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.ColorFilter
import androidx.compose.ui.graphics.ColorMatrix
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import coil.compose.AsyncImage
import com.example.ai.EditorFilterPreset
import com.example.ai.ImageEditorAdjustments
import com.example.ai.OfflineImageEditorEngine
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
import com.example.ui.theme.VibrantPinkPill
import java.io.File
import kotlin.math.roundToInt

@Composable
fun ImageEditorScreen(
    preferences: AppPreferencesState,
    editorImageUri: Uri?,
    latestCreation: GeneratedCreation?,
    useLatestCreationInEditor: Boolean,
    latestEditedCreation: GeneratedCreation?,
    isSavingEdit: Boolean,
    onPickEditorPhoto: () -> Unit,
    onLoadLatestCreation: () -> Unit,
    onSaveEditedPhoto: (adjustments: ImageEditorAdjustments, presetLabel: String) -> Unit,
    onExportToGallery: (GeneratedCreation) -> Unit,
    onBack: () -> Unit,
    modifier: Modifier = Modifier
) {
    BackHandler { onBack() }

    val isTamil = preferences.isTamil
    val darkTheme = preferences.darkTheme
    val textColor = if (darkTheme) Color.White else DeepPlumText
    val subColor = if (darkTheme) Color(0xFFE1C9F7) else Color(0xFF4F376E)

    var selectedPreset by remember { mutableStateOf(EditorFilterPreset.ORIGINAL) }
    var brightness by remember { mutableFloatStateOf(0f) }
    var contrast by remember { mutableFloatStateOf(1.0f) }
    var saturation by remember { mutableFloatStateOf(1.0f) }
    var warmth by remember { mutableFloatStateOf(0f) }
    var showOriginal by remember { mutableStateOf(false) }

    val adjustments = ImageEditorAdjustments(
        brightness = brightness,
        contrast = contrast,
        saturation = saturation,
        warmth = warmth
    )
    val liveMatrixValues = remember(brightness, contrast, saturation, warmth) {
        OfflineImageEditorEngine.buildColorMatrixValues(adjustments)
    }

    val activeImageModel: Any? = when {
        useLatestCreationInEditor && latestCreation != null && File(latestCreation.filePath).exists() ->
            File(latestCreation.filePath)
        editorImageUri != null -> editorImageUri
        latestCreation != null && File(latestCreation.filePath).exists() ->
            File(latestCreation.filePath)
        else -> null
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
            // Top Bar
            Row(
                verticalAlignment = Alignment.CenterVertically,
                modifier = Modifier.fillMaxWidth()
            ) {
                CircularGlowIconButton(
                    onClick = onBack,
                    contentDescription = "Back",
                    testTag = "editor_back_button"
                ) {
                    Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "Back", tint = Color.White)
                }
                Spacer(modifier = Modifier.width(12.dp))
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        text = "Image Editor",
                        color = textColor,
                        style = MaterialTheme.typography.headlineMedium.copy(fontWeight = FontWeight.ExtraBold)
                    )
                    Text(
                        text = if (isTamil) "உடனடி நிறம் & ஒளி திருத்தம் (100% Offline)" else "Real-Time Color & Glow Studio (Offline)",
                        color = subColor,
                        style = MaterialTheme.typography.bodyMedium
                    )
                }
            }

            // 1. Live Preview Canvas Card
            FrostedGlassCard(darkTheme = darkTheme) {
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
                            .border(1.dp, Color.White.copy(alpha = 0.85f), RoundedCornerShape(14.dp))
                            .clickable(onClick = onPickEditorPhoto)
                            .padding(vertical = 10.dp, horizontal = 8.dp)
                            .testTag("editor_pick_photo_button")
                    ) {
                        Icon(Icons.Filled.AddPhotoAlternate, contentDescription = null, tint = Color.White, modifier = Modifier.size(18.dp))
                        Spacer(modifier = Modifier.width(6.dp))
                        Text(
                            text = if (isTamil) "படம் தேர்வு" else "Choose Photo",
                            color = Color.White,
                            style = MaterialTheme.typography.labelLarge.copy(fontWeight = FontWeight.Bold)
                        )
                    }

                    if (latestCreation != null && File(latestCreation.filePath).exists()) {
                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.Center,
                            modifier = Modifier
                                .weight(1f)
                                .minimumInteractiveComponentSize()
                                .clip(RoundedCornerShape(14.dp))
                                .background(Brush.horizontalGradient(listOf(PinkGlowStart, PinkGlowEnd)))
                                .border(1.dp, Color.White.copy(alpha = 0.85f), RoundedCornerShape(14.dp))
                                .clickable(onClick = onLoadLatestCreation)
                                .padding(vertical = 10.dp, horizontal = 8.dp)
                                .testTag("editor_load_latest_button")
                        ) {
                            Icon(Icons.Filled.AutoAwesome, contentDescription = null, tint = Color.White, modifier = Modifier.size(18.dp))
                            Spacer(modifier = Modifier.width(6.dp))
                            Text(
                                text = if (isTamil) "கடைசி AI படம்" else "Latest AI Photo",
                                color = Color.White,
                                style = MaterialTheme.typography.labelLarge.copy(fontWeight = FontWeight.Bold)
                            )
                        }
                    }
                }

                Spacer(modifier = Modifier.height(10.dp))

                if (activeImageModel != null) {
                    Box(modifier = Modifier.fillMaxWidth()) {
                        AsyncImage(
                            model = activeImageModel,
                            contentDescription = "Edited Image Preview",
                            contentScale = ContentScale.Crop,
                            colorFilter = if (showOriginal) null else ColorFilter.colorMatrix(ColorMatrix(liveMatrixValues)),
                            modifier = Modifier
                                .fillMaxWidth()
                                .aspectRatio(1f)
                                .clip(RoundedCornerShape(18.dp))
                                .border(1.4.dp, Color.White, RoundedCornerShape(18.dp))
                        )

                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            modifier = Modifier
                                .align(Alignment.TopEnd)
                                .padding(10.dp)
                                .minimumInteractiveComponentSize()
                                .clip(RoundedCornerShape(50))
                                .background(Color.Black.copy(alpha = 0.58f))
                                .border(1.dp, Color.White.copy(alpha = 0.8f), RoundedCornerShape(50))
                                .clickable { showOriginal = !showOriginal }
                                .padding(horizontal = 12.dp, vertical = 6.dp)
                                .testTag("editor_compare_toggle")
                        ) {
                            Icon(Icons.Filled.Visibility, contentDescription = null, tint = Color.White, modifier = Modifier.size(16.dp))
                            Spacer(modifier = Modifier.width(4.dp))
                            Text(
                                text = if (showOriginal) "Original" else "Edited",
                                color = Color.White,
                                style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.Bold)
                            )
                        }
                    }
                } else {
                    Box(
                        contentAlignment = Alignment.Center,
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(170.dp)
                            .clip(RoundedCornerShape(16.dp))
                            .background(Color.White.copy(alpha = if (darkTheme) 0.08f else 0.48f))
                            .border(1.dp, Color.White.copy(alpha = 0.7f), RoundedCornerShape(16.dp))
                            .clickable(onClick = onPickEditorPhoto)
                            .padding(16.dp)
                    ) {
                        Text(
                            text = if (isTamil) {
                                "திருத்த வேண்டிய புகைப்படத்தை Gallery-யில் இருந்து தேர்வு செய்யவும்"
                            } else {
                                "Tap to select any photo from Gallery for instant offline editing"
                            },
                            color = subColor,
                            style = MaterialTheme.typography.bodyMedium,
                            textAlign = TextAlign.Center
                        )
                    }
                }
            }

            // 2. Presets + Fine Color Sliders Card
            FrostedGlassCard(darkTheme = darkTheme) {
                Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.fillMaxWidth()) {
                    Text(
                        text = if (isTamil) "நிற ஃபில்டர்கள் (Presets)" else "One-Tap Filter Presets",
                        color = textColor,
                        style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold),
                        modifier = Modifier.weight(1f)
                    )
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        modifier = Modifier
                            .minimumInteractiveComponentSize()
                            .clickable {
                                selectedPreset = EditorFilterPreset.ORIGINAL
                                brightness = 0f
                                contrast = 1.0f
                                saturation = 1.0f
                                warmth = 0f
                            }
                            .padding(4.dp)
                            .testTag("editor_reset_button")
                    ) {
                        Icon(Icons.Filled.Refresh, contentDescription = "Reset", tint = VibrantPinkPill, modifier = Modifier.size(16.dp))
                        Spacer(modifier = Modifier.width(4.dp))
                        Text("Reset", color = VibrantPinkPill, style = MaterialTheme.typography.labelLarge.copy(fontWeight = FontWeight.Bold))
                    }
                }

                Spacer(modifier = Modifier.height(6.dp))

                Row(
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                    modifier = Modifier
                        .fillMaxWidth()
                        .horizontalScroll(rememberScrollState())
                ) {
                    EditorFilterPreset.entries.forEach { preset ->
                        val selected = selectedPreset == preset
                        Box(
                            modifier = Modifier
                                .minimumInteractiveComponentSize()
                                .clip(RoundedCornerShape(50))
                                .background(
                                    if (selected) Brush.horizontalGradient(listOf(PinkGlowStart, PinkGlowEnd))
                                    else Brush.horizontalGradient(listOf(Color.White.copy(alpha = 0.55f), Color.White.copy(alpha = 0.42f)))
                                )
                                .border(1.dp, Color.White.copy(alpha = 0.85f), RoundedCornerShape(50))
                                .clickable {
                                    selectedPreset = preset
                                    brightness = preset.brightness
                                    contrast = preset.contrast
                                    saturation = preset.saturation
                                    warmth = preset.warmth
                                }
                                .padding(horizontal = 14.dp, vertical = 8.dp)
                                .testTag("editor_preset_${preset.name.lowercase()}")
                        ) {
                            Text(
                                text = if (isTamil) preset.labelTa else preset.labelEn,
                                color = if (selected) Color.White else textColor,
                                style = MaterialTheme.typography.labelLarge.copy(fontWeight = FontWeight.Bold)
                            )
                        }
                    }
                }

                Spacer(modifier = Modifier.height(10.dp))

                // Brightness
                EditorSliderRow(
                    label = if (isTamil) "வெளிச்சம் (Brightness)" else "Brightness",
                    valueText = "${brightness.roundToInt()}",
                    value = brightness,
                    range = -50f..50f,
                    testTag = "editor_brightness_slider",
                    textColor = textColor,
                    onValueChange = { brightness = it }
                )

                // Contrast
                EditorSliderRow(
                    label = if (isTamil) "துல்லியம் (Contrast)" else "Contrast",
                    valueText = "${(contrast * 100).roundToInt()}%",
                    value = contrast,
                    range = 0.50f..1.50f,
                    testTag = "editor_contrast_slider",
                    textColor = textColor,
                    onValueChange = { contrast = it }
                )

                // Saturation
                EditorSliderRow(
                    label = if (isTamil) "நிறச் செறிவு (Saturation)" else "Saturation",
                    valueText = "${(saturation * 100).roundToInt()}%",
                    value = saturation,
                    range = 0.0f..2.0f,
                    testTag = "editor_saturation_slider",
                    textColor = textColor,
                    onValueChange = { saturation = it }
                )

                // Warmth
                EditorSliderRow(
                    label = if (isTamil) "வெப்ப நிறம் (Warmth)" else "Warmth",
                    valueText = "${warmth.roundToInt()}",
                    value = warmth,
                    range = -50f..50f,
                    testTag = "editor_warmth_slider",
                    textColor = textColor,
                    onValueChange = { warmth = it }
                )

                Spacer(modifier = Modifier.height(8.dp))

                val btnShape = RoundedCornerShape(22.dp)
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.Center,
                    modifier = Modifier
                        .fillMaxWidth()
                        .minimumInteractiveComponentSize()
                        .shadow(10.dp, btnShape, spotColor = PurpleActionEnd)
                        .clip(btnShape)
                        .background(Brush.horizontalGradient(listOf(PinkGlowStart, PinkGlowEnd)))
                        .border(1.3.dp, Color.White, btnShape)
                        .clickable(enabled = activeImageModel != null && !isSavingEdit) {
                            onSaveEditedPhoto(adjustments, selectedPreset.labelEn)
                        }
                        .padding(vertical = 13.dp)
                        .testTag("editor_save_button")
                ) {
                    Icon(Icons.Filled.AutoFixHigh, contentDescription = null, tint = Color.White)
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(
                        text = if (isSavingEdit) {
                            "Saving..."
                        } else if (isTamil) {
                            "திருத்திய படத்தைச் சேமி (Save Edited Image)"
                        } else {
                            "Save Edited Photo"
                        },
                        color = Color.White,
                        style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold)
                    )
                }

                if (latestEditedCreation != null && File(latestEditedCreation.filePath).exists()) {
                    Spacer(modifier = Modifier.height(10.dp))
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.Center,
                        modifier = Modifier
                            .fillMaxWidth()
                            .minimumInteractiveComponentSize()
                            .clip(RoundedCornerShape(16.dp))
                            .background(Brush.horizontalGradient(listOf(PurpleActionStart, PurpleActionEnd)))
                            .border(1.dp, Color.White, RoundedCornerShape(16.dp))
                            .clickable { onExportToGallery(latestEditedCreation) }
                            .padding(vertical = 10.dp)
                            .testTag("editor_export_gallery_button")
                    ) {
                        Icon(Icons.Filled.Download, contentDescription = null, tint = Color.White, modifier = Modifier.size(18.dp))
                        Spacer(modifier = Modifier.width(6.dp))
                        Text(
                            text = if (isTamil) "Phone Gallery-ல் சேமி ✓" else "Export Saved Edit to Phone Gallery ✓",
                            color = NeonReadyGreen,
                            style = MaterialTheme.typography.labelLarge.copy(fontWeight = FontWeight.Bold)
                        )
                    }
                }
            }

            Spacer(modifier = Modifier.height(16.dp))
        }
    }
}

@Composable
private fun EditorSliderRow(
    label: String,
    valueText: String,
    value: Float,
    range: ClosedFloatingPointRange<Float>,
    testTag: String,
    textColor: Color,
    onValueChange: (Float) -> Unit
) {
    Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.fillMaxWidth()) {
        Text(
            text = label,
            color = textColor,
            style = MaterialTheme.typography.labelLarge.copy(fontWeight = FontWeight.SemiBold),
            modifier = Modifier.weight(1f)
        )
        Text(
            text = valueText,
            color = VibrantPinkPill,
            style = MaterialTheme.typography.labelLarge.copy(fontWeight = FontWeight.Bold)
        )
    }
    Slider(
        value = value,
        onValueChange = onValueChange,
        valueRange = range,
        modifier = Modifier.testTag(testTag)
    )
}
