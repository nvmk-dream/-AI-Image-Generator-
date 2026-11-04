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
import androidx.compose.material.icons.filled.Edit
import androidx.compose.material.icons.filled.RestartAlt
import androidx.compose.material.icons.filled.Save
import androidx.compose.material.icons.filled.Security
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Switch
import androidx.compose.material3.SwitchDefaults
import androidx.compose.material3.Text
import androidx.compose.material3.minimumInteractiveComponentSize
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import com.example.data.AppPreferencesState
import com.example.data.ContentSafetyDefaults
import com.example.data.ContentSafetyMode
import com.example.ui.components.CircularGlowIconButton
import com.example.ui.components.FrostedGlassCard
import com.example.ui.theme.DeepPlumText
import com.example.ui.theme.NeonReadyGreen
import com.example.ui.theme.PinkGlowEnd
import com.example.ui.theme.PinkGlowStart
import com.example.ui.theme.PurpleActionEnd
import com.example.ui.theme.PurpleActionStart
import com.example.ui.theme.VibrantPinkPill

@Composable
fun ContentSafetySettingsScreen(
    preferences: AppPreferencesState,
    currentUserPromptDraft: String,
    onBack: () -> Unit,
    onToggleContentFilter: (Boolean) -> Unit,
    onSelectModePreset: (ContentSafetyMode, String) -> Unit,
    onSaveContentSafetyPrompt: (String) -> Unit,
    onResetContentSafetyPrompt: () -> Unit,
    modifier: Modifier = Modifier
) {
    BackHandler { onBack() }

    val isTamil = preferences.isTamil
    val darkTheme = preferences.darkTheme
    val primaryColor = if (darkTheme) Color.White else DeepPlumText
    val subColor = if (darkTheme) Color(0xFFE1C9F7) else Color(0xFF4F376E)

    var editorText by rememberSaveable { mutableStateOf(preferences.savedContentSafetyPrompt) }
    var isEditing by rememberSaveable { mutableStateOf(false) }

    LaunchedEffect(preferences.savedContentSafetyPrompt) {
        if (!isEditing) {
            editorText = preferences.savedContentSafetyPrompt
        }
    }

    val sampleUserPrompt = currentUserPromptDraft.ifBlank {
        if (isTamil) "அழகான பட்டுப்புடவை அணிந்த பெண், இயற்கை ஒளி" else "Portrait of woman in silk saree, soft bokeh"
    }
    val previewCombined = if (!preferences.contentFilterEnabled || editorText.isBlank()) {
        sampleUserPrompt
    } else {
        "$sampleUserPrompt, ${editorText.trim()}"
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
                    contentDescription = "Back to Settings",
                    testTag = "content_safety_back_button"
                ) {
                    Icon(
                        imageVector = Icons.AutoMirrored.Filled.ArrowBack,
                        contentDescription = "Back",
                        tint = Color.White
                    )
                }
                Text(
                    text = if (isTamil) "Content Safety Settings" else "Content Safety Settings",
                    color = primaryColor,
                    style = MaterialTheme.typography.headlineMedium.copy(fontWeight = FontWeight.ExtraBold),
                    textAlign = TextAlign.Center,
                    modifier = Modifier
                        .weight(1f)
                        .padding(horizontal = 10.dp)
                )
                Spacer(modifier = Modifier.size(48.dp))
            }

            // Master Switch + Mode Presets Card
            FrostedGlassCard(darkTheme = darkTheme) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Icon(
                        imageVector = Icons.Filled.Security,
                        contentDescription = null,
                        tint = VibrantPinkPill,
                        modifier = Modifier.size(22.dp)
                    )
                    Spacer(modifier = Modifier.width(10.dp))
                    Column(modifier = Modifier.weight(1f)) {
                        Text(
                            text = if (isTamil) "Apply Content Safety Prompt" else "Apply Content Safety Prompt",
                            color = primaryColor,
                            style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold)
                        )
                        Text(
                            text = if (isTamil) {
                                "User Prompt உடன் Safety Prompt-ஐ இணைத்து அனுப்பும்"
                            } else {
                                "Combines Saved Content Safety Prompt with User Prompt"
                            },
                            color = subColor,
                            style = MaterialTheme.typography.labelSmall
                        )
                    }
                    Switch(
                        checked = preferences.contentFilterEnabled,
                        onCheckedChange = onToggleContentFilter,
                        colors = SwitchDefaults.colors(
                            checkedThumbColor = Color.White,
                            checkedTrackColor = NeonReadyGreen
                        ),
                        modifier = Modifier.testTag("safety_page_master_switch")
                    )
                }

                Spacer(modifier = Modifier.height(10.dp))

                // Quick Preset Chips (Strict | Balanced | Adult | Custom)
                Row(
                    horizontalArrangement = Arrangement.spacedBy(6.dp),
                    modifier = Modifier.fillMaxWidth()
                ) {
                    val modes = listOf(
                        Triple(
                            ContentSafetyMode.STRICT,
                            "Strict",
                            ContentSafetyDefaults.STRICT_TEMPLATE_PROMPT
                        ),
                        Triple(
                            ContentSafetyMode.BALANCED,
                            "Balanced",
                            ContentSafetyDefaults.BALANCED_TEMPLATE_PROMPT
                        ),
                        Triple(
                            ContentSafetyMode.ADULT,
                            "Adult",
                            ContentSafetyDefaults.ADULT_OPEN_TEMPLATE_PROMPT
                        ),
                        Triple(
                            ContentSafetyMode.CUSTOM,
                            "Custom",
                            editorText
                        )
                    )
                    modes.forEach { (mode, label, templateText) ->
                        val selected = preferences.safetyMode == mode
                        val shape = RoundedCornerShape(14.dp)
                        Box(
                            contentAlignment = Alignment.Center,
                            modifier = Modifier
                                .weight(1f)
                                .minimumInteractiveComponentSize()
                                .clip(shape)
                                .background(
                                    if (selected) {
                                        Brush.horizontalGradient(listOf(PinkGlowStart, PinkGlowEnd))
                                    } else {
                                        Brush.horizontalGradient(
                                            listOf(
                                                Color.White.copy(alpha = if (darkTheme) 0.14f else 0.6f),
                                                Color.White.copy(alpha = if (darkTheme) 0.08f else 0.5f)
                                            )
                                        )
                                    }
                                )
                                .border(1.dp, Color.White.copy(alpha = 0.7f), shape)
                                .clickable {
                                    if (mode == ContentSafetyMode.CUSTOM) {
                                        isEditing = true
                                        onSelectModePreset(mode, editorText)
                                    } else {
                                        editorText = templateText
                                        isEditing = false
                                        onSelectModePreset(mode, templateText)
                                    }
                                }
                                .padding(vertical = 8.dp)
                                .testTag("safety_preset_${mode.key}")
                        ) {
                            Text(
                                text = label,
                                color = if (selected) Color.White else primaryColor,
                                style = MaterialTheme.typography.labelLarge.copy(fontWeight = FontWeight.Bold)
                            )
                        }
                    }
                }
            }

            // Editable Content Safety Prompt Card (Edit / Save / Reset)
            FrostedGlassCard(darkTheme = darkTheme) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text(
                        text = if (isTamil) {
                            "Content Safety Prompt (மாற்றக்கூடியது)"
                        } else {
                            "Content Safety Prompt Editor"
                        },
                        color = primaryColor,
                        style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold),
                        modifier = Modifier.weight(1f)
                    )
                    Text(
                        text = if (isEditing) "Editing" else "Saved",
                        color = if (isEditing) VibrantPinkPill else NeonReadyGreen,
                        style = MaterialTheme.typography.labelLarge.copy(fontWeight = FontWeight.Bold)
                    )
                }

                Spacer(modifier = Modifier.height(6.dp))
                Text(
                    text = if (isTamil) {
                        "இது User Image Prompt-ஐ மாற்றாமல் தனியாகச் சேமிக்கப்படும்."
                    } else {
                        "Stored separately on-device. Never overwrites your User Image Prompt."
                    },
                    color = subColor,
                    style = MaterialTheme.typography.bodyMedium
                )

                Spacer(modifier = Modifier.height(10.dp))

                OutlinedTextField(
                    value = editorText,
                    onValueChange = {
                        editorText = it
                        isEditing = true
                    },
                    enabled = isEditing,
                    minLines = 4,
                    maxLines = 20,
                    modifier = Modifier
                        .fillMaxWidth()
                        .testTag("content_safety_prompt_editor")
                )

                Spacer(modifier = Modifier.height(12.dp))

                // Edit | Save | Reset Action Row
                Row(
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                    modifier = Modifier.fillMaxWidth()
                ) {
                    // 1. Edit Button
                    SafetyActionButton(
                        label = if (isTamil) "Edit" else "Edit",
                        icon = Icons.Filled.Edit,
                        gradient = listOf(Color(0xFF5C6BC0), Color(0xFF3949AB)),
                        testTag = "safety_prompt_edit_button",
                        onClick = { isEditing = true },
                        modifier = Modifier.weight(1f)
                    )

                    // 2. Save Button
                    SafetyActionButton(
                        label = if (isTamil) "Save" else "Save",
                        icon = Icons.Filled.Save,
                        gradient = listOf(PinkGlowStart, PinkGlowEnd),
                        testTag = "safety_prompt_save_button",
                        onClick = {
                            onSaveContentSafetyPrompt(editorText)
                            isEditing = false
                        },
                        modifier = Modifier.weight(1f)
                    )

                    // 3. Reset Button
                    SafetyActionButton(
                        label = if (isTamil) "Reset" else "Reset",
                        icon = Icons.Filled.RestartAlt,
                        gradient = listOf(PurpleActionStart, PurpleActionEnd),
                        testTag = "safety_prompt_reset_button",
                        onClick = {
                            editorText = ContentSafetyDefaults.DEFAULT_CONTENT_SAFETY_PROMPT
                            isEditing = false
                            onResetContentSafetyPrompt()
                        },
                        modifier = Modifier.weight(1f)
                    )
                }
            }

            // Live Final Generation Input Preview Card
            FrostedGlassCard(darkTheme = darkTheme) {
                Text(
                    text = if (isTamil) {
                        "Final Generation Input முன்னோட்டம்"
                    } else {
                        "Final Generation Input Preview"
                    },
                    color = primaryColor,
                    style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold)
                )
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    text = "Final Generation Input = User Image Prompt + Saved Content Safety Prompt",
                    color = VibrantPinkPill,
                    style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.Bold)
                )
                Spacer(modifier = Modifier.height(8.dp))
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clip(RoundedCornerShape(12.dp))
                        .background(
                            if (darkTheme) Color.White.copy(alpha = 0.08f) else Color.White.copy(alpha = 0.58f)
                        )
                        .padding(10.dp)
                ) {
                    Text(
                        text = previewCombined,
                        color = subColor,
                        style = MaterialTheme.typography.bodyMedium
                    )
                }
            }

            Spacer(modifier = Modifier.height(14.dp))
        }
    }
}

@Composable
private fun SafetyActionButton(
    label: String,
    icon: androidx.compose.ui.graphics.vector.ImageVector,
    gradient: List<Color>,
    testTag: String,
    onClick: () -> Unit,
    modifier: Modifier = Modifier
) {
    val shape = RoundedCornerShape(16.dp)
    Row(
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.Center,
        modifier = modifier
            .minimumInteractiveComponentSize()
            .clip(shape)
            .background(Brush.horizontalGradient(gradient))
            .border(1.dp, Color.White.copy(alpha = 0.85f), shape)
            .clickable(onClick = onClick)
            .padding(vertical = 10.dp, horizontal = 8.dp)
            .testTag(testTag)
    ) {
        Icon(
            imageVector = icon,
            contentDescription = null,
            tint = Color.White,
            modifier = Modifier.size(17.dp)
        )
        Spacer(modifier = Modifier.width(5.dp))
        Text(
            text = label,
            color = Color.White,
            style = MaterialTheme.typography.labelLarge.copy(fontWeight = FontWeight.Bold)
        )
    }
}
