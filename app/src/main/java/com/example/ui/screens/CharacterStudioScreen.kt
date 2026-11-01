package com.example.ui.screens

import android.net.Uri
import androidx.activity.compose.BackHandler
import androidx.compose.animation.AnimatedVisibility
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
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.AddPhotoAlternate
import androidx.compose.material.icons.filled.AutoAwesome
import androidx.compose.material.icons.filled.Cancel
import androidx.compose.material.icons.filled.DeleteOutline
import androidx.compose.material.icons.filled.Download
import androidx.compose.material.icons.filled.FolderSpecial
import androidx.compose.material.icons.filled.Person
import androidx.compose.material.icons.filled.Security
import androidx.compose.material.icons.filled.Shuffle
import androidx.compose.material.icons.filled.Star
import androidx.compose.material.icons.outlined.StarOutline
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.material3.minimumInteractiveComponentSize
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableLongStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import coil.compose.AsyncImage
import com.example.ai.GenerationProgress
import com.example.ai.RealDeviceAndModelMetrics
import com.example.ai.SamplerType
import com.example.data.AppPreferencesState
import com.example.data.CharacterProfile
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
import kotlin.random.Random

@Composable
fun CharacterStudioScreen(
    preferences: AppPreferencesState,
    metrics: RealDeviceAndModelMetrics,
    characters: List<CharacterProfile>,
    selectedCharacterId: String?,
    pendingCharacterPhotoUri: Uri?,
    isGenerating: Boolean,
    progress: GenerationProgress?,
    latestCharacterCreation: GeneratedCreation?,
    errorMessage: String?,
    onSelectCharacter: (String) -> Unit,
    onPickCharacterPhoto: () -> Unit,
    onClearCharacterPhoto: () -> Unit,
    onSaveCharacter: (name: String, appearancePrompt: String, fixedSeed: Long) -> Unit,
    onDeleteCharacter: (String) -> Unit,
    onGenerateCharacterScene: (
        character: CharacterProfile,
        scenePrompt: String,
        useReferenceImg2Img: Boolean,
        strength: Float,
        sampler: SamplerType,
        steps: Int,
        guidanceScale: Float,
        negativePrompt: String
    ) -> Unit,
    onCancelGeneration: () -> Unit,
    onExportToGallery: (GeneratedCreation) -> Unit,
    onToggleFavorite: (String) -> Unit,
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

    val activeChar = characters.firstOrNull { it.id == selectedCharacterId } ?: characters.firstOrNull()

    var showCreateForm by remember { mutableStateOf(false) }
    var newName by remember { mutableStateOf("") }
    var newAppearance by remember {
        mutableStateOf("graceful young Indian woman, expressive dark eyes, silky black hair, photorealistic face")
    }
    var newSeed by remember { mutableLongStateOf(Random.nextLong(100_000L, 999_999_999L)) }

    var scenePrompt by remember {
        mutableStateOf("wearing royal pink silk saree in golden temple courtyard, soft warm sunlight")
    }
    var useReferenceImg2Img by remember { mutableStateOf(false) }
    var strength by remember { mutableFloatStateOf(0.55f) }
    var selectedSampler by remember { mutableStateOf(SamplerType.LCM_FAST) }
    var steps by remember { mutableIntStateOf(6) }
    var guidanceScale by remember { mutableFloatStateOf(1.5f) }
    val negativePrompt = "blurry, deformed hands, distorted face, low quality, extra fingers"
    var showCombinedSafety by remember { mutableStateOf(false) }

    LaunchedEffect(activeChar?.id) {
        if (activeChar?.referencePhotoPath == null) {
            useReferenceImg2Img = false
        }
    }

    val combinedCharacterPrompt = activeChar?.buildCharacterScenePrompt(scenePrompt) ?: scenePrompt
    val finalSafetyCombined = preferences.buildFinalGenerationInput(combinedCharacterPrompt)

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
            // Top Header
            Row(
                verticalAlignment = Alignment.CenterVertically,
                modifier = Modifier.fillMaxWidth()
            ) {
                CircularGlowIconButton(
                    onClick = onBack,
                    contentDescription = "Back",
                    testTag = "char_studio_back_button"
                ) {
                    Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "Back", tint = Color.White)
                }
                Spacer(modifier = Modifier.width(12.dp))
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        text = "Character Studio",
                        color = textColor,
                        style = MaterialTheme.typography.headlineMedium.copy(fontWeight = FontWeight.ExtraBold)
                    )
                    Text(
                        text = if (isTamil) "ஒரே கதாபாத்திரம் • பல உடைகள் & காட்சிகள்" else "Consistent Character • Multi-Scene Offline",
                        color = subColor,
                        style = MaterialTheme.typography.bodyMedium
                    )
                }
                CircularGlowIconButton(
                    onClick = onOpenContentSafetySettings,
                    contentDescription = "Content Safety Settings",
                    testTag = "char_open_safety_button"
                ) {
                    Icon(Icons.Filled.Security, contentDescription = "Safety Prompt", tint = Color.White)
                }
            }

            // 1. Saved Character Profiles + Creator
            FrostedGlassCard(darkTheme = darkTheme) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text(
                        text = if (isTamil) "1. சேமித்த கதாபாத்திரங்கள்" else "1. Saved Characters",
                        color = textColor,
                        style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold),
                        modifier = Modifier.weight(1f)
                    )
                    Box(
                        modifier = Modifier
                            .minimumInteractiveComponentSize()
                            .clip(RoundedCornerShape(50))
                            .background(Brush.horizontalGradient(listOf(PurpleActionStart, PurpleActionEnd)))
                            .border(1.dp, Color.White, RoundedCornerShape(50))
                            .clickable(enabled = !isGenerating) { showCreateForm = !showCreateForm }
                            .padding(horizontal = 12.dp, vertical = 6.dp)
                            .testTag("toggle_new_character_button")
                    ) {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Icon(Icons.Filled.Add, contentDescription = null, tint = Color.White, modifier = Modifier.size(16.dp))
                            Spacer(modifier = Modifier.width(4.dp))
                            Text(
                                text = if (showCreateForm) (if (isTamil) "மூடு" else "Close") else (if (isTamil) "புதியது" else "New"),
                                color = Color.White,
                                style = MaterialTheme.typography.labelLarge.copy(fontWeight = FontWeight.Bold)
                            )
                        }
                    }
                }

                Spacer(modifier = Modifier.height(8.dp))

                Row(
                    horizontalArrangement = Arrangement.spacedBy(10.dp),
                    modifier = Modifier
                        .fillMaxWidth()
                        .horizontalScroll(rememberScrollState())
                ) {
                    characters.forEach { item ->
                        val isSelected = item.id == activeChar?.id
                        val cardShape = RoundedCornerShape(16.dp)
                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            modifier = Modifier
                                .clip(cardShape)
                                .background(
                                    if (isSelected) {
                                        Brush.horizontalGradient(listOf(PinkGlowStart, PinkGlowEnd))
                                    } else {
                                        Brush.horizontalGradient(
                                            listOf(
                                                Color.White.copy(alpha = if (darkTheme) 0.14f else 0.62f),
                                                Color.White.copy(alpha = if (darkTheme) 0.08f else 0.48f)
                                            )
                                        )
                                    }
                                )
                                .border(1.2.dp, Color.White.copy(alpha = 0.85f), cardShape)
                                .clickable(enabled = !isGenerating) { onSelectCharacter(item.id) }
                                .padding(horizontal = 12.dp, vertical = 8.dp)
                                .testTag("character_chip_${item.id}")
                        ) {
                            if (item.referencePhotoPath != null && File(item.referencePhotoPath).exists()) {
                                AsyncImage(
                                    model = File(item.referencePhotoPath),
                                    contentDescription = item.name,
                                    contentScale = ContentScale.Crop,
                                    modifier = Modifier
                                        .size(38.dp)
                                        .clip(CircleShape)
                                        .border(1.dp, Color.White, CircleShape)
                                )
                            } else {
                                Box(
                                    contentAlignment = Alignment.Center,
                                    modifier = Modifier
                                        .size(38.dp)
                                        .clip(CircleShape)
                                        .background(PurpleActionEnd)
                                ) {
                                    Icon(Icons.Filled.Person, contentDescription = null, tint = Color.White)
                                }
                            }
                            Spacer(modifier = Modifier.width(8.dp))
                            Column {
                                Text(
                                    text = item.name,
                                    color = if (isSelected) Color.White else textColor,
                                    style = MaterialTheme.typography.labelLarge.copy(fontWeight = FontWeight.Bold),
                                    maxLines = 1,
                                    overflow = TextOverflow.Ellipsis
                                )
                                Text(
                                    text = "Seed ${item.fixedSeed}",
                                    color = if (isSelected) Color.White.copy(alpha = 0.9f) else subColor,
                                    style = MaterialTheme.typography.labelSmall
                                )
                            }
                            if (characters.size > 1) {
                                IconButton(
                                    onClick = { onDeleteCharacter(item.id) },
                                    modifier = Modifier.size(36.dp)
                                ) {
                                    Icon(
                                        imageVector = Icons.Filled.DeleteOutline,
                                        contentDescription = "Delete ${item.name}",
                                        tint = if (isSelected) Color.White else Color(0xFFD32F2F),
                                        modifier = Modifier.size(18.dp)
                                    )
                                }
                            }
                        }
                    }
                }

                AnimatedVisibility(visible = showCreateForm) {
                    Column(modifier = Modifier.padding(top = 12.dp)) {
                        OutlinedTextField(
                            value = newName,
                            onValueChange = { newName = it },
                            label = { Text(if (isTamil) "கதாபாத்திரப் பெயர்" else "Character Name") },
                            singleLine = true,
                            modifier = Modifier
                                .fillMaxWidth()
                                .testTag("char_name_input")
                        )
                        Spacer(modifier = Modifier.height(8.dp))
                        OutlinedTextField(
                            value = newAppearance,
                            onValueChange = { newAppearance = it },
                            label = { Text(if (isTamil) "முக அமைப்பு & தோற்றம் (Appearance)" else "Consistent Face & Hair Traits") },
                            minLines = 2,
                            maxLines = 4,
                            modifier = Modifier
                                .fillMaxWidth()
                                .testTag("char_appearance_input")
                        )
                        Spacer(modifier = Modifier.height(8.dp))
                        Row(
                            verticalAlignment = Alignment.CenterVertically,
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
                                    .clickable(onClick = onPickCharacterPhoto)
                                    .padding(vertical = 10.dp, horizontal = 8.dp)
                                    .testTag("char_pick_ref_photo_button")
                            ) {
                                Icon(Icons.Filled.AddPhotoAlternate, contentDescription = null, tint = Color.White, modifier = Modifier.size(18.dp))
                                Spacer(modifier = Modifier.width(6.dp))
                                Text(
                                    text = if (pendingCharacterPhotoUri != null) "Photo Attached ✓" else (if (isTamil) "முகம் சேர் (விருப்பம்)" else "Ref Photo (Optional)"),
                                    color = Color.White,
                                    style = MaterialTheme.typography.labelLarge.copy(fontWeight = FontWeight.Bold)
                                )
                            }
                            if (pendingCharacterPhotoUri != null) {
                                IconButton(onClick = onClearCharacterPhoto) {
                                    Icon(Icons.Filled.Cancel, contentDescription = "Clear photo", tint = VibrantPinkPill)
                                }
                            }
                            IconButton(onClick = { newSeed = Random.nextLong(100_000L, 999_999_999L) }) {
                                Icon(Icons.Filled.Shuffle, contentDescription = "New Seed", tint = PurpleActionEnd)
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
                                .background(Brush.horizontalGradient(listOf(PinkGlowStart, PinkGlowEnd)))
                                .border(1.dp, Color.White, RoundedCornerShape(16.dp))
                                .clickable {
                                    onSaveCharacter(newName, newAppearance, newSeed)
                                    newName = ""
                                    showCreateForm = false
                                }
                                .padding(vertical = 11.dp)
                                .testTag("char_save_profile_button")
                        ) {
                            Text(
                                text = if (isTamil) "கதாபாத்திரத்தைச் சேமி (Seed $newSeed)" else "Save Character Profile (Seed $newSeed)",
                                color = Color.White,
                                style = MaterialTheme.typography.labelLarge.copy(fontWeight = FontWeight.Bold)
                            )
                        }
                    }
                }
            }

            // 2. Scene & Outfit Prompt Card
            if (activeChar != null) {
                FrostedGlassCard(darkTheme = darkTheme) {
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Text(
                            text = if (isTamil) "2. காட்சி & உடை: ${activeChar.name}" else "2. Scene & Outfit: ${activeChar.name}",
                            color = textColor,
                            style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold),
                            modifier = Modifier.weight(1f)
                        )
                        Text(
                            text = "Safety Prompt ✓",
                            color = VibrantPinkPill,
                            style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.Bold),
                            modifier = Modifier.clickable { showCombinedSafety = !showCombinedSafety }
                        )
                    }

                    Text(
                        text = activeChar.appearancePrompt,
                        color = subColor,
                        style = MaterialTheme.typography.labelSmall,
                        maxLines = 2,
                        overflow = TextOverflow.Ellipsis
                    )

                    if (activeChar.referencePhotoPath != null) {
                        Spacer(modifier = Modifier.height(8.dp))
                        Row(
                            horizontalArrangement = Arrangement.spacedBy(8.dp),
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            val modes = listOf(
                                false to (if (isTamil) "Seed Lock (Txt2Img)" else "Seed-Locked Txt2Img"),
                                true to (if (isTamil) "Ref Photo (Img2Img)" else "Ref Photo Img2Img")
                            )
                            modes.forEach { (isRefMode, label) ->
                                val active = useReferenceImg2Img == isRefMode
                                Box(
                                    contentAlignment = Alignment.Center,
                                    modifier = Modifier
                                        .weight(1f)
                                        .minimumInteractiveComponentSize()
                                        .clip(RoundedCornerShape(12.dp))
                                        .background(
                                            if (active) Brush.horizontalGradient(listOf(PinkGlowStart, PinkGlowEnd))
                                            else Brush.horizontalGradient(listOf(Color.White.copy(alpha = 0.45f), Color.White.copy(alpha = 0.35f)))
                                        )
                                        .border(1.dp, Color.White.copy(alpha = 0.8f), RoundedCornerShape(12.dp))
                                        .clickable(enabled = !isGenerating) { useReferenceImg2Img = isRefMode }
                                        .padding(vertical = 8.dp)
                                ) {
                                    Text(
                                        text = label,
                                        color = if (active) Color.White else textColor,
                                        style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.Bold)
                                    )
                                }
                            }
                        }
                    }

                    Spacer(modifier = Modifier.height(8.dp))

                    val scenePresets = listOf(
                        "பட்டுப்புடவை அணிந்து கோயில் பிரகாரத்தில், இயற்கை ஒளி" to "Silk Saree Temple",
                        "wearing modern pastel evening gown on beach at golden sunset, bokeh" to "Beach Sunset",
                        "in royal palace balcony with festive lamps, warm cinematic portrait" to "Palace Lights"
                    )
                    Row(
                        horizontalArrangement = Arrangement.spacedBy(8.dp),
                        modifier = Modifier
                            .fillMaxWidth()
                            .horizontalScroll(rememberScrollState())
                    ) {
                        scenePresets.forEachIndexed { idx, (presetText, chipTitle) ->
                            Box(
                                modifier = Modifier
                                    .clip(RoundedCornerShape(50))
                                    .background(Color.White.copy(alpha = if (darkTheme) 0.14f else 0.65f))
                                    .border(1.dp, Color.White.copy(alpha = 0.75f), RoundedCornerShape(50))
                                    .clickable(enabled = !isGenerating) { scenePrompt = presetText }
                                    .padding(horizontal = 12.dp, vertical = 6.dp)
                                    .testTag("char_scene_chip_$idx")
                            ) {
                                Text(chipTitle, color = textColor, style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.Bold))
                            }
                        }
                    }

                    Spacer(modifier = Modifier.height(8.dp))

                    OutlinedTextField(
                        value = scenePrompt,
                        onValueChange = { scenePrompt = it },
                        enabled = !isGenerating,
                        label = { Text(if (isTamil) "புதிய உடை / இடம் (Scene Prompt)" else "New Outfit / Scene Description") },
                        minLines = 2,
                        maxLines = 4,
                        modifier = Modifier
                            .fillMaxWidth()
                            .testTag("char_scene_prompt_input")
                    )

                    if (showCombinedSafety) {
                        Spacer(modifier = Modifier.height(6.dp))
                        Text(
                            text = "Final Input: $finalSafetyCombined",
                            color = subColor,
                            style = MaterialTheme.typography.labelSmall
                        )
                    }

                    if (useReferenceImg2Img) {
                        Spacer(modifier = Modifier.height(8.dp))
                        Text(
                            text = "Reference Strength: ${(strength * 100).roundToInt()}%",
                            color = textColor,
                            style = MaterialTheme.typography.labelLarge.copy(fontWeight = FontWeight.Bold)
                        )
                        Slider(
                            value = strength,
                            onValueChange = { strength = (it * 20f).roundToInt() / 20f },
                            enabled = !isGenerating,
                            valueRange = 0.30f..0.80f
                        )
                    }

                    Spacer(modifier = Modifier.height(8.dp))

                    Row(
                        horizontalArrangement = Arrangement.spacedBy(8.dp),
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        SamplerType.entries.forEach { sampler ->
                            val selected = selectedSampler == sampler
                            Box(
                                contentAlignment = Alignment.Center,
                                modifier = Modifier
                                    .weight(1f)
                                    .minimumInteractiveComponentSize()
                                    .clip(RoundedCornerShape(14.dp))
                                    .background(
                                        if (selected) Brush.horizontalGradient(listOf(PinkGlowStart, PinkGlowEnd))
                                        else Brush.horizontalGradient(listOf(Color.White.copy(alpha = 0.45f), Color.White.copy(alpha = 0.35f)))
                                    )
                                    .border(1.dp, Color.White.copy(alpha = 0.7f), RoundedCornerShape(14.dp))
                                    .clickable(enabled = !isGenerating) {
                                        selectedSampler = sampler
                                        steps = if (sampler == SamplerType.LCM_FAST) 6 else 16
                                        guidanceScale = sampler.defaultCfg
                                    }
                                    .padding(vertical = 8.dp)
                            ) {
                                Text(
                                    text = if (sampler == SamplerType.LCM_FAST) "⚡ Fast LCM ($steps)" else "🎨 Standard SD ($steps)",
                                    color = if (selected) Color.White else textColor,
                                    style = MaterialTheme.typography.labelLarge.copy(fontWeight = FontWeight.Bold)
                                )
                            }
                        }
                    }

                    Spacer(modifier = Modifier.height(10.dp))

                    val ready = if (useReferenceImg2Img) metrics.isAllFourReady else metrics.isTxt2ImgReady
                    val btnShape = RoundedCornerShape(22.dp)

                    if (isGenerating) {
                        val pct = (progress?.percent ?: 5).coerceIn(1, 100)
                        Text(
                            text = "${progress?.stageTitle ?: "Generating Character..."} ($pct%)",
                            color = textColor,
                            style = MaterialTheme.typography.bodyMedium.copy(fontWeight = FontWeight.SemiBold)
                        )
                        Spacer(modifier = Modifier.height(6.dp))
                        LinearProgressIndicator(
                            progress = { pct / 100f },
                            color = VibrantPinkPill,
                            modifier = Modifier.fillMaxWidth().height(8.dp).clip(RoundedCornerShape(50))
                        )
                        Spacer(modifier = Modifier.height(8.dp))
                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.Center,
                            modifier = Modifier
                                .fillMaxWidth()
                                .minimumInteractiveComponentSize()
                                .clip(btnShape)
                                .background(Color(0xFFD32F2F))
                                .clickable(onClick = onCancelGeneration)
                                .padding(vertical = 12.dp)
                                .testTag("char_cancel_button")
                        ) {
                            Icon(Icons.Filled.Cancel, contentDescription = null, tint = Color.White)
                            Spacer(modifier = Modifier.width(6.dp))
                            Text("Cancel", color = Color.White, style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold))
                        }
                    } else {
                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.Center,
                            modifier = Modifier
                                .fillMaxWidth()
                                .minimumInteractiveComponentSize()
                                .shadow(10.dp, btnShape, spotColor = PurpleActionEnd)
                                .clip(btnShape)
                                .background(
                                    Brush.horizontalGradient(
                                        if (ready) listOf(PinkGlowStart, PinkGlowEnd)
                                        else listOf(PurpleActionStart, PurpleActionEnd)
                                    )
                                )
                                .border(1.3.dp, Color.White, btnShape)
                                .clickable {
                                    if (!ready) {
                                        onOpenModelsTab()
                                    } else {
                                        onGenerateCharacterScene(
                                            activeChar,
                                            scenePrompt,
                                            useReferenceImg2Img,
                                            strength,
                                            selectedSampler,
                                            steps,
                                            guidanceScale,
                                            negativePrompt
                                        )
                                    }
                                }
                                .padding(vertical = 13.dp, horizontal = 14.dp)
                                .testTag("char_generate_scene_button")
                        ) {
                            Icon(
                                imageVector = if (ready) Icons.Filled.AutoAwesome else Icons.Filled.FolderSpecial,
                                contentDescription = null,
                                tint = Color.White
                            )
                            Spacer(modifier = Modifier.width(8.dp))
                            Text(
                                text = if (ready) {
                                    if (isTamil) "${activeChar.name} படம் உருவாக்கு" else "Generate ${activeChar.name} Scene"
                                } else {
                                    if (isTamil) "ONNX மாடல்களை ஏற்றவும் (${metrics.installedCount}/4)" else "Upload ONNX Models (${metrics.installedCount}/4)"
                                },
                                color = Color.White,
                                style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold)
                            )
                        }
                    }

                    if (!errorMessage.isNullOrBlank()) {
                        Spacer(modifier = Modifier.height(8.dp))
                        Text(errorMessage, color = Color(0xFFD32F2F), style = MaterialTheme.typography.bodyMedium)
                    }
                }
            }

            // 3. Output Card
            if (latestCharacterCreation != null && File(latestCharacterCreation.filePath).exists()) {
                FrostedGlassCard(darkTheme = darkTheme) {
                    Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.fillMaxWidth()) {
                        Text(
                            text = if (isTamil) "கதாபாத்திரப் படைப்பு" else "Character Scene Result",
                            color = textColor,
                            style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold),
                            modifier = Modifier.weight(1f)
                        )
                        Text(
                            text = "Seed ${latestCharacterCreation.seed}",
                            color = NeonReadyGreen,
                            style = MaterialTheme.typography.labelLarge.copy(fontWeight = FontWeight.Bold)
                        )
                    }
                    Spacer(modifier = Modifier.height(8.dp))
                    AsyncImage(
                        model = File(latestCharacterCreation.filePath),
                        contentDescription = latestCharacterCreation.userPrompt,
                        contentScale = ContentScale.Crop,
                        modifier = Modifier
                            .fillMaxWidth()
                            .aspectRatio(1f)
                            .clip(RoundedCornerShape(18.dp))
                            .border(1.4.dp, Color.White, RoundedCornerShape(18.dp))
                    )
                    Spacer(modifier = Modifier.height(10.dp))
                    Row(horizontalArrangement = Arrangement.spacedBy(10.dp), modifier = Modifier.fillMaxWidth()) {
                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.Center,
                            modifier = Modifier
                                .weight(1f)
                                .minimumInteractiveComponentSize()
                                .clip(RoundedCornerShape(16.dp))
                                .background(Brush.horizontalGradient(listOf(PurpleActionStart, PurpleActionEnd)))
                                .border(1.dp, Color.White, RoundedCornerShape(16.dp))
                                .clickable { onExportToGallery(latestCharacterCreation) }
                                .padding(vertical = 10.dp)
                                .testTag("char_save_gallery_button")
                        ) {
                            Icon(Icons.Filled.Download, contentDescription = null, tint = Color.White, modifier = Modifier.size(18.dp))
                            Spacer(modifier = Modifier.width(6.dp))
                            Text(
                                text = if (isTamil) "Gallery-ல் சேமி" else "Save to Gallery",
                                color = Color.White,
                                style = MaterialTheme.typography.labelLarge.copy(fontWeight = FontWeight.Bold)
                            )
                        }
                        IconButton(
                            onClick = { onToggleFavorite(latestCharacterCreation.id) },
                            modifier = Modifier.testTag("char_favorite_button")
                        ) {
                            Icon(
                                imageVector = if (latestCharacterCreation.isFavorite) Icons.Filled.Star else Icons.Outlined.StarOutline,
                                contentDescription = "Favorite",
                                tint = VibrantPinkPill
                            )
                        }
                    }
                }
            }
            Spacer(modifier = Modifier.height(16.dp))
        }
    }
}
