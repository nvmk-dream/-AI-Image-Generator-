package com.example.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Edit
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.example.ai.OfflineTamilPromptTranslator
import com.example.ui.theme.DeepPlumText
import com.example.ui.theme.PurpleActionEnd
import com.example.ui.theme.VibrantPinkPill

private fun mainText(darkTheme: Boolean) = if (darkTheme) Color.White else DeepPlumText
private fun subText(darkTheme: Boolean) = if (darkTheme) Color(0xFFE1C9F7) else Color(0xFF4F376E)

/** "Auto (சிறந்தது)" switch. ON = app picks size / steps / strength itself. */
@Composable
fun AutoQualityCard(
    darkTheme: Boolean,
    auto: Boolean,
    summary: String,
    enabled: Boolean,
    onToggle: (Boolean) -> Unit,
    testTag: String
) {
    FrostedGlassCard(darkTheme = darkTheme) {
        Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.fillMaxWidth()) {
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = "Auto (சிறந்தது)",
                    color = mainText(darkTheme),
                    style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold)
                )
                Text(
                    text = if (auto) summary else "Manual: கீழே உள்ள settings-ஐ நீங்களே மாற்றலாம்",
                    color = subText(darkTheme),
                    style = MaterialTheme.typography.bodySmall,
                    modifier = Modifier.testTag("${testTag}_summary")
                )
            }
            Switch(
                checked = auto,
                onCheckedChange = onToggle,
                enabled = enabled,
                modifier = Modifier.testTag(testTag)
            )
        }
    }
}

/** Shows the English the AI will receive, and Tamil words the dictionary does not know (with quick add). */
@Composable
fun TamilPromptPreview(
    prompt: String,
    dictionary: List<Pair<String, String>>,
    darkTheme: Boolean,
    onAddWord: (tamil: String, english: String) -> Unit,
    testTagPrefix: String
) {
    if (!OfflineTamilPromptTranslator.containsTamil(prompt)) return
    val english = remember(prompt, dictionary) { OfflineTamilPromptTranslator.normalizeForClip(prompt, dictionary) }
    val unknown = remember(prompt, dictionary) { OfflineTamilPromptTranslator.unknownTamilWords(prompt, dictionary) }
    var addingWord by remember { mutableStateOf<String?>(null) }
    var englishInput by remember { mutableStateOf("") }

    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(top = 8.dp)
            .clip(RoundedCornerShape(12.dp))
            .background(Color.White.copy(alpha = if (darkTheme) 0.08f else 0.55f))
            .padding(10.dp)
    ) {
        Text(
            text = "AI-க்கு போகும் English:",
            color = PurpleActionEnd,
            style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.Bold)
        )
        Text(
            text = english,
            color = subText(darkTheme),
            style = MaterialTheme.typography.bodyMedium,
            modifier = Modifier.testTag("${testTagPrefix}_english_preview")
        )
        if (unknown.isNotEmpty()) {
            Spacer(modifier = Modifier.height(6.dp))
            Text(
                text = "இந்த வார்த்தைகள் அகராதியில் இல்லை (AI-க்கு போகாது). Tap செய்து சேர்க்கவும்:",
                color = Color(0xFFD32F2F),
                style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.Bold)
            )
            Spacer(modifier = Modifier.height(4.dp))
            Row(
                horizontalArrangement = Arrangement.spacedBy(6.dp),
                modifier = Modifier
                    .fillMaxWidth()
                    .horizontalScroll(rememberScrollState())
            ) {
                unknown.forEach { word ->
                    Box(
                        modifier = Modifier
                            .clip(RoundedCornerShape(50))
                            .border(1.dp, Color(0xFFD32F2F), RoundedCornerShape(50))
                            .clickable {
                                addingWord = word
                                englishInput = ""
                            }
                            .padding(horizontal = 10.dp, vertical = 6.dp)
                            .testTag("${testTagPrefix}_unknown_$word")
                    ) {
                        Text(
                            text = "$word  + சேர்",
                            color = Color(0xFFD32F2F),
                            style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.Bold)
                        )
                    }
                }
            }
        }
    }

    addingWord?.let { word ->
        AlertDialog(
            onDismissRequest = { addingWord = null },
            title = { Text("\"$word\" → English") },
            text = {
                OutlinedTextField(
                    value = englishInput,
                    onValueChange = { englishInput = it },
                    singleLine = true,
                    placeholder = { Text("e.g. full body shot") },
                    modifier = Modifier.testTag("${testTagPrefix}_add_word_input")
                )
            },
            confirmButton = {
                TextButton(
                    onClick = {
                        if (englishInput.isNotBlank()) {
                            onAddWord(word, englishInput.trim())
                            addingWord = null
                        }
                    },
                    modifier = Modifier.testTag("${testTagPrefix}_add_word_save")
                ) { Text("Save") }
            },
            dismissButton = { TextButton(onClick = { addingWord = null }) { Text("Cancel") } }
        )
    }
}

/** Settings card: editable Tamil → English dictionary used by the offline prompt translator. */
@Composable
fun TamilDictionaryCard(
    darkTheme: Boolean,
    entries: List<Pair<String, String>>,
    onUpsert: (tamil: String, english: String) -> Unit,
    onDelete: (tamil: String) -> Unit,
    onReset: () -> Unit
) {
    var expanded by rememberSaveable { mutableStateOf(false) }
    var search by rememberSaveable { mutableStateOf("") }
    var tamil by rememberSaveable { mutableStateOf("") }
    var english by rememberSaveable { mutableStateOf("") }
    var confirmReset by remember { mutableStateOf(false) }

    FrostedGlassCard(darkTheme = darkTheme) {
        Text(
            text = "Tamil அகராதி (Tamil → English)",
            color = mainText(darkTheme),
            style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold)
        )
        Text(
            text = "Prompt-ல் உள்ள Tamil வார்த்தைகள் இந்த list படி English-ஆக மாறி AI-க்கு போகும். மொத்தம்: ${entries.size}",
            color = subText(darkTheme),
            style = MaterialTheme.typography.bodySmall
        )
        Spacer(modifier = Modifier.height(8.dp))
        OutlinedTextField(
            value = tamil,
            onValueChange = { tamil = it },
            label = { Text("Tamil") },
            singleLine = true,
            modifier = Modifier
                .fillMaxWidth()
                .testTag("dict_tamil_input")
        )
        OutlinedTextField(
            value = english,
            onValueChange = { english = it },
            label = { Text("English") },
            singleLine = true,
            modifier = Modifier
                .fillMaxWidth()
                .testTag("dict_english_input")
        )
        Spacer(modifier = Modifier.height(8.dp))
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.fillMaxWidth()) {
            Button(
                onClick = {
                    if (tamil.isNotBlank() && english.isNotBlank()) {
                        onUpsert(tamil.trim(), english.trim())
                        tamil = ""
                        english = ""
                    }
                },
                modifier = Modifier
                    .weight(1f)
                    .testTag("dict_save_button")
            ) { Text("Add / Save") }
            OutlinedButton(
                onClick = { confirmReset = true },
                modifier = Modifier
                    .weight(1f)
                    .testTag("dict_reset_button")
            ) { Text("Reset") }
        }
        TextButton(onClick = { expanded = !expanded }, modifier = Modifier.testTag("dict_toggle_list")) {
            Text(if (expanded) "List-ஐ மறை" else "முழு list பார்க்க (${entries.size})", color = VibrantPinkPill)
        }
        if (expanded) {
            OutlinedTextField(
                value = search,
                onValueChange = { search = it },
                label = { Text("தேடு / Search") },
                singleLine = true,
                modifier = Modifier
                    .fillMaxWidth()
                    .testTag("dict_search_input")
            )
            entries
                .filter { search.isBlank() || it.first.contains(search, true) || it.second.contains(search, true) }
                .forEach { (ta, en) ->
                    Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.fillMaxWidth()) {
                        Column(
                            modifier = Modifier
                                .weight(1f)
                                .padding(vertical = 4.dp)
                        ) {
                            Text(
                                text = ta,
                                color = mainText(darkTheme),
                                style = MaterialTheme.typography.bodyLarge.copy(fontWeight = FontWeight.SemiBold)
                            )
                            Text(
                                text = en.ifBlank { "(நீக்கப்படும் / removed)" },
                                color = subText(darkTheme),
                                style = MaterialTheme.typography.bodySmall
                            )
                        }
                        IconButton(onClick = {
                            tamil = ta
                            english = en
                        }) {
                            Icon(Icons.Filled.Edit, contentDescription = "Edit $ta", tint = PurpleActionEnd)
                        }
                        IconButton(onClick = { onDelete(ta) }) {
                            Icon(Icons.Filled.Delete, contentDescription = "Delete $ta", tint = Color(0xFFD32F2F))
                        }
                    }
                }
        }
    }

    if (confirmReset) {
        AlertDialog(
            onDismissRequest = { confirmReset = false },
            title = { Text("Reset அகராதி?") },
            text = { Text("நீங்க சேர்த்த / மாற்றிய எல்லா வார்த்தைகளும் நீங்கி, default list திரும்ப வரும்.") },
            confirmButton = {
                TextButton(onClick = {
                    onReset()
                    confirmReset = false
                }) { Text("Reset") }
            },
            dismissButton = { TextButton(onClick = { confirmReset = false }) { Text("Cancel") } }
        )
    }
}
