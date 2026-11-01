package com.example.ui.components

import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.content.Intent
import android.net.Uri
import android.widget.Toast
import androidx.compose.animation.AnimatedVisibility
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
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.OpenInNew
import androidx.compose.material.icons.filled.CloudDownload
import androidx.compose.material.icons.filled.ContentCopy
import androidx.compose.material.icons.filled.ExpandLess
import androidx.compose.material.icons.filled.ExpandMore
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.Star
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
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
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import com.example.R
import com.example.ai.OnnxDownloadGroup
import com.example.ai.OnnxDownloadLinkItem
import com.example.ai.OnnxModelDownloadCatalog
import com.example.ui.theme.DeepPlumText
import com.example.ui.theme.NeonReadyGreen
import com.example.ui.theme.PinkGlowStart
import com.example.ui.theme.PurpleActionEnd
import com.example.ui.theme.PurpleActionStart
import com.example.ui.theme.SoftAmberStatus
import com.example.ui.theme.VibrantPinkPill

@Composable
fun OnnxDownloadLinksCard(
    isTamil: Boolean,
    darkTheme: Boolean,
    modifier: Modifier = Modifier
) {
    val context = LocalContext.current
    val primaryTextColor = if (darkTheme) Color.White else DeepPlumText
    val secondaryTextColor = if (darkTheme) Color(0xFFE1C9F7) else Color(0xFF4F376E)
    var showExtraModels by remember { mutableStateOf(false) }

    val primaryGroups = remember { OnnxModelDownloadCatalog.groups.take(2) }
    val extraGroups = remember { OnnxModelDownloadCatalog.groups.drop(2) }

    FrostedGlassCard(
        darkTheme = darkTheme,
        modifier = modifier.testTag("settings_onnx_download_card")
    ) {
        // Card Header
        Row(
            verticalAlignment = Alignment.CenterVertically,
            modifier = Modifier.fillMaxWidth()
        ) {
            Icon(
                imageVector = Icons.Filled.CloudDownload,
                contentDescription = null,
                tint = VibrantPinkPill,
                modifier = Modifier.size(24.dp)
            )
            Spacer(modifier = Modifier.width(8.dp))
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = if (isTamil) {
                        "ONNX மாடல் Download Links (Hugging Face)"
                    } else {
                        stringResource(R.string.settings_onnx_downloads_title)
                    },
                    color = primaryTextColor,
                    style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold)
                )
                Text(
                    text = if (isTamil) {
                        "நேரடியாக Browser-ல் திறக்க அல்லது Link நகலெடுக்க (Copy)"
                    } else {
                        "Direct Hugging Face links to download complete ONNX model pairs"
                    },
                    color = secondaryTextColor,
                    style = MaterialTheme.typography.labelSmall
                )
            }
        }

        Spacer(modifier = Modifier.height(10.dp))

        // Important Companion-File Tip Banner
        Row(
            verticalAlignment = Alignment.Top,
            modifier = Modifier
                .fillMaxWidth()
                .clip(RoundedCornerShape(14.dp))
                .background(
                    if (darkTheme) Color(0x33FFB300) else Color(0x26FF8F00)
                )
                .border(
                    width = 1.dp,
                    color = SoftAmberStatus.copy(alpha = 0.7f),
                    shape = RoundedCornerShape(14.dp)
                )
                .padding(10.dp)
        ) {
            Icon(
                imageVector = Icons.Filled.Info,
                contentDescription = null,
                tint = SoftAmberStatus,
                modifier = Modifier.size(18.dp)
            )
            Spacer(modifier = Modifier.width(8.dp))
            Text(
                text = if (isTamil) {
                    "முக்கிய குறிப்பு: UNet மாடலில் model.onnx உடன் model.onnx.data (அல்லது weights.pb) கட்டாயம் இருக்க வேண்டும். பெயரை மாற்றாமல் 'unet' என்ற ஃபோல்டரில் வைத்து Models பக்கத்தில் ஏற்றவும்."
                } else {
                    "Important: Split UNet models require BOTH model.onnx and model.onnx.data (or weights.pb). Keep exact filenames inside a 'unet' folder and import in the Models tab."
                },
                color = primaryTextColor,
                style = MaterialTheme.typography.labelSmall
            )
        }

        Spacer(modifier = Modifier.height(10.dp))

        primaryGroups.forEachIndexed { idx, group ->
            if (idx > 0) {
                HorizontalDivider(
                    modifier = Modifier.padding(vertical = 10.dp),
                    thickness = 0.8.dp,
                    color = if (darkTheme) Color.White.copy(alpha = 0.14f) else Color.White.copy(alpha = 0.55f)
                )
            }
            DownloadGroupSection(
                group = group,
                isTamil = isTamil,
                darkTheme = darkTheme,
                primaryTextColor = primaryTextColor,
                secondaryTextColor = secondaryTextColor,
                context = context
            )
        }

        if (extraGroups.isNotEmpty()) {
            Spacer(modifier = Modifier.height(10.dp))
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.Center,
                modifier = Modifier
                    .fillMaxWidth()
                    .minimumInteractiveComponentSize()
                    .clip(RoundedCornerShape(14.dp))
                    .background(Color.White.copy(alpha = if (darkTheme) 0.08f else 0.52f))
                    .clickable { showExtraModels = !showExtraModels }
                    .padding(horizontal = 12.dp, vertical = 8.dp)
                    .testTag("settings_toggle_extra_links")
            ) {
                Text(
                    text = if (isTamil) {
                        if (showExtraModels) "மற்ற Core & Face Swap Links மறை" else "மற்ற Core & Face Swap மாடல் Links காட்டு"
                    } else {
                        if (showExtraModels) "Hide Core & Face Swap Links" else "Show Core Encoders & Face Swap Links"
                    },
                    color = primaryTextColor,
                    style = MaterialTheme.typography.labelLarge.copy(fontWeight = FontWeight.Bold),
                    modifier = Modifier.weight(1f)
                )
                Icon(
                    imageVector = if (showExtraModels) Icons.Filled.ExpandLess else Icons.Filled.ExpandMore,
                    contentDescription = null,
                    tint = primaryTextColor
                )
            }

            AnimatedVisibility(visible = showExtraModels) {
                Column(modifier = Modifier.fillMaxWidth()) {
                    extraGroups.forEach { group ->
                        HorizontalDivider(
                            modifier = Modifier.padding(vertical = 10.dp),
                            thickness = 0.8.dp,
                            color = if (darkTheme) Color.White.copy(alpha = 0.14f) else Color.White.copy(alpha = 0.55f)
                        )
                        DownloadGroupSection(
                            group = group,
                            isTamil = isTamil,
                            darkTheme = darkTheme,
                            primaryTextColor = primaryTextColor,
                            secondaryTextColor = secondaryTextColor,
                            context = context
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun DownloadGroupSection(
    group: OnnxDownloadGroup,
    isTamil: Boolean,
    darkTheme: Boolean,
    primaryTextColor: Color,
    secondaryTextColor: Color,
    context: Context
) {
    Column(
        verticalArrangement = Arrangement.spacedBy(6.dp),
        modifier = Modifier.fillMaxWidth()
    ) {
        // Recommendation / Category Badge
        Row(verticalAlignment = Alignment.CenterVertically) {
            Box(
                modifier = Modifier
                    .clip(RoundedCornerShape(50))
                    .background(
                        if (group.isRecommended) {
                            Brush.horizontalGradient(listOf(PinkGlowStart, VibrantPinkPill))
                        } else {
                            Brush.horizontalGradient(listOf(PurpleActionStart, PurpleActionEnd))
                        }
                    )
                    .padding(horizontal = 10.dp, vertical = 3.dp)
            ) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    if (group.isRecommended) {
                        Icon(
                            imageVector = Icons.Filled.Star,
                            contentDescription = null,
                            tint = Color.White,
                            modifier = Modifier.size(13.dp)
                        )
                        Spacer(modifier = Modifier.width(4.dp))
                    }
                    Text(
                        text = if (isTamil) group.badgeLabelTa else group.badgeLabelEn,
                        color = Color.White,
                        style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.Bold)
                    )
                }
            }
        }

        Text(
            text = if (isTamil) group.titleTa else group.titleEn,
            color = primaryTextColor,
            style = MaterialTheme.typography.bodyLarge.copy(fontWeight = FontWeight.ExtraBold)
        )

        Text(
            text = if (isTamil) group.descriptionTa else group.descriptionEn,
            color = secondaryTextColor,
            style = MaterialTheme.typography.labelSmall
        )

        Spacer(modifier = Modifier.height(2.dp))

        group.links.forEach { link ->
            DownloadLinkItemRow(
                item = link,
                isTamil = isTamil,
                darkTheme = darkTheme,
                primaryTextColor = primaryTextColor,
                secondaryTextColor = secondaryTextColor,
                onOpenUrl = { openExternalUrl(context, link.url) },
                onCopyUrl = { copyUrlToClipboard(context, link.titleEn, link.url, isTamil) }
            )
        }
    }
}

@Composable
private fun DownloadLinkItemRow(
    item: OnnxDownloadLinkItem,
    isTamil: Boolean,
    darkTheme: Boolean,
    primaryTextColor: Color,
    secondaryTextColor: Color,
    onOpenUrl: () -> Unit,
    onCopyUrl: () -> Unit
) {
    val rowShape = RoundedCornerShape(14.dp)
    Row(
        verticalAlignment = Alignment.CenterVertically,
        modifier = Modifier
            .fillMaxWidth()
            .clip(rowShape)
            .background(Color.White.copy(alpha = if (darkTheme) 0.08f else 0.58f))
            .border(1.dp, Color.White.copy(alpha = if (darkTheme) 0.2f else 0.75f), rowShape)
            .padding(start = 10.dp, end = 6.dp, top = 8.dp, bottom = 8.dp)
    ) {
        Column(modifier = Modifier.weight(1f)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text(
                    text = if (isTamil) item.titleTa else item.titleEn,
                    color = primaryTextColor,
                    style = MaterialTheme.typography.bodyMedium.copy(fontWeight = FontWeight.Bold),
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                    modifier = Modifier.weight(1f, fill = false)
                )
                Spacer(modifier = Modifier.width(6.dp))
                Box(
                    modifier = Modifier
                        .clip(RoundedCornerShape(6.dp))
                        .background(
                            if (item.isDirectFile) NeonReadyGreen.copy(alpha = 0.22f)
                            else PurpleActionEnd.copy(alpha = 0.22f)
                        )
                        .padding(horizontal = 6.dp, vertical = 2.dp)
                ) {
                    Text(
                        text = item.sizeBadge,
                        color = primaryTextColor,
                        style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.Bold)
                    )
                }
            }
            Text(
                text = if (isTamil) item.subtitleTa else item.subtitleEn,
                color = secondaryTextColor,
                style = MaterialTheme.typography.labelSmall,
                maxLines = 2,
                overflow = TextOverflow.Ellipsis
            )
        }

        IconButton(
            onClick = onCopyUrl,
            modifier = Modifier.testTag("settings_download_copy_${item.id}")
        ) {
            Icon(
                imageVector = Icons.Filled.ContentCopy,
                contentDescription = stringResource(R.string.cd_copy_download_link),
                tint = PurpleActionEnd,
                modifier = Modifier.size(19.dp)
            )
        }

        val btnShape = RoundedCornerShape(50)
        Row(
            verticalAlignment = Alignment.CenterVertically,
            modifier = Modifier
                .minimumInteractiveComponentSize()
                .clip(btnShape)
                .background(
                    Brush.horizontalGradient(
                        if (item.isDirectFile) {
                            listOf(PinkGlowStart, VibrantPinkPill)
                        } else {
                            listOf(PurpleActionStart, PurpleActionEnd)
                        }
                    )
                )
                .border(1.dp, Color.White.copy(alpha = 0.85f), btnShape)
                .clickable(onClick = onOpenUrl)
                .padding(horizontal = 12.dp, vertical = 7.dp)
                .testTag("settings_download_open_${item.id}")
        ) {
            Icon(
                imageVector = if (item.isDirectFile) {
                    Icons.Filled.CloudDownload
                } else {
                    Icons.AutoMirrored.Filled.OpenInNew
                },
                contentDescription = stringResource(R.string.cd_open_download_link),
                tint = Color.White,
                modifier = Modifier.size(16.dp)
            )
            Spacer(modifier = Modifier.width(4.dp))
            Text(
                text = if (item.isDirectFile) {
                    stringResource(R.string.settings_onnx_downloads_open)
                } else {
                    stringResource(R.string.settings_onnx_downloads_browse)
                },
                color = Color.White,
                style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.Bold)
            )
        }
    }
}

private fun openExternalUrl(context: Context, url: String) {
    runCatching {
        val intent = Intent(Intent.ACTION_VIEW, Uri.parse(url)).apply {
            addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        }
        context.startActivity(intent)
    }.onFailure {
        Toast.makeText(
            context,
            "Unable to open browser. Tap Copy icon to copy link.",
            Toast.LENGTH_SHORT
        ).show()
    }
}

private fun copyUrlToClipboard(context: Context, label: String, url: String, isTamil: Boolean) {
    runCatching {
        val clipboard = context.getSystemService(Context.CLIPBOARD_SERVICE) as? ClipboardManager
        clipboard?.setPrimaryClip(ClipData.newPlainText(label, url))
        val msg = if (isTamil) {
            "Download Link நகலெடுக்கப்பட்டது (Copied)!"
        } else {
            "Copied link: $label"
        }
        Toast.makeText(context, msg, Toast.LENGTH_SHORT).show()
    }
}
