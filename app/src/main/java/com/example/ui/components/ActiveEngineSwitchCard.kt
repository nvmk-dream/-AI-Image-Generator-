package com.example.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.minimumInteractiveComponentSize
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import com.example.ai.RealDeviceAndModelMetrics
import com.example.data.ActiveDiffusionEngine
import com.example.ui.theme.DeepPlumText
import com.example.ui.theme.NeonReadyGreen
import com.example.ui.theme.PinkGlowEnd
import com.example.ui.theme.PinkGlowStart
import com.example.ui.theme.PurpleActionEnd
import com.example.ui.theme.PurpleActionStart
import com.example.ui.theme.SoftAmberStatus

@Composable
fun ActiveEngineSwitchCard(
    activeEngine: ActiveDiffusionEngine,
    metrics: RealDeviceAndModelMetrics,
    isTamil: Boolean,
    darkTheme: Boolean,
    enabled: Boolean = true,
    tagPrefix: String = "engine",
    onSelectEngine: (ActiveDiffusionEngine) -> Unit,
    extraBottomContent: (@Composable ColumnScope.() -> Unit)? = null
) {
    val textColor = if (darkTheme) Color.White else DeepPlumText
    val subColor = if (darkTheme) Color(0xFFE1C9F7) else Color(0xFF4F376E)
    val selectedReady = metrics.isEngineReady(activeEngine)

    FrostedGlassCard(darkTheme = darkTheme) {
        Row(
            verticalAlignment = Alignment.CenterVertically,
            modifier = Modifier.fillMaxWidth()
        ) {
            Text(
                text = if (isTamil) {
                    "Active Model தேர்வு (Default: Fast LCM)"
                } else {
                    "Active Model Switch (Default: Fast LCM)"
                },
                color = textColor,
                style = MaterialTheme.typography.titleSmall.copy(fontWeight = FontWeight.ExtraBold),
                modifier = Modifier.weight(1f)
            )
            Text(
                text = if (selectedReady) {
                    if (isTamil) "🔗 தயார்" else "🔗 Linked"
                } else {
                    if (isTamil) "மாடல் ஏற்றவும்" else "Upload Pair"
                },
                color = if (selectedReady) NeonReadyGreen else SoftAmberStatus,
                style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.ExtraBold)
            )
        }

        Spacer(modifier = Modifier.height(8.dp))

        Row(
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            modifier = Modifier.fillMaxWidth()
        ) {
            EngineOptionPill(
                title = "⚡ Fast LCM",
                subtitle = if (isTamil) "வேகமானது • 4–8 Steps" else "Fast • 4–8 Steps",
                isSelected = activeEngine == ActiveDiffusionEngine.FAST_LCM,
                isReadyOnDisk = metrics.isFastLcmEngineReady,
                activeBrush = Brush.horizontalGradient(listOf(PinkGlowStart, PinkGlowEnd)),
                darkTheme = darkTheme,
                textColor = textColor,
                subColor = subColor,
                enabled = enabled,
                testTag = "${tagPrefix}_switch_fast_lcm",
                onClick = { onSelectEngine(ActiveDiffusionEngine.FAST_LCM) },
                modifier = Modifier.weight(1f)
            )

            EngineOptionPill(
                title = "🎨 Standard SD",
                subtitle = if (isTamil) "ஹை-குவாலிட்டி • 16+ Steps" else "High Quality • 16+ Steps",
                isSelected = activeEngine == ActiveDiffusionEngine.STANDARD_SD,
                isReadyOnDisk = metrics.isStandardSdEngineReady,
                activeBrush = Brush.horizontalGradient(listOf(PurpleActionStart, PurpleActionEnd)),
                darkTheme = darkTheme,
                textColor = textColor,
                subColor = subColor,
                enabled = enabled,
                testTag = "${tagPrefix}_switch_standard_sd",
                onClick = { onSelectEngine(ActiveDiffusionEngine.STANDARD_SD) },
                modifier = Modifier.weight(1f)
            )
        }

        Spacer(modifier = Modifier.height(8.dp))

        Box(
            modifier = Modifier
                .fillMaxWidth()
                .clip(RoundedCornerShape(12.dp))
                .background(
                    Color.White.copy(alpha = if (darkTheme) 0.10f else 0.62f)
                )
                .border(
                    width = 1.dp,
                    color = if (selectedReady) NeonReadyGreen.copy(alpha = 0.85f)
                    else SoftAmberStatus.copy(alpha = 0.75f),
                    shape = RoundedCornerShape(12.dp)
                )
                .padding(horizontal = 10.dp, vertical = 7.dp)
        ) {
            Text(
                text = metrics.activeEngineSummaryLine(activeEngine, isTamil),
                color = textColor,
                style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.SemiBold)
            )
        }

        if (extraBottomContent != null) {
            Spacer(modifier = Modifier.height(8.dp))
            extraBottomContent()
        }
    }
}

@Composable
private fun EngineOptionPill(
    title: String,
    subtitle: String,
    isSelected: Boolean,
    isReadyOnDisk: Boolean,
    activeBrush: Brush,
    darkTheme: Boolean,
    textColor: Color,
    subColor: Color,
    enabled: Boolean,
    testTag: String,
    onClick: () -> Unit,
    modifier: Modifier = Modifier
) {
    val pillShape = RoundedCornerShape(16.dp)
    val inactiveBrush = Brush.horizontalGradient(
        listOf(
            Color.White.copy(alpha = if (darkTheme) 0.14f else 0.58f),
            Color.White.copy(alpha = if (darkTheme) 0.08f else 0.46f)
        )
    )

    Column(
        horizontalAlignment = Alignment.CenterHorizontally,
        modifier = modifier
            .minimumInteractiveComponentSize()
            .clip(pillShape)
            .background(if (isSelected) activeBrush else inactiveBrush)
            .border(
                width = if (isSelected) 1.5.dp else 1.dp,
                color = if (isSelected) Color.White else Color.White.copy(alpha = 0.72f),
                shape = pillShape
            )
            .clickable(enabled = enabled, onClick = onClick)
            .padding(vertical = 8.dp, horizontal = 8.dp)
            .testTag(testTag)
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Box(
                modifier = Modifier
                    .size(8.dp)
                    .clip(CircleShape)
                    .background(if (isReadyOnDisk) NeonReadyGreen else SoftAmberStatus)
            )
            Spacer(modifier = Modifier.width(6.dp))
            Text(
                text = title,
                color = if (isSelected) Color.White else textColor,
                style = MaterialTheme.typography.labelLarge.copy(fontWeight = FontWeight.ExtraBold),
                maxLines = 1,
                overflow = TextOverflow.Ellipsis
            )
        }
        Spacer(modifier = Modifier.height(2.dp))
        Text(
            text = subtitle,
            color = if (isSelected) Color.White.copy(alpha = 0.92f) else subColor,
            style = MaterialTheme.typography.labelSmall,
            maxLines = 1,
            overflow = TextOverflow.Ellipsis
        )
    }
}
