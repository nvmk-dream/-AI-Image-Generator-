package com.example

import android.content.Context
import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.performScrollTo
import androidx.test.core.app.ApplicationProvider
import com.example.ai.DiffusionModelSlot
import com.example.ai.OnnxModelManager
import com.example.ai.RealDeviceAndModelMetrics
import com.example.data.AppPreferencesState
import com.example.ui.screens.SettingsScreen
import com.example.ui.theme.MyApplicationTheme
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import java.io.ByteArrayOutputStream
import java.io.File

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [36])
class ExampleRobolectricTest {

    @get:Rule
    val composeTestRule = createComposeRule()

    @Test
    fun `read string from context`() {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val appName = context.getString(R.string.app_name)
        assertEquals("Image Generator", appName)
    }

    @Test
    fun `incomplete unet shell without weights pb is marked not ready until companion attached`() {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val manager = OnnxModelManager(context)
        manager.deleteAllModels()

        val unetDir = File(manager.modelsRootDir, DiffusionModelSlot.UNET.folderHint).apply { mkdirs() }
        val unetGraph = File(unetDir, "model.onnx")
        val locBytes = "model.onnx.data".toByteArray(Charsets.UTF_8)
        val shellPayload = ByteArrayOutputStream().apply {
            write("onnx_graph_shell_".toByteArray(Charsets.UTF_8))
            write(byteArrayOf(0x0A, 0x08))
            write("location".toByteArray(Charsets.UTF_8))
            write(byteArrayOf(0x12, locBytes.size.toByte()))
            write(locBytes)
            write(ByteArray(4096))
        }.toByteArray()
        unetGraph.writeBytes(shellPayload)

        val metricsBefore = manager.inspectCurrentMetrics()
        val unetBefore = metricsBefore.slots.first { it.slot == DiffusionModelSlot.UNET }
        assertFalse(unetBefore.isInstalled)
        assertTrue(unetBefore.isIncompleteExternalGraph)
        assertEquals(listOf("model.onnx.data"), unetBefore.missingCompanionFiles)

        // Attach companion weights -> slot becomes Ready!
        File(unetDir, "model.onnx.data").writeBytes(ByteArray(80 * 1024) { 2 })
        val metricsAfter = manager.inspectCurrentMetrics()
        val unetAfter = metricsAfter.slots.first { it.slot == DiffusionModelSlot.UNET }
        assertTrue(unetAfter.isInstalled)
        assertFalse(unetAfter.isIncompleteExternalGraph)
        assertTrue(unetAfter.missingCompanionFiles.isEmpty())

        manager.deleteAllModels()
    }

    @Test
    fun `settings screen displays onnx model download links card`() {
        composeTestRule.setContent {
            MyApplicationTheme(darkTheme = false) {
                SettingsScreen(
                    preferences = AppPreferencesState(),
                    metrics = RealDeviceAndModelMetrics(),
                    onBack = {},
                    onOpenModelsManager = {},
                    onOpenContentSafetyPage = {},
                    onResolutionChange = {},
                    onInferenceModeChange = {},
                    onOfflineModeChange = {},
                    onLanguageChange = {},
                    onDarkThemeChange = {},
                    onNotificationsChange = {},
                    onContentFilterToggle = {},
                    onClearCache = {},
                    onUnloadRam = {}
                )
            }
        }
        composeTestRule.waitForIdle()

        composeTestRule.onNodeWithTag("settings_onnx_download_card")
            .performScrollTo()
            .assertIsDisplayed()
        composeTestRule.onNodeWithTag("settings_download_open_lcm_unet_data")
            .performScrollTo()
            .assertIsDisplayed()
        composeTestRule.onNodeWithTag("settings_download_open_sd_compvis_weights_pb")
            .performScrollTo()
            .assertIsDisplayed()
    }
}
