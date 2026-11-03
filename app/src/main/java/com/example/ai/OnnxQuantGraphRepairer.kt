package com.example.ai

import com.example.ai.OnnxProtoCodec.DT_INT8
import com.example.ai.OnnxProtoCodec.DT_UINT8
import com.example.ai.OnnxProtoCodec.OnnxGraphProto
import com.example.ai.OnnxProtoCodec.OnnxModelProto
import com.example.ai.OnnxProtoCodec.OnnxNodeProto
import com.example.ai.OnnxProtoCodec.OnnxOpsetProto
import com.example.ai.OnnxProtoCodec.OnnxTensorProto
import java.io.File
import java.io.RandomAccessFile

enum class ConvRepairMode {
    CLEAN,
    TIER1_UINT8_CONV_INTEGER,
    TIER2_QDQ_CONV
}

/**
 * Automatic on-device graph & weight compatibility repairer for dynamically quantized ONNX models
 * (such as `latentdivergence/lcm-dreamshaper-int8-onnx`) that contain `ConvInteger(10)` nodes with
 * signed `INT8` weights (`ORT_NOT_IMPLEMENTED` on `onnxruntime-android` CPU EP).
 *
 * Strategy:
 * - Tier 1 (Primary, 0 extra RAM, MLAS 8-bit speed): If all `ConvInteger` weights are `INT8` with
 *   scalar zero-points, shifts signed `INT8` weights (`w xor 0x80`) and zero-points (`zp + 128`)
 *   to `UINT8`, which is mathematically bit-identical in `int32`:
 *   `(x - x_zp) * ((w + 128) - (w_zp + 128)) == (x - x_zp) * (w - w_zp)`.
 * - Tier 2 (Universal Fallback, 0 companion file changes): Rewrites `ConvInteger` subgraphs into
 *   standard `DequantizeLinear -> Conv` (supporting both scalar & 1D per-channel scales/zero-points
 *   and both `INT8` & `UINT8` weights) when per-channel zero-points exist or if mobile ORT rejects
 *   `ConvInteger` altogether.
 */
object OnnxQuantGraphRepairer {

    private const val MARKER_FILE_NAME = ".conv_repair.meta"
    private const val JOURNAL_FILE_NAME = ".conv_u8_journal.meta"

    // Exact protobuf tag+len+string for NodeProto.op_type (field 4, wire 2, len 11) = "ConvInteger"
    private val CONV_INTEGER_OP_BYTES = byteArrayOf(
        0x22, 0x0B,
        'C'.code.toByte(), 'o'.code.toByte(), 'n'.code.toByte(), 'v'.code.toByte(),
        'I'.code.toByte(), 'n'.code.toByte(), 't'.code.toByte(), 'e'.code.toByte(),
        'g'.code.toByte(), 'e'.code.toByte(), 'r'.code.toByte()
    )

    private val repairLock = Any()

    fun clearRepairMarkers(slotDir: File?) {
        if (slotDir == null || !slotDir.exists()) return
        runCatching { File(slotDir, MARKER_FILE_NAME).delete() }
        runCatching { File(slotDir, JOURNAL_FILE_NAME).delete() }
    }

    fun getRecordedRepairMode(modelFile: File): ConvRepairMode? = synchronized(repairLock) {
        readCachedMode(modelFile)
    }

    fun ensureCompatibleBeforeSession(
        modelFile: File,
        onStatus: ((String) -> Unit)? = null
    ): ConvRepairMode = synchronized(repairLock) {
        if (!modelFile.exists() || !modelFile.isFile || modelFile.length() < 32L) {
            return ConvRepairMode.CLEAN
        }

        val dir = modelFile.parentFile
        val hasJournal = dir != null && File(dir, JOURNAL_FILE_NAME).exists()
        if (!hasJournal) {
            readCachedMode(modelFile)?.let { return it }
        }

        if (!containsConvIntegerOp(modelFile)) {
            writeCachedMode(modelFile, ConvRepairMode.CLEAN)
            return ConvRepairMode.CLEAN
        }

        val maxSafeProtoBytes = (Runtime.getRuntime().maxMemory() / 2L)
            .coerceAtLeast(96L * 1024L * 1024L)
        if (modelFile.length() > maxSafeProtoBytes) {
            return ConvRepairMode.CLEAN
        }

        val model = runCatching { OnnxProtoCodec.parseModel(modelFile.readBytes()) }.getOrNull()
            ?: return ConvRepairMode.CLEAN
        val graph = model.graph ?: return ConvRepairMode.CLEAN
        val convNodes = graph.nodes.filter { it.opType == "ConvInteger" }
        if (convNodes.isEmpty()) {
            writeCachedMode(modelFile, ConvRepairMode.CLEAN)
            return ConvRepairMode.CLEAN
        }

        val initMap = graph.initializers.associateBy { it.name }
        if (!hasJournal && isAlreadyUint8Compatible(convNodes, initMap)) {
            writeCachedMode(modelFile, ConvRepairMode.TIER1_UINT8_CONV_INTEGER)
            return ConvRepairMode.TIER1_UINT8_CONV_INTEGER
        }

        if (canRepairViaTier1Uint8(convNodes, graph.nodes, initMap, dir)) {
            onStatus?.invoke("Optimizing INT8 UNet for Android CPU (One-time)...")
            val ok = runCatching {
                applyTier1Uint8Repair(modelFile, model, graph, convNodes, initMap)
            }.getOrDefault(false)
            if (ok) {
                writeCachedMode(modelFile, ConvRepairMode.TIER1_UINT8_CONV_INTEGER)
                return ConvRepairMode.TIER1_UINT8_CONV_INTEGER
            }
        }

        onStatus?.invoke("Optimizing INT8 Conv Graph for Android CPU (One-time)...")
        val qdqOk = runCatching {
            applyTier2QdqRewrite(modelFile, model, graph)
        }.getOrDefault(false)
        if (qdqOk) {
            writeCachedMode(modelFile, ConvRepairMode.TIER2_QDQ_CONV)
            return ConvRepairMode.TIER2_QDQ_CONV
        }

        return ConvRepairMode.CLEAN
    }

    /**
     * Triggered automatically if ONNX Runtime on a device still throws ORT_NOT_IMPLEMENTED for
     * ConvInteger(10) even after Tier-1 UINT8 conversion (e.g., stripped mobile ORT build).
     */
    fun upgradeToQdqConv(
        modelFile: File,
        onStatus: ((String) -> Unit)? = null
    ): Boolean = synchronized(repairLock) {
        if (!modelFile.exists() || !containsConvIntegerOp(modelFile)) return false
        onStatus?.invoke("Rebuilding INT8 Conv layers for mobile CPU (One-time)...")
        val model = runCatching { OnnxProtoCodec.parseModel(modelFile.readBytes()) }.getOrNull()
            ?: return false
        val graph = model.graph ?: return false
        val ok = runCatching { applyTier2QdqRewrite(modelFile, model, graph) }.getOrDefault(false)
        if (ok) {
            writeCachedMode(modelFile, ConvRepairMode.TIER2_QDQ_CONV)
        }
        return ok
    }

    fun containsConvIntegerOp(onnxFile: File): Boolean {
        if (!onnxFile.exists() || !onnxFile.isFile) return false
        val needle = CONV_INTEGER_OP_BYTES
        val buf = ByteArray(64 * 1024)
        var carry = ByteArray(0)
        return runCatching {
            onnxFile.inputStream().use { input ->
                while (true) {
                    val read = input.read(buf)
                    if (read <= 0) break
                    val window = ByteArray(carry.size + read)
                    System.arraycopy(carry, 0, window, 0, carry.size)
                    System.arraycopy(buf, 0, window, carry.size, read)
                    if (indexOfBytes(window, needle) >= 0) return@use true
                    val keep = minOf(window.size, needle.size)
                    carry = window.copyOfRange(window.size - keep, window.size)
                }
                false
            }
        }.getOrDefault(false)
    }

    private fun indexOfBytes(haystack: ByteArray, needle: ByteArray): Int {
        val max = haystack.size - needle.size
        if (max < 0) return -1
        outer@ for (i in 0..max) {
            for (j in needle.indices) {
                if (haystack[i + j] != needle[j]) continue@outer
            }
            return i
        }
        return -1
    }

    private fun isAlreadyUint8Compatible(
        convNodes: List<OnnxNodeProto>,
        initMap: Map<String, OnnxTensorProto>
    ): Boolean {
        return convNodes.all { n ->
            if (n.inputs.size < 2) return@all false
            val w = initMap[n.inputs[1]] ?: return@all false
            if (w.dataType != DT_UINT8) return@all false
            val wZpName = n.inputs.getOrNull(3).orEmpty()
            if (wZpName.isEmpty()) return@all true
            val wZp = initMap[wZpName] ?: return@all false
            wZp.dataType == DT_UINT8 && wZp.elementCount() == 1L
        }
    }

    private fun canRepairViaTier1Uint8(
        convNodes: List<OnnxNodeProto>,
        allNodes: List<OnnxNodeProto>,
        initMap: Map<String, OnnxTensorProto>,
        modelDir: File?
    ): Boolean {
        val consumers = HashMap<String, MutableList<OnnxNodeProto>>()
        for (n in allNodes) {
            for (inp in n.inputs) {
                if (inp.isNotEmpty()) consumers.getOrPut(inp) { mutableListOf() }.add(n)
            }
        }
        for (n in convNodes) {
            if (n.inputs.size < 2) return false
            val wName = n.inputs[1]
            val w = initMap[wName] ?: return false
            if (w.dataType != DT_INT8 && w.dataType != DT_UINT8) return false
            if (consumers[wName].orEmpty().any { it.opType != "ConvInteger" }) return false

            if (w.isExternal()) {
                if (modelDir == null) return false
                val ext = w.externalMap()
                val loc = ext["location"]?.takeIf { it.isNotBlank() } ?: return false
                val extFile = File(modelDir, loc)
                if (!extFile.exists()) return false
                val offset = ext["offset"]?.toLongOrNull() ?: 0L
                val length = ext["length"]?.toLongOrNull() ?: w.elementCount()
                if (length <= 0L || offset < 0L || offset + length > extFile.length()) return false
            }

            val xZpName = n.inputs.getOrNull(2).orEmpty()
            if (xZpName.isNotEmpty()) {
                initMap[xZpName]?.let { if (it.elementCount() != 1L) return false }
            }
            val wZpName = n.inputs.getOrNull(3).orEmpty()
            if (wZpName.isNotEmpty()) {
                val wZp = initMap[wZpName] ?: return false
                if (wZp.elementCount() != 1L) return false
                val bytes = OnnxProtoCodec.readTensor8BitBytes(wZp, modelDir) ?: return false
                if (bytes.size != 1) return false
            }
        }
        return true
    }

    private fun applyTier1Uint8Repair(
        modelFile: File,
        model: OnnxModelProto,
        graph: OnnxGraphProto,
        convNodes: List<OnnxNodeProto>,
        initMap: Map<String, OnnxTensorProto>
    ): Boolean {
        val modelDir = modelFile.parentFile
        val journalFile = modelDir?.let { File(it, JOURNAL_FILE_NAME) }
        val completedExtRanges = journalFile?.takeIf { it.exists() }
            ?.readLines()?.map { it.trim() }?.filter { it.isNotEmpty() }?.toMutableSet()
            ?: mutableSetOf()

        val flippedTensorNames = HashSet<String>()
        val createdU8ZpInitializers = LinkedHashMap<Int, String>()

        fun getOrCreateU8ZpName(u8Val: Int): String {
            val clamped = u8Val and 0xFF
            return createdU8ZpInitializers.getOrPut(clamped) {
                val zpInitName = "__ort_conv_u8_zp_$clamped"
                if (graph.initializers.none { it.name == zpInitName }) {
                    graph.initializers.add(
                        OnnxProtoCodec.create8BitInitializer(
                            name = zpInitName,
                            dataType = DT_UINT8,
                            dims = longArrayOf(),
                            values = byteArrayOf(clamped.toByte())
                        )
                    )
                }
                zpInitName
            }
        }

        for (node in convNodes) {
            val wName = node.inputs[1]
            val w = initMap[wName] ?: return false
            val wasSignedInt8 = w.dataType == DT_INT8

            if (wasSignedInt8 && flippedTensorNames.add(wName)) {
                if (w.isExternal()) {
                    val ext = w.externalMap()
                    val loc = ext["location"] ?: return false
                    val extFile = File(modelDir, loc)
                    val offset = ext["offset"]?.toLongOrNull() ?: 0L
                    val length = ext["length"]?.toLongOrNull() ?: w.elementCount()
                    val rangeKey = "$loc:$offset:$length"
                    if (rangeKey !in completedExtRanges) {
                        xorFlipExternalRange(extFile, offset, length)
                        completedExtRanges.add(rangeKey)
                        journalFile?.appendText("$rangeKey\n")
                    }
                } else {
                    val bytes = OnnxProtoCodec.readTensor8BitBytes(w, modelDir) ?: return false
                    for (i in bytes.indices) {
                        bytes[i] = (bytes[i].toInt() xor 0x80).toByte()
                    }
                    w.int32Data.clear()
                    w.rawData = bytes
                }
                w.dataType = DT_UINT8
            }

            // Compute matching UINT8 w_zero_point without mutating any shared INT8 zero-point initializer
            val wZpName = node.inputs.getOrNull(3).orEmpty()
            val rawZpByte = if (wZpName.isNotEmpty()) {
                val zpInit = initMap[wZpName] ?: return false
                val b = OnnxProtoCodec.readTensor8BitBytes(zpInit, modelDir)?.firstOrNull() ?: return false
                b to zpInit.dataType
            } else {
                0.toByte() to (if (wasSignedInt8) DT_INT8 else DT_UINT8)
            }

            val u8ZpValue = if (rawZpByte.second == DT_INT8) {
                (rawZpByte.first.toInt() + 128) and 0xFF
            } else {
                rawZpByte.first.toInt() and 0xFF
            }
            val newZpInitName = getOrCreateU8ZpName(u8ZpValue)
            while (node.inputs.size < 4) {
                node.inputs.add("")
            }
            node.inputs[3] = newZpInitName
        }

        // Remove any legacy duplicate entries in graph.inputs / valueInfos for modified weight tensors
        if (flippedTensorNames.isNotEmpty()) {
            graph.inputs.removeAll { it.name in flippedTensorNames }
            graph.valueInfos.removeAll { it.name in flippedTensorNames }
        }

        atomicWriteModel(modelFile, OnnxProtoCodec.serializeModel(model))
        journalFile?.delete()
        return true
    }

    private fun xorFlipExternalRange(file: File, offset: Long, length: Long) {
        val chunk = ByteArray(1024 * 1024)
        RandomAccessFile(file, "rw").use { raf ->
            var pos = offset
            var remaining = length
            while (remaining > 0L) {
                val toProcess = minOf(chunk.size.toLong(), remaining).toInt()
                raf.seek(pos)
                raf.readFully(chunk, 0, toProcess)
                for (i in 0 until toProcess) {
                    chunk[i] = (chunk[i].toInt() xor 0x80).toByte()
                }
                raf.seek(pos)
                raf.write(chunk, 0, toProcess)
                pos += toProcess
                remaining -= toProcess
            }
            runCatching { raf.fd.sync() }
        }
    }

    private data class ConvQdqMatch(
        val convIntNode: OnnxNodeProto,
        val finalScaleMulNode: OnnxNodeProto,
        val xFloatTensor: String,
        val wQuantTensor: String,
        val wDataType: Int,
        val scaleFloats: FloatArray,
        val zpBytes: ByteArray,
        val isPerChannel: Boolean,
        val targetOutputTensor: String,
        val helperNodes: List<OnnxNodeProto>
    )

    private fun applyTier2QdqRewrite(
        modelFile: File,
        model: OnnxModelProto,
        graph: OnnxGraphProto
    ): Boolean {
        val modelDir = modelFile.parentFile
        val initMap = graph.initializers.associateBy { it.name }.toMutableMap()
        val producerByOut = HashMap<String, OnnxNodeProto>()
        val consumersByIn = HashMap<String, MutableList<OnnxNodeProto>>()

        for (n in graph.nodes) {
            for (o in n.outputs) if (o.isNotEmpty()) producerByOut[o] = n
            for (inp in n.inputs) if (inp.isNotEmpty()) consumersByIn.getOrPut(inp) { mutableListOf() }.add(n)
        }

        fun unwrapShapeOps(tensorName: String, collectedHelpers: MutableList<OnnxNodeProto>? = null): String {
            var cur = tensorName
            var guard = 0
            while (guard++ < 6) {
                val p = producerByOut[cur] ?: break
                if (p.opType in setOf("Reshape", "Unsqueeze", "Squeeze", "Identity") && p.inputs.isNotEmpty()) {
                    collectedHelpers?.add(p)
                    cur = p.inputs[0]
                } else {
                    break
                }
            }
            return cur
        }

        val convNodes = graph.nodes.filter { it.opType == "ConvInteger" }
        if (convNodes.isEmpty()) return false

        val matches = LinkedHashMap<OnnxNodeProto, ConvQdqMatch>()
        var anyPerChannel = false

        for ((idx, convNode) in convNodes.withIndex()) {
            if (convNode.inputs.size < 2 || convNode.outputs.isEmpty()) return false
            val xQuant = convNode.inputs[0]
            val wQuant = convNode.inputs[1]
            val wZpName = convNode.inputs.getOrNull(3).orEmpty()
            val yInt32 = convNode.outputs[0]

            val wInit = initMap[wQuant] ?: return false
            if (wInit.dataType != DT_INT8 && wInit.dataType != DT_UINT8) return false
            val cOut = (wInit.dims.firstOrNull() ?: 1L).toInt().coerceAtLeast(1)

            val helpers = mutableListOf<OnnxNodeProto>(convNode)
            val xQuantRootProd = producerByOut[unwrapShapeOps(xQuant, helpers)]
            val xFloat: String
            val xScale: String
            when (xQuantRootProd?.opType) {
                "DynamicQuantizeLinear" -> {
                    xFloat = xQuantRootProd.inputs.firstOrNull() ?: return false
                    xScale = xQuantRootProd.outputs.getOrNull(1).orEmpty()
                    helpers.add(xQuantRootProd)
                }
                "QuantizeLinear" -> {
                    xFloat = xQuantRootProd.inputs.firstOrNull() ?: return false
                    xScale = xQuantRootProd.inputs.getOrNull(1).orEmpty()
                    helpers.add(xQuantRootProd)
                }
                else -> return false
            }
            if (xFloat.isEmpty() || xScale.isEmpty()) return false
            val xScaleRoot = unwrapShapeOps(xScale)

            val castNode = consumersByIn[yInt32]?.singleOrNull()?.takeIf { it.opType == "Cast" }
                ?: return false
            helpers.add(castNode)
            val castOut = castNode.outputs.firstOrNull() ?: return false

            val mul1 = consumersByIn[castOut]?.singleOrNull()?.takeIf { it.opType == "Mul" && it.inputs.size == 2 }
                ?: return false
            helpers.add(mul1)
            val mul1Other = if (mul1.inputs[0] == castOut) mul1.inputs[1] else mul1.inputs[0]

            var wScaleExpr = ""
            var finalMulNode: OnnxNodeProto = mul1
            val scaleProd = producerByOut[mul1Other]

            if (scaleProd != null && scaleProd.opType == "Mul" && scaleProd.inputs.size == 2) {
                val s0Root = unwrapShapeOps(scaleProd.inputs[0])
                val s1Root = unwrapShapeOps(scaleProd.inputs[1])
                if (s0Root == xScaleRoot) {
                    wScaleExpr = scaleProd.inputs[1]
                    helpers.add(scaleProd)
                } else if (s1Root == xScaleRoot) {
                    wScaleExpr = scaleProd.inputs[0]
                    helpers.add(scaleProd)
                }
            }

            if (wScaleExpr.isEmpty()) {
                val mul2 = consumersByIn[mul1.outputs[0]]?.singleOrNull()
                    ?.takeIf { it.opType == "Mul" && it.inputs.size == 2 }
                    ?: return false
                helpers.add(mul2)
                finalMulNode = mul2
                val mul2Other = if (mul2.inputs[0] == mul1.outputs[0]) mul2.inputs[1] else mul2.inputs[0]
                if (unwrapShapeOps(mul1Other) == xScaleRoot) {
                    wScaleExpr = mul2Other
                } else if (unwrapShapeOps(mul2Other) == xScaleRoot) {
                    wScaleExpr = mul1Other
                } else {
                    return false
                }
            }

            val wScaleRoot = unwrapShapeOps(wScaleExpr, helpers)
            val wScaleInit = initMap[wScaleRoot] ?: return false
            val rawScales = OnnxProtoCodec.readTensorFloats(wScaleInit, modelDir) ?: return false
            if (rawScales.isEmpty() || (rawScales.size != 1 && rawScales.size != cOut)) return false

            val rawZps = if (wZpName.isNotEmpty()) {
                val wZpRoot = unwrapShapeOps(wZpName, helpers)
                val wZpInit = initMap[wZpRoot] ?: return false
                OnnxProtoCodec.readTensor8BitBytes(wZpInit, modelDir) ?: return false
            } else {
                byteArrayOf(0)
            }
            if (rawZps.isEmpty() || (rawZps.size != 1 && rawZps.size != cOut)) return false

            val isPerChannel = rawScales.size > 1 || rawZps.size > 1
            if (isPerChannel) anyPerChannel = true

            val finalScales = if (isPerChannel && rawScales.size == 1) {
                FloatArray(cOut) { rawScales[0] }
            } else {
                rawScales
            }
            val finalZps = if (isPerChannel && rawZps.size == 1) {
                ByteArray(cOut) { rawZps[0] }
            } else {
                rawZps
            }

            matches[finalMulNode] = ConvQdqMatch(
                convIntNode = convNode,
                finalScaleMulNode = finalMulNode,
                xFloatTensor = xFloat,
                wQuantTensor = wQuant,
                wDataType = wInit.dataType,
                scaleFloats = finalScales,
                zpBytes = finalZps,
                isPerChannel = isPerChannel,
                targetOutputTensor = finalMulNode.outputs.firstOrNull() ?: return false,
                helperNodes = helpers
            )
            // Touch idx so names stay unique even if node.name is blank
            if (convNode.name.isBlank()) convNode.name = "conv_quant_$idx"
        }

        // Replace each finalScaleMulNode in-place with [DequantizeLinear, Conv] to preserve topological order
        val rewrittenNodes = ArrayList<OnnxNodeProto>(graph.nodes.size + matches.size)
        val candidateDeadNodes = HashSet<OnnxNodeProto>()
        val removedIntermediateNames = HashSet<String>()

        var uid = 0
        for (node in graph.nodes) {
            val match = matches[node]
            if (match != null) {
                uid++
                val safeBase = match.convIntNode.name.replace(Regex("[^A-Za-z0-9_/.-]"), "_")
                val dqScaleName = "${safeBase}_ort_dq_scale_$uid"
                val dqZpName = "${safeBase}_ort_dq_zp_$uid"
                val dqOutName = "${safeBase}_ort_w_fp32_$uid"
                val paramDims = if (match.isPerChannel) {
                    longArrayOf(match.scaleFloats.size.toLong())
                } else {
                    longArrayOf()
                }

                graph.initializers.add(
                    OnnxProtoCodec.createFloatInitializer(dqScaleName, paramDims, match.scaleFloats)
                )
                graph.initializers.add(
                    OnnxProtoCodec.create8BitInitializer(dqZpName, match.wDataType, paramDims, match.zpBytes)
                )

                val dqNode = OnnxNodeProto(
                    inputs = mutableListOf(match.wQuantTensor, dqScaleName, dqZpName),
                    outputs = mutableListOf(dqOutName),
                    name = "${safeBase}_ort_dq_$uid",
                    opType = "DequantizeLinear",
                    attributesRaw = if (match.isPerChannel) {
                        mutableListOf(OnnxProtoCodec.buildIntAttributeRaw("axis", 0L))
                    } else {
                        mutableListOf()
                    }
                )
                val convFpNode = OnnxNodeProto(
                    inputs = mutableListOf(match.xFloatTensor, dqOutName),
                    outputs = mutableListOf(match.targetOutputTensor),
                    name = "${safeBase}_ort_conv_$uid",
                    opType = "Conv",
                    domain = match.convIntNode.domain,
                    attributesRaw = match.convIntNode.attributesRaw.toMutableList()
                )

                rewrittenNodes.add(dqNode)
                rewrittenNodes.add(convFpNode)
                candidateDeadNodes.addAll(match.helperNodes)
            } else {
                rewrittenNodes.add(node)
            }
        }

        // Iterative dead-node pruning strictly limited to quantization helper nodes
        val graphOutputsSet = graph.outputs.map { it.name }.toSet()
        var pruned = true
        while (pruned) {
            pruned = false
            val liveInputs = HashSet<String>(graphOutputsSet)
            for (n in rewrittenNodes) {
                for (inp in n.inputs) if (inp.isNotEmpty()) liveInputs.add(inp)
            }
            val it = rewrittenNodes.iterator()
            while (it.hasNext()) {
                val n = it.next()
                if (n in candidateDeadNodes && n.outputs.all { o -> o.isEmpty() || o !in liveInputs }) {
                    n.outputs.filter { it.isNotEmpty() }.forEach { removedIntermediateNames.add(it) }
                    it.remove()
                    pruned = true
                }
            }
        }

        graph.nodes.clear()
        graph.nodes.addAll(rewrittenNodes)
        if (removedIntermediateNames.isNotEmpty()) {
            graph.valueInfos.removeAll { it.name in removedIntermediateNames }
        }

        if (anyPerChannel) {
            val stdOpset = model.opsets.firstOrNull { it.domain.isEmpty() || it.domain == "ai.onnx" }
            if (stdOpset == null) {
                model.opsets.add(OnnxOpsetProto(domain = "", version = 13L))
            } else if (stdOpset.version < 13L) {
                stdOpset.version = 13L
            }
        }

        atomicWriteModel(modelFile, OnnxProtoCodec.serializeModel(model))
        return true
    }

    private fun atomicWriteModel(targetFile: File, bytes: ByteArray) {
        val dir = targetFile.parentFile ?: error("Missing model parent directory")
        val tmpFile = File(dir, "${targetFile.name}.repair_tmp")
        tmpFile.outputStream().use { out ->
            out.write(bytes)
            out.flush()
            runCatching { out.fd.sync() }
        }
        if (tmpFile.length() < 32L) {
            tmpFile.delete()
            error("Serialized ONNX model unexpectedly small")
        }
        if (!tmpFile.renameTo(targetFile)) {
            tmpFile.copyTo(targetFile, overwrite = true)
            tmpFile.delete()
        }
    }

    private fun readCachedMode(modelFile: File): ConvRepairMode? {
        val dir = modelFile.parentFile ?: return null
        val marker = File(dir, MARKER_FILE_NAME)
        if (!marker.exists()) return null
        val parts = runCatching { marker.readText().trim().split("|") }.getOrNull() ?: return null
        if (parts.size < 3) return null
        val expectedLen = parts[1].toLongOrNull() ?: return null
        val expectedMtime = parts[2].toLongOrNull() ?: return null
        if (modelFile.length() != expectedLen || modelFile.lastModified() != expectedMtime) {
            return null
        }
        return runCatching { ConvRepairMode.valueOf(parts[0]) }.getOrNull()
    }

    private fun writeCachedMode(modelFile: File, mode: ConvRepairMode) {
        val dir = modelFile.parentFile ?: return
        runCatching {
            File(dir, MARKER_FILE_NAME).writeText(
                "${mode.name}|${modelFile.length()}|${modelFile.lastModified()}"
            )
        }
    }
}
