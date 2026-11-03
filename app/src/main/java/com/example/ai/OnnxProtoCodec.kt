package com.example.ai

import java.io.ByteArrayOutputStream
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Lightweight, zero-dependency ONNX binary Protobuf codec (onnx.proto3 wire format).
 * Preserves unknown/unmodified fields verbatim so external_data offsets, metadata_props,
 * shapes, and operator attributes remain byte-accurate when repairing quantized graphs.
 */
internal object OnnxProtoCodec {

    const val WIRE_VARINT = 0
    const val WIRE_FIXED64 = 1
    const val WIRE_LENGTH_DELIMITED = 2
    const val WIRE_FIXED32 = 5

    // ONNX TensorProto.DataType constants
    const val DT_FLOAT = 1
    const val DT_UINT8 = 2
    const val DT_INT8 = 3
    const val DT_INT32 = 6
    const val DT_INT64 = 7
    const val DT_FLOAT16 = 10

    // ONNX TensorProto.DataLocation constants
    const val DATA_LOC_DEFAULT = 0
    const val DATA_LOC_EXTERNAL = 1

    class RawField(
        val fieldNumber: Int,
        val wireType: Int,
        val varintVal: Long = 0L,
        val fixed32Val: Int = 0,
        val fixed64Val: Long = 0L,
        val bytesVal: ByteArray? = null
    )

    class OnnxOpsetProto(
        var domain: String = "",
        var version: Long = 0L,
        val otherFields: MutableList<RawField> = mutableListOf()
    )

    class OnnxExternalEntry(
        var key: String = "",
        var value: String = ""
    )

    class OnnxTensorProto(
        val dims: MutableList<Long> = mutableListOf(),
        var dataType: Int = 0,
        val floatData: MutableList<Float> = mutableListOf(),
        val int32Data: MutableList<Int> = mutableListOf(),
        var name: String = "",
        var rawData: ByteArray? = null,
        val externalData: MutableList<OnnxExternalEntry> = mutableListOf(),
        var dataLocation: Int = DATA_LOC_DEFAULT,
        val otherFields: MutableList<RawField> = mutableListOf()
    ) {
        fun elementCount(): Long {
            if (dims.isEmpty()) return 1L
            var prod = 1L
            for (d in dims) {
                if (d <= 0L) return 0L
                prod *= d
            }
            return prod
        }

        fun externalMap(): Map<String, String> =
            externalData.associate { it.key to it.value }

        fun isExternal(): Boolean =
            dataLocation == DATA_LOC_EXTERNAL || externalData.any { it.key == "location" }
    }

    class OnnxNodeProto(
        val inputs: MutableList<String> = mutableListOf(),
        val outputs: MutableList<String> = mutableListOf(),
        var name: String = "",
        var opType: String = "",
        var domain: String = "",
        val attributesRaw: MutableList<ByteArray> = mutableListOf(),
        val otherFields: MutableList<RawField> = mutableListOf()
    )

    class OnnxValueInfoProto(
        val name: String,
        val rawBytes: ByteArray
    )

    class OnnxGraphProto(
        var name: String = "",
        val nodes: MutableList<OnnxNodeProto> = mutableListOf(),
        val initializers: MutableList<OnnxTensorProto> = mutableListOf(),
        val inputs: MutableList<OnnxValueInfoProto> = mutableListOf(),
        val outputs: MutableList<OnnxValueInfoProto> = mutableListOf(),
        val valueInfos: MutableList<OnnxValueInfoProto> = mutableListOf(),
        val otherFields: MutableList<RawField> = mutableListOf()
    )

    class OnnxModelProto(
        var irVersion: Long = 0L,
        val opsets: MutableList<OnnxOpsetProto> = mutableListOf(),
        var graph: OnnxGraphProto? = null,
        val otherFields: MutableList<RawField> = mutableListOf()
    )

    class ProtoReader(
        private val buf: ByteArray,
        private var pos: Int = 0,
        private val limit: Int = buf.size
    ) {
        fun hasRemaining(): Boolean = pos < limit

        fun readTag(): Int = if (pos >= limit) 0 else readRawVarint32()

        fun readRawVarint32(): Int = readRawVarint64().toInt()

        fun readRawVarint64(): Long {
            var result = 0L
            var shift = 0
            while (shift < 64) {
                if (pos >= limit) error("Truncated ONNX protobuf varint at $pos/$limit")
                val b = buf[pos++].toInt()
                result = result or ((b and 0x7F).toLong() shl shift)
                if ((b and 0x80) == 0) return result
                shift += 7
            }
            error("Malformed ONNX protobuf varint at $pos")
        }

        fun readFixed32(): Int {
            if (pos + 4 > limit) error("Truncated fixed32 at $pos")
            val v = (buf[pos].toInt() and 0xFF) or
                ((buf[pos + 1].toInt() and 0xFF) shl 8) or
                ((buf[pos + 2].toInt() and 0xFF) shl 16) or
                ((buf[pos + 3].toInt() and 0xFF) shl 24)
            pos += 4
            return v
        }

        fun readFixed64(): Long {
            val lo = readFixed32().toLong() and 0xFFFFFFFFL
            val hi = readFixed32().toLong() and 0xFFFFFFFFL
            return lo or (hi shl 32)
        }

        fun readBytes(): ByteArray {
            val len = readRawVarint32()
            if (len < 0 || pos + len > limit) error("Invalid length-delimited size $len at $pos/$limit")
            val out = buf.copyOfRange(pos, pos + len)
            pos += len
            return out
        }

        fun readSubSlice(): Pair<Int, Int> {
            val len = readRawVarint32()
            if (len < 0 || pos + len > limit) error("Invalid submessage size $len at $pos/$limit")
            val start = pos
            pos += len
            return start to (start + len)
        }

        fun readString(): String {
            val len = readRawVarint32()
            if (len < 0 || pos + len > limit) error("Invalid string size $len at $pos/$limit")
            val s = String(buf, pos, len, Charsets.UTF_8)
            pos += len
            return s
        }

        fun readUnknownField(fieldNumber: Int, wireType: Int): RawField = when (wireType) {
            WIRE_VARINT -> RawField(fieldNumber, wireType, varintVal = readRawVarint64())
            WIRE_FIXED64 -> RawField(fieldNumber, wireType, fixed64Val = readFixed64())
            WIRE_LENGTH_DELIMITED -> RawField(fieldNumber, wireType, bytesVal = readBytes())
            WIRE_FIXED32 -> RawField(fieldNumber, wireType, fixed32Val = readFixed32())
            else -> error("Unsupported protobuf wireType=$wireType for field=$fieldNumber")
        }
    }

    class ProtoWriter(initialCapacity: Int = 256) {
        private val out = ByteArrayOutputStream(initialCapacity)

        fun toByteArray(): ByteArray = out.toByteArray()

        fun writeTag(fieldNumber: Int, wireType: Int) {
            writeRawVarint32((fieldNumber shl 3) or wireType)
        }

        fun writeRawVarint32(value: Int) {
            writeRawVarint64(value.toLong() and 0xFFFFFFFFL)
        }

        fun writeSignedInt32Varint(value: Int) {
            // Protobuf standard int32 encodes negative numbers as 10-byte sign-extended 64-bit varints
            writeRawVarint64(value.toLong())
        }

        fun writeRawVarint64(value: Long) {
            var v = value
            while (true) {
                if ((v and 0x7FL.inv()) == 0L) {
                    out.write(v.toInt())
                    return
                }
                out.write((v.toInt() and 0x7F) or 0x80)
                v = v ushr 7
            }
        }

        fun writeFixed32(value: Int) {
            out.write(value and 0xFF)
            out.write((value ushr 8) and 0xFF)
            out.write((value ushr 16) and 0xFF)
            out.write((value ushr 24) and 0xFF)
        }

        fun writeFixed64(value: Long) {
            writeFixed32(value.toInt())
            writeFixed32((value ushr 32).toInt())
        }

        fun writeBytesField(fieldNumber: Int, bytes: ByteArray) {
            writeTag(fieldNumber, WIRE_LENGTH_DELIMITED)
            writeRawVarint32(bytes.size)
            out.write(bytes)
        }

        fun writeStringField(fieldNumber: Int, value: String, writeIfEmpty: Boolean = false) {
            if (value.isEmpty() && !writeIfEmpty) return
            writeBytesField(fieldNumber, value.toByteArray(Charsets.UTF_8))
        }

        fun writeVarintField(fieldNumber: Int, value: Long, writeIfZero: Boolean = false) {
            if (value == 0L && !writeIfZero) return
            writeTag(fieldNumber, WIRE_VARINT)
            writeRawVarint64(value)
        }

        fun writeRawField(f: RawField) {
            writeTag(f.fieldNumber, f.wireType)
            when (f.wireType) {
                WIRE_VARINT -> writeRawVarint64(f.varintVal)
                WIRE_FIXED64 -> writeFixed64(f.fixed64Val)
                WIRE_LENGTH_DELIMITED -> {
                    val b = f.bytesVal ?: ByteArray(0)
                    writeRawVarint32(b.size)
                    out.write(b)
                }
                WIRE_FIXED32 -> writeFixed32(f.fixed32Val)
            }
        }
    }

    fun parseModel(bytes: ByteArray): OnnxModelProto {
        val model = OnnxModelProto()
        val r = ProtoReader(bytes)
        while (r.hasRemaining()) {
            val tag = r.readTag()
            if (tag == 0) break
            val fn = tag ushr 3
            val wt = tag and 0x07
            when {
                fn == 1 && wt == WIRE_VARINT -> model.irVersion = r.readRawVarint64()
                fn == 7 && wt == WIRE_LENGTH_DELIMITED -> {
                    val (s, e) = r.readSubSlice()
                    model.graph = parseGraph(bytes, s, e)
                }
                fn == 8 && wt == WIRE_LENGTH_DELIMITED -> {
                    val (s, e) = r.readSubSlice()
                    model.opsets.add(parseOpset(bytes, s, e))
                }
                else -> model.otherFields.add(r.readUnknownField(fn, wt))
            }
        }
        return model
    }

    private fun parseOpset(buf: ByteArray, start: Int, end: Int): OnnxOpsetProto {
        val opset = OnnxOpsetProto()
        val r = ProtoReader(buf, start, end)
        while (r.hasRemaining()) {
            val tag = r.readTag()
            if (tag == 0) break
            val fn = tag ushr 3
            val wt = tag and 0x07
            when {
                fn == 1 && wt == WIRE_LENGTH_DELIMITED -> opset.domain = r.readString()
                fn == 2 && wt == WIRE_VARINT -> opset.version = r.readRawVarint64()
                else -> opset.otherFields.add(r.readUnknownField(fn, wt))
            }
        }
        return opset
    }

    private fun parseGraph(buf: ByteArray, start: Int, end: Int): OnnxGraphProto {
        val g = OnnxGraphProto()
        val r = ProtoReader(buf, start, end)
        while (r.hasRemaining()) {
            val tag = r.readTag()
            if (tag == 0) break
            val fn = tag ushr 3
            val wt = tag and 0x07
            when {
                fn == 1 && wt == WIRE_LENGTH_DELIMITED -> {
                    val (s, e) = r.readSubSlice()
                    g.nodes.add(parseNode(buf, s, e))
                }
                fn == 2 && wt == WIRE_LENGTH_DELIMITED -> g.name = r.readString()
                fn == 5 && wt == WIRE_LENGTH_DELIMITED -> {
                    val (s, e) = r.readSubSlice()
                    g.initializers.add(parseTensor(buf, s, e))
                }
                fn == 11 && wt == WIRE_LENGTH_DELIMITED -> {
                    val raw = r.readBytes()
                    g.inputs.add(OnnxValueInfoProto(extractFirstFieldString(raw), raw))
                }
                fn == 12 && wt == WIRE_LENGTH_DELIMITED -> {
                    val raw = r.readBytes()
                    g.outputs.add(OnnxValueInfoProto(extractFirstFieldString(raw), raw))
                }
                fn == 13 && wt == WIRE_LENGTH_DELIMITED -> {
                    val raw = r.readBytes()
                    g.valueInfos.add(OnnxValueInfoProto(extractFirstFieldString(raw), raw))
                }
                else -> g.otherFields.add(r.readUnknownField(fn, wt))
            }
        }
        return g
    }

    private fun extractFirstFieldString(raw: ByteArray): String {
        val r = ProtoReader(raw)
        while (r.hasRemaining()) {
            val tag = r.readTag()
            if (tag == 0) break
            val fn = tag ushr 3
            val wt = tag and 0x07
            if (fn == 1 && wt == WIRE_LENGTH_DELIMITED) return r.readString()
            r.readUnknownField(fn, wt)
        }
        return ""
    }

    private fun parseNode(buf: ByteArray, start: Int, end: Int): OnnxNodeProto {
        val n = OnnxNodeProto()
        val r = ProtoReader(buf, start, end)
        while (r.hasRemaining()) {
            val tag = r.readTag()
            if (tag == 0) break
            val fn = tag ushr 3
            val wt = tag and 0x07
            when {
                fn == 1 && wt == WIRE_LENGTH_DELIMITED -> n.inputs.add(r.readString())
                fn == 2 && wt == WIRE_LENGTH_DELIMITED -> n.outputs.add(r.readString())
                fn == 3 && wt == WIRE_LENGTH_DELIMITED -> n.name = r.readString()
                fn == 4 && wt == WIRE_LENGTH_DELIMITED -> n.opType = r.readString()
                fn == 5 && wt == WIRE_LENGTH_DELIMITED -> n.domain = r.readString()
                fn == 6 && wt == WIRE_LENGTH_DELIMITED -> n.attributesRaw.add(r.readBytes())
                else -> n.otherFields.add(r.readUnknownField(fn, wt))
            }
        }
        return n
    }

    private fun parseTensor(buf: ByteArray, start: Int, end: Int): OnnxTensorProto {
        val t = OnnxTensorProto()
        val r = ProtoReader(buf, start, end)
        while (r.hasRemaining()) {
            val tag = r.readTag()
            if (tag == 0) break
            val fn = tag ushr 3
            val wt = tag and 0x07
            when {
                fn == 1 && wt == WIRE_VARINT -> t.dims.add(r.readRawVarint64())
                fn == 1 && wt == WIRE_LENGTH_DELIMITED -> {
                    val (ps, pe) = r.readSubSlice()
                    val pr = ProtoReader(buf, ps, pe)
                    while (pr.hasRemaining()) t.dims.add(pr.readRawVarint64())
                }
                fn == 2 && wt == WIRE_VARINT -> t.dataType = r.readRawVarint32()
                fn == 4 && wt == WIRE_FIXED32 -> t.floatData.add(Float.fromBits(r.readFixed32()))
                fn == 4 && wt == WIRE_LENGTH_DELIMITED -> {
                    val (ps, pe) = r.readSubSlice()
                    val pr = ProtoReader(buf, ps, pe)
                    while (pr.hasRemaining()) t.floatData.add(Float.fromBits(pr.readFixed32()))
                }
                fn == 5 && wt == WIRE_VARINT -> t.int32Data.add(r.readRawVarint64().toInt())
                fn == 5 && wt == WIRE_LENGTH_DELIMITED -> {
                    val (ps, pe) = r.readSubSlice()
                    val pr = ProtoReader(buf, ps, pe)
                    while (pr.hasRemaining()) t.int32Data.add(pr.readRawVarint64().toInt())
                }
                fn == 8 && wt == WIRE_LENGTH_DELIMITED -> t.name = r.readString()
                fn == 9 && wt == WIRE_LENGTH_DELIMITED -> t.rawData = r.readBytes()
                fn == 13 && wt == WIRE_LENGTH_DELIMITED -> {
                    val (es, ee) = r.readSubSlice()
                    t.externalData.add(parseExternalEntry(buf, es, ee))
                }
                fn == 14 && wt == WIRE_VARINT -> t.dataLocation = r.readRawVarint32()
                else -> t.otherFields.add(r.readUnknownField(fn, wt))
            }
        }
        return t
    }

    private fun parseExternalEntry(buf: ByteArray, start: Int, end: Int): OnnxExternalEntry {
        val e = OnnxExternalEntry()
        val r = ProtoReader(buf, start, end)
        while (r.hasRemaining()) {
            val tag = r.readTag()
            if (tag == 0) break
            val fn = tag ushr 3
            val wt = tag and 0x07
            when {
                fn == 1 && wt == WIRE_LENGTH_DELIMITED -> e.key = r.readString()
                fn == 2 && wt == WIRE_LENGTH_DELIMITED -> e.value = r.readString()
                else -> r.readUnknownField(fn, wt)
            }
        }
        return e
    }

    fun serializeModel(model: OnnxModelProto): ByteArray {
        val w = ProtoWriter(1024 * 64)
        w.writeVarintField(1, model.irVersion)
        for (f in model.otherFields) {
            if (f.fieldNumber < 7) w.writeRawField(f)
        }
        model.graph?.let { g ->
            w.writeBytesField(7, serializeGraph(g))
        }
        for (opset in model.opsets) {
            val ow = ProtoWriter(32)
            ow.writeStringField(1, opset.domain)
            ow.writeVarintField(2, opset.version, writeIfZero = true)
            opset.otherFields.forEach { ow.writeRawField(it) }
            w.writeBytesField(8, ow.toByteArray())
        }
        for (f in model.otherFields) {
            if (f.fieldNumber >= 7) w.writeRawField(f)
        }
        return w.toByteArray()
    }

    private fun serializeGraph(g: OnnxGraphProto): ByteArray {
        val w = ProtoWriter(1024 * 64)
        for (n in g.nodes) w.writeBytesField(1, serializeNode(n))
        w.writeStringField(2, g.name)
        for (init in g.initializers) w.writeBytesField(5, serializeTensor(init))
        for (f in g.otherFields) {
            if (f.fieldNumber < 11) w.writeRawField(f)
        }
        for (inp in g.inputs) w.writeBytesField(11, inp.rawBytes)
        for (outp in g.outputs) w.writeBytesField(12, outp.rawBytes)
        for (vi in g.valueInfos) w.writeBytesField(13, vi.rawBytes)
        for (f in g.otherFields) {
            if (f.fieldNumber >= 11) w.writeRawField(f)
        }
        return w.toByteArray()
    }

    private fun serializeNode(n: OnnxNodeProto): ByteArray {
        val w = ProtoWriter(160)
        // Important: empty string "" in NodeProto.inputs represents an omitted optional input in ONNX!
        for (inp in n.inputs) w.writeStringField(1, inp, writeIfEmpty = true)
        for (outp in n.outputs) w.writeStringField(2, outp, writeIfEmpty = true)
        w.writeStringField(3, n.name)
        w.writeStringField(4, n.opType)
        w.writeStringField(5, n.domain)
        for (attr in n.attributesRaw) w.writeBytesField(6, attr)
        n.otherFields.forEach { w.writeRawField(it) }
        return w.toByteArray()
    }

    private fun serializeTensor(t: OnnxTensorProto): ByteArray {
        val w = ProtoWriter(128 + (t.rawData?.size ?: 0))
        for (d in t.dims) w.writeVarintField(1, d, writeIfZero = true)
        w.writeVarintField(2, t.dataType.toLong(), writeIfZero = true)
        if (t.floatData.isNotEmpty()) {
            val pw = ProtoWriter(t.floatData.size * 4)
            for (fv in t.floatData) pw.writeFixed32(fv.toRawBits())
            w.writeBytesField(4, pw.toByteArray())
        }
        if (t.int32Data.isNotEmpty()) {
            val pw = ProtoWriter(t.int32Data.size * 2)
            for (iv in t.int32Data) pw.writeSignedInt32Varint(iv)
            w.writeBytesField(5, pw.toByteArray())
        }
        w.writeStringField(8, t.name)
        t.rawData?.let { w.writeBytesField(9, it) }
        for (ext in t.externalData) {
            val ew = ProtoWriter(48)
            ew.writeStringField(1, ext.key, writeIfEmpty = true)
            ew.writeStringField(2, ext.value, writeIfEmpty = true)
            w.writeBytesField(13, ew.toByteArray())
        }
        if (t.dataLocation != DATA_LOC_DEFAULT) {
            w.writeVarintField(14, t.dataLocation.toLong())
        }
        t.otherFields.forEach { w.writeRawField(it) }
        return w.toByteArray()
    }

    fun buildIntAttributeRaw(name: String, value: Long): ByteArray {
        val w = ProtoWriter(24)
        w.writeStringField(1, name)
        w.writeVarintField(3, value, writeIfZero = true)
        w.writeVarintField(20, 2L) // AttributeProto.AttributeType.INT = 2
        return w.toByteArray()
    }

    fun createFloatInitializer(name: String, dims: LongArray, values: FloatArray): OnnxTensorProto {
        val bb = ByteBuffer.allocate(values.size * 4).order(ByteOrder.LITTLE_ENDIAN)
        for (v in values) bb.putFloat(v)
        return OnnxTensorProto(
            dims = dims.toMutableList(),
            dataType = DT_FLOAT,
            name = name,
            rawData = bb.array()
        )
    }

    fun create8BitInitializer(name: String, dataType: Int, dims: LongArray, values: ByteArray): OnnxTensorProto {
        return OnnxTensorProto(
            dims = dims.toMutableList(),
            dataType = dataType,
            name = name,
            rawData = values.copyOf()
        )
    }

    fun readTensorFloats(t: OnnxTensorProto, modelDir: File?): FloatArray? {
        if (t.floatData.isNotEmpty()) return t.floatData.toFloatArray()
        val raw = readRawOrExternalBytes(t, modelDir, bytesPerElem = if (t.dataType == DT_FLOAT16) 2 else 4)
            ?: return null
        return if (t.dataType == DT_FLOAT16) {
            val sb = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer()
            FloatArray(sb.remaining()) { OnnxDiffusionEngine.halfBitsToFloat(sb.get()) }
        } else {
            val fb = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
            FloatArray(fb.remaining()).also { fb.get(it) }
        }
    }

    fun readTensor8BitBytes(t: OnnxTensorProto, modelDir: File?): ByteArray? {
        if (t.int32Data.isNotEmpty()) {
            return ByteArray(t.int32Data.size) { idx -> (t.int32Data[idx] and 0xFF).toByte() }
        }
        return readRawOrExternalBytes(t, modelDir, bytesPerElem = 1)
    }

    private fun readRawOrExternalBytes(t: OnnxTensorProto, modelDir: File?, bytesPerElem: Int): ByteArray? {
        t.rawData?.let { if (it.isNotEmpty()) return it.copyOf() }
        if (!t.isExternal() || modelDir == null) return null
        val ext = t.externalMap()
        val loc = ext["location"]?.takeIf { it.isNotBlank() } ?: return null
        val extFile = File(modelDir, loc)
        if (!extFile.exists() || !extFile.isFile) return null
        val offset = ext["offset"]?.toLongOrNull() ?: 0L
        val expectedBytes = ext["length"]?.toLongOrNull()
            ?: (t.elementCount() * bytesPerElem.toLong())
        if (expectedBytes <= 0L || expectedBytes > 64L * 1024L * 1024L) return null
        if (offset < 0L || offset + expectedBytes > extFile.length()) return null
        val out = ByteArray(expectedBytes.toInt())
        RandomAccessFile(extFile, "r").use { raf ->
            raf.seek(offset)
            raf.readFully(out)
        }
        return out
    }
}
