"""On-disk containers for actual DeltaShot entropy-coded streams."""
from __future__ import annotations

import struct
from pathlib import Path


_NEURAL_HEADER = struct.Struct(">4sBHHHHHHII")
_NEURAL_MAGIC = b"DSC1"
_MODE_MAGIC = b"DSM1"
CODEC_IDS = {"noref": 1, "reference_conditioned": 2}
CODEC_NAMES = {value: key for key, value in CODEC_IDS.items()}


def write_neural_bitstream(path, codec, original_size, padded_size, latent_shape, streams):
    if codec not in CODEC_IDS:
        raise ValueError(f"unsupported neural codec: {codec}")
    if len(streams) != 2:
        raise ValueError("CompressAI hyperprior codecs must provide y and z streams")
    width, height = original_size
    padded_width, padded_height = padded_size
    latent_height, latent_width = latent_shape
    y_stream, z_stream = (bytes(stream) for stream in streams)
    header = _NEURAL_HEADER.pack(
        _NEURAL_MAGIC, CODEC_IDS[codec], width, height, padded_width,
        padded_height, latent_width, latent_height, len(y_stream), len(z_stream),
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(header + y_stream + z_stream)
    return destination


def read_neural_bitstream(source):
    data = Path(source).read_bytes() if isinstance(source, (str, Path)) else bytes(source)
    if len(data) < _NEURAL_HEADER.size:
        raise ValueError("truncated DeltaShot neural bitstream")
    (magic, codec_id, width, height, padded_width, padded_height,
     latent_width, latent_height, y_length, z_length) = _NEURAL_HEADER.unpack_from(data)
    if magic != _NEURAL_MAGIC or codec_id not in CODEC_NAMES:
        raise ValueError("unsupported DeltaShot neural bitstream header")
    offset = _NEURAL_HEADER.size
    if len(data) != offset + y_length + z_length:
        raise ValueError("DeltaShot neural bitstream length does not match its header")
    y_stream = data[offset:offset + y_length]
    z_stream = data[offset + y_length:]
    return {
        "codec": CODEC_NAMES[codec_id],
        "original_size": (width, height),
        "padded_size": (padded_width, padded_height),
        "latent_shape": (latent_height, latent_width),
        "streams": (y_stream, z_stream),
    }


def compressai_stream_batch(streams):
    """Restore one-image streams to CompressAI's list-of-batches API shape."""
    if len(streams) != 2:
        raise ValueError("CompressAI hyperprior bitstreams must contain y and z streams")
    return [[bytes(stream)] for stream in streams]


def decode_compressai_bitstream(model, container, reference=None):
    """Decode a one-image container, keeping side information off base codecs."""
    streams = compressai_stream_batch(container["streams"])
    if reference is None:
        return model.decompress(streams, container["latent_shape"])
    decode_with_reference = getattr(model, "decompress_with_reference", None)
    if decode_with_reference is None:
        raise TypeError("reference decoding requires a ReferenceConditionedCodec wrapper")
    return decode_with_reference(streams, container["latent_shape"], reference)


def decode_mode_bitstream(models, container, reference=None):
    """Decode the selected codec recorded in a mode container."""
    codec = container.get("codec")
    if codec not in CODEC_IDS:
        raise ValueError(f"unsupported mode-selected codec: {codec}")
    model = models[codec]
    if codec == "noref":
        return decode_compressai_bitstream(model, container)
    if reference is None:
        raise ValueError("reference-conditioned mode bitstream requires its reference image")
    return decode_compressai_bitstream(model, container, reference)


def write_mode_bitstream(path, codec, neural_bitstream):
    if codec not in CODEC_IDS:
        raise ValueError(f"unsupported neural codec: {codec}")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(_MODE_MAGIC + bytes((CODEC_IDS[codec],)) + bytes(neural_bitstream))
    return destination


def read_mode_bitstream(source):
    data = Path(source).read_bytes() if isinstance(source, (str, Path)) else bytes(source)
    if len(data) <= 5 or data[:4] != _MODE_MAGIC or data[4] not in CODEC_NAMES:
        raise ValueError("unsupported or truncated DeltaShot mode-selection bitstream")
    codec = CODEC_NAMES[data[4]]
    inner = read_neural_bitstream(data[5:])
    if inner["codec"] != codec:
        raise ValueError("mode flag and contained neural codec do not match")
    return inner
