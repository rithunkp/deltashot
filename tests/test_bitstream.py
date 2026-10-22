import pytest

from scripts.bitstream import (
    compressai_stream_batch,
    decode_compressai_bitstream,
    decode_mode_bitstream,
    read_mode_bitstream,
    read_neural_bitstream,
    write_mode_bitstream,
    write_neural_bitstream,
)


def test_neural_streams_restore_compressai_single_image_batch_shape():
    assert compressai_stream_batch((b"encoded-y", b"encoded-z")) == [
        [b"encoded-y"], [b"encoded-z"],
    ]


def test_reference_decode_uses_explicit_wrapper_method_and_saved_streams():
    class BaseCodecStub:
        def decompress(self, strings, shape):
            raise AssertionError("plain base codec must not receive a reference")

    class ReferenceCodecStub(BaseCodecStub):
        def decompress_with_reference(self, strings, shape, reference):
            return strings, shape, reference

    payload = {"streams": (b"saved-y", b"saved-z"), "latent_shape": (8, 8)}
    reference = object()

    restored = decode_compressai_bitstream(ReferenceCodecStub(), payload, reference)

    assert restored == ([[b"saved-y"], [b"saved-z"]], (8, 8), reference)


def test_reference_decode_rejects_plain_scale_hyperprior():
    class BaseCodecStub:
        def decompress(self, strings, shape):
            return strings, shape

    payload = {"streams": (b"y", b"z"), "latent_shape": (2, 2)}
    with pytest.raises(TypeError, match="ReferenceConditionedCodec"):
        decode_compressai_bitstream(BaseCodecStub(), payload, object())


@pytest.mark.parametrize("codec", ["noref", "reference_conditioned"])
def test_mode_decode_uses_codec_recorded_in_container_even_when_reference_is_present(codec):
    calls = []

    class BaseCodecStub:
        def decompress(self, strings, shape):
            calls.append(("noref", strings, shape))
            return "noref-output"

    class ReferenceCodecStub:
        def decompress_with_reference(self, strings, shape, reference):
            calls.append(("reference_conditioned", strings, shape, reference))
            return "reference-output"

    reference = object()
    payload = {"codec": codec, "streams": (b"y", b"z"), "latent_shape": (2, 2)}

    decoded = decode_mode_bitstream(
        {"noref": BaseCodecStub(), "reference_conditioned": ReferenceCodecStub()},
        payload,
        reference,
    )

    expected_codec = codec
    assert calls[0][0] == expected_codec
    assert calls[0][1:3] == ([[b"y"], [b"z"]], (2, 2))
    if codec == "noref":
        assert decoded == "noref-output"
        assert len(calls[0]) == 3
    else:
        assert decoded == "reference-output"
        assert calls[0][3] is reference


def test_mode_decode_rejects_reference_mode_without_reference():
    payload = {"codec": "reference_conditioned", "streams": (b"y", b"z"), "latent_shape": (2, 2)}
    with pytest.raises(ValueError, match="requires its reference"):
        decode_mode_bitstream({"reference_conditioned": object()}, payload)


@pytest.mark.parametrize("codec", ["noref", "reference_conditioned"])
def test_neural_bitstream_round_trips_file_metadata_and_streams(tmp_path, codec):
    path = tmp_path / "codec.bin"
    write_neural_bitstream(
        path, codec, (512, 512), (512, 512), (8, 8), (b"encoded-y", b"encoded-z"),
    )

    restored = read_neural_bitstream(path)

    assert restored == {
        "codec": codec,
        "original_size": (512, 512),
        "padded_size": (512, 512),
        "latent_shape": (8, 8),
        "streams": (b"encoded-y", b"encoded-z"),
    }


def test_mode_bitstream_preserves_mode_and_embedded_neural_stream(tmp_path):
    neural = tmp_path / "noref.bin"
    mode = tmp_path / "mode.bin"
    write_neural_bitstream(neural, "noref", (64, 64), (64, 64), (1, 1), (b"y", b"z"))
    write_mode_bitstream(mode, "noref", neural.read_bytes())

    restored = read_mode_bitstream(mode)

    assert restored["codec"] == "noref"
    assert restored["streams"] == (b"y", b"z")
    assert mode.stat().st_size == neural.stat().st_size + 5


def test_neural_bitstream_rejects_truncation(tmp_path):
    path = tmp_path / "truncated.bin"
    write_neural_bitstream(path, "noref", (64, 64), (64, 64), (1, 1), (b"y", b"z"))
    path.write_bytes(path.read_bytes()[:-1])

    with pytest.raises(ValueError, match="length"):
        read_neural_bitstream(path)
