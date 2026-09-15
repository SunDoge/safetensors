"""Real MLX round trips plus host-buffer ownership and Metal aliasing checks."""

import gc
import mmap
import os
import sys

import numpy as np
import pytest
from safetensors.numpy import save_file

from safetensors import TensorSpec, safe_open, serialize_file

mx = pytest.importorskip("mlx.core")
pytestmark = pytest.mark.skipif(
    os.name != "posix" or sys.byteorder != "little" or not hasattr(mx, "asarray"),
    reason="Aligned MLX buffers require little-endian Unix and mx.asarray",
)


def capture_buffers(monkeypatch, *, require_no_copy=False):
    original = mx.asarray
    buffers = []

    def asarray(value):
        buffers.append(value)
        if require_no_copy:
            return original(value, copy=False)
        return original(value)

    monkeypatch.setattr(mx, "asarray", asarray)
    return buffers


@pytest.mark.parametrize("backend", ["mmap", "pread"])
@pytest.mark.parametrize(
    "dtype",
    [
        "bool",
        "int8",
        "uint8",
        "int16",
        "uint16",
        "int32",
        "uint32",
        "int64",
        "uint64",
        "float16",
        "float32",
        "float64",
        "complex64",
    ],
)
@pytest.mark.parametrize("shape", [(), (0, 3), (3, 5), (17001,)])
def test_roundtrip(tmp_path, backend, dtype, shape):
    if dtype == "float64" and not hasattr(mx, "float64"):
        pytest.skip("This MLX version does not support float64")
    expected = (np.arange(np.prod(shape, dtype=int)) % 17).astype(dtype).reshape(shape)
    path = tmp_path / "tensor.safetensors"
    save_file({"x": expected}, path)
    with safe_open(path, framework="mlx", backend=backend) as f:
        result = f.get_tensors()["x"]
    del f
    gc.collect()
    np.testing.assert_array_equal(np.asarray(result), expected)
    assert np.asarray(result).dtype == expected.dtype


@pytest.mark.parametrize("backend", ["mmap", "pread"])
def test_buffer_alignment_ownership_and_write_isolation(tmp_path, monkeypatch, backend):
    expected = np.arange(17001, dtype=np.float32)
    path = tmp_path / "tensor.safetensors"
    save_file({"x": expected}, path)
    original_file = path.read_bytes()
    buffers = capture_buffers(monkeypatch)
    with safe_open(path, framework="mlx", backend=backend) as f:
        first = f.get_tensor("x")
        second = f.get_tensor("x")
    del f
    gc.collect()

    assert len(buffers) == 2
    for buffer in buffers:
        assert buffer.ctypes.data % mmap.PAGESIZE == 0
        assert buffer.nbytes % mmap.PAGESIZE == 0
        assert not buffer.flags.owndata
        assert buffer.flags.writeable
        assert buffer.ctypes.data == buffer.base.__array_interface__["data"][0]
    assert not np.shares_memory(*buffers)
    np.testing.assert_array_equal(np.asarray(first), expected)
    np.testing.assert_array_equal(np.asarray(second), expected)

    untouched = buffers[1].copy()
    buffers[0][:] = 0
    np.testing.assert_array_equal(buffers[1], untouched)
    assert path.read_bytes() == original_file


def test_mmap_uses_original_file_handle(tmp_path):
    path = tmp_path / "tensor.safetensors"
    replacement = tmp_path / "replacement.safetensors"
    expected = np.arange(15, dtype=np.float32)
    save_file({"x": expected}, path)
    save_file({"x": expected + 100}, replacement)
    with safe_open(path, framework="mlx") as f:
        os.replace(replacement, path)
        result = f.get_tensor("x")
    np.testing.assert_array_equal(np.asarray(result), expected)


@pytest.mark.parametrize("backend", ["mmap", "pread"])
def test_bfloat16_without_numpy_dtype(tmp_path, backend):
    bits = np.array([0x3F80, 0x4000, 0xC040], dtype=np.uint16)
    path = tmp_path / "bf16.safetensors"
    serialize_file(
        {
            "x": TensorSpec(
                dtype="bfloat16",
                shape=[3],
                data_ptr=bits.ctypes.data,
                data_len=bits.nbytes,
            )
        },
        path,
    )
    with safe_open(path, framework="mlx", backend=backend) as f:
        result = f.get_tensor("x")
    assert result.dtype == mx.bfloat16
    np.testing.assert_array_equal(np.asarray(result.astype(mx.float32)), [1, 2, -3])


@pytest.mark.parametrize("backend", ["mmap", "pread"])
def test_old_mlx_fallback(tmp_path, monkeypatch, backend):
    expected = np.arange(12, dtype=np.float32).reshape(3, 4)
    path = tmp_path / "tensor.safetensors"
    save_file({"x": expected}, path)
    monkeypatch.delattr(mx, "asarray")
    with safe_open(path, framework="mlx", backend=backend) as f:
        result = f.get_tensor("x")
        sliced = f.get_slice("x")[:, 1:3]
    np.testing.assert_array_equal(np.asarray(result), expected)
    np.testing.assert_array_equal(np.asarray(sliced), expected[:, 1:3])


def test_import_errors_are_not_silently_copied(tmp_path, monkeypatch):
    path = tmp_path / "tensor.safetensors"
    save_file({"x": np.arange(3, dtype=np.float32)}, path)

    def fail(value):
        raise RuntimeError("MLX import failed")

    monkeypatch.setattr(mx, "asarray", fail)
    with (
        safe_open(path, framework="mlx") as f,
        pytest.raises(RuntimeError, match="MLX import failed"),
    ):
        f.get_tensor("x")


@pytest.mark.skipif(not mx.metal.is_available(), reason="Requires Metal")
@pytest.mark.parametrize("backend", ["mmap", "pread"])
def test_metal_shares_storage_after_close_and_gpu_eval(tmp_path, monkeypatch, backend):
    # Distinguish an older MLX without host adoption from a broken loader.
    probe_owner = mmap.mmap(-1, mmap.PAGESIZE)
    probe = np.frombuffer(probe_owner, dtype=np.uint8)
    try:
        mx.asarray(probe, copy=False)
    except (TypeError, ValueError):
        pytest.skip("This MLX version cannot adopt aligned host buffers")

    expected = np.arange(17001, dtype=np.float32)
    path = tmp_path / "tensor.safetensors"
    save_file({"x": expected}, path)
    buffers = capture_buffers(monkeypatch, require_no_copy=True)
    with safe_open(path, framework="mlx", backend=backend) as f:
        result = f.get_tensor("x")
    del f
    mx.eval(result)
    assert np.shares_memory(buffers[0], np.asarray(result))
    buffers.clear()
    gc.collect()

    output = mx.add(result, 1, stream=mx.gpu)
    mx.eval(output)
    np.testing.assert_array_equal(np.asarray(output), expected + 1)
    np.testing.assert_array_equal(np.asarray(result), expected)
