//! MLX loading through owned, page-aligned host buffers.
//!
//! Recent MLX versions can wrap these buffers with Metal through `asarray`.
//! Older versions and CPU-only backends may copy. Import the whole aligned
//! byte buffer before taking a typed view: tensor offsets and lengths need
//! not satisfy Metal's page alignment requirements themselves.

use std::fs::File;

use memmap2::{MmapMut, MmapOptions};
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyDict, PySlice};
use safetensors::tensor::{Dtype, TensorInfo};

use crate::{read_exact_at, Backend, SafetensorError};

/// NumPy retains this object as its array's base; MLX retains the NumPy owner
/// when adopting its storage. Rust never accesses the bytes after export.
#[pyclass(frozen)]
struct MlxBuffer {
    _mapping: MmapMut,
    ptr: usize,
    len: usize,
}

#[pymethods]
impl MlxBuffer {
    #[getter]
    fn __array_interface__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let interface = PyDict::new(py);
        interface.set_item("version", 3)?;
        interface.set_item("shape", (self.len,))?;
        interface.set_item("typestr", "|u1")?;
        interface.set_item("data", (self.ptr, false))?;
        Ok(interface)
    }
}

fn dtype_name(dtype: Dtype) -> Option<&'static str> {
    Some(match dtype {
        Dtype::BOOL => "bool_",
        Dtype::U8 => "uint8",
        Dtype::I8 => "int8",
        Dtype::U16 => "uint16",
        Dtype::I16 => "int16",
        Dtype::U32 => "uint32",
        Dtype::I32 => "int32",
        Dtype::U64 => "uint64",
        Dtype::I64 => "int64",
        Dtype::F16 => "float16",
        Dtype::BF16 => "bfloat16",
        Dtype::F32 => "float32",
        Dtype::F64 => "float64",
        Dtype::C64 => "complex64",
        _ => return None,
    })
}

pub(crate) fn load_tensor(
    py: Python<'_>,
    file: &File,
    backend: Backend,
    data_offset: usize,
    info: &TensorInfo,
) -> PyResult<Option<Py<PyAny>>> {
    let core = PyModule::import(py, "mlx.core")?;
    let Some(dtype_name) = dtype_name(info.dtype) else {
        return Ok(None);
    };
    if !core.hasattr("asarray")? || !core.hasattr(dtype_name)? {
        return Ok(None);
    }
    let dtype = core.getattr(dtype_name)?;
    let (begin, end) = info.data_offsets;
    let nbytes = end - begin;
    if nbytes == 0 {
        let kwargs = [("dtype", dtype)].into_py_dict(py)?;
        return Ok(Some(
            core.call_method("zeros", (info.shape.clone(),), Some(&kwargs))?
                .unbind(),
        ));
    }

    // SAFETY: sysconf takes no pointers and _SC_PAGESIZE is a valid key.
    let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if page_size <= 0 {
        return Err(SafetensorError::new_err("Could not determine page size"));
    }
    let page_size = page_size as usize;
    let file_offset = data_offset + begin;
    let prefix = if backend == Backend::Mmap {
        file_offset % page_size
    } else {
        0
    };
    let len = prefix
        .checked_add(nbytes)
        .and_then(|n| n.checked_add(page_size - 1))
        .map(|n| n / page_size * page_size)
        .filter(|&n| n <= isize::MAX as usize)
        .ok_or_else(|| SafetensorError::new_err("MLX buffer size overflow"))?;
    // MLX dimensions are int32; the generic typed path can still represent
    // tensors whose byte count does not fit in a single uint8 dimension.
    if len > i32::MAX as usize {
        return Ok(None);
    }
    let mut mapping = if backend == Backend::Mmap {
        // SAFETY: offsets were validated against this open file. On Unix the
        // final partial file page is zero-filled; rounding to its end never
        // maps a page wholly beyond EOF. Each load gets an independent private
        // writable map, so framework writes cannot affect the file, another
        // returned tensor, or the read-only metadata/slicing map. As with the
        // existing mmap backend, callers must not truncate the file in use.
        unsafe {
            MmapOptions::new()
                .offset((file_offset - prefix) as u64)
                .len(len)
                .map_copy(file)?
        }
    } else {
        let mut mapping = MmapOptions::new().len(len).map_anon()?;
        py.detach(|| read_exact_at(file, &mut mapping[..nbytes], file_offset as u64))?;
        mapping
    };
    let ptr = mapping.as_mut_ptr() as usize;
    let owner = Py::new(
        py,
        MlxBuffer {
            _mapping: mapping,
            ptr,
            len,
        },
    )?;
    let numpy = PyModule::import(py, "numpy")?;
    let bytes = numpy.call_method1("asarray", (owner,))?;
    // No copy keyword: older asarray implementations do not accept it.
    // Current MLX defaults to sharing where possible and copying otherwise.
    let array = core.call_method1("asarray", (bytes,))?;
    let tensor = array
        .get_item(PySlice::new(
            py,
            prefix as isize,
            (prefix + nbytes) as isize,
            1,
        ))?
        .call_method1("view", (dtype,))?
        .call_method1("reshape", (info.shape.clone(),))?;
    Ok(Some(tensor.unbind()))
}
