//! GGUF header reader. Reads metadata and the tensor table; never touches tensor data.
//!
//! Format reference: <https://github.com/ggml-org/ggml/blob/master/docs/gguf.md>.

mod types;
pub mod writer;

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{self, BufReader, Read};
use std::path::{Path, PathBuf};

pub use types::{GgmlType, file_type_name};

pub const MAGIC: u32 = 0x4655_4747; // "GGUF" little-endian
pub const DEFAULT_ALIGNMENT: u64 = 32;

/// Arrays longer than this are not materialised; only their length is kept. Vocabularies
/// and merge tables are the only long arrays in practice and nothing here reads them.
const MAX_MATERIALISED_ARRAY: u64 = 1024;
const MAX_STRING: u64 = 256 << 20;
const MAX_TENSORS: u64 = 1 << 20;
const MAX_DIMS: u32 = 4;

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    String(String),
    Array(Vec<Value>),
    /// An array whose contents were skipped; see [`MAX_MATERIALISED_ARRAY`].
    LongArray {
        elem_type: u32,
        len: u64,
    },
}

impl Value {
    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            Value::U8(v) => Some(v.into()),
            Value::U16(v) => Some(v.into()),
            Value::U32(v) => Some(v.into()),
            Value::U64(v) => Some(v),
            Value::I8(v) => u64::try_from(v).ok(),
            Value::I16(v) => u64::try_from(v).ok(),
            Value::I32(v) => u64::try_from(v).ok(),
            Value::I64(v) => u64::try_from(v).ok(),
            _ => None,
        }
    }

    pub fn as_u32(&self) -> Option<u32> {
        self.as_u64().and_then(|v| u32::try_from(v).ok())
    }

    pub fn as_f64(&self) -> Option<f64> {
        match *self {
            Value::F32(v) => Some(v.into()),
            Value::F64(v) => Some(v),
            _ => self.as_u64().map(|v| v as f64),
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn array_len(&self) -> Option<u64> {
        match self {
            Value::Array(v) => Some(v.len() as u64),
            Value::LongArray { len, .. } => Some(*len),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TensorInfo {
    pub name: String,
    /// ggml order: `dims[0]` is the contiguous (row) dimension.
    pub dims: Vec<u64>,
    pub ty: GgmlType,
    /// Offset relative to the start of the tensor data section.
    pub offset: u64,
    pub n_bytes: u64,
}

impl TensorInfo {
    pub fn n_elements(&self) -> u64 {
        self.dims.iter().product()
    }
}

/// A parsed GGUF header.
#[derive(Debug, Clone)]
pub struct Gguf {
    pub path: PathBuf,
    pub version: u32,
    pub metadata: BTreeMap<String, Value>,
    pub tensors: Vec<TensorInfo>,
    pub alignment: u64,
    /// Absolute offset of the tensor data section (end of header plus padding).
    pub data_offset: u64,
    pub file_size: u64,
}

impl Gguf {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, Error> {
        let path = path.as_ref();
        let io_err = |source| Error::Io {
            path: path.to_path_buf(),
            source,
        };
        let file = File::open(path).map_err(io_err)?;
        let file_size = file.metadata().map_err(io_err)?.len();
        let mut r = Reader {
            inner: BufReader::with_capacity(1 << 16, file),
            pos: 0,
        };
        let mut gguf = parse(&mut r, file_size).map_err(|e| e.with_path(path))?;
        gguf.path = path.to_path_buf();
        Ok(gguf)
    }

    pub fn parse_bytes(bytes: &[u8]) -> Result<Self, Error> {
        let mut r = Reader {
            inner: bytes,
            pos: 0,
        };
        parse(&mut r, bytes.len() as u64).map_err(|e| e.with_path(Path::new("<memory>")))
    }

    pub fn get(&self, key: &str) -> Option<&Value> {
        self.metadata.get(key)
    }

    pub fn get_u32(&self, key: &str) -> Option<u32> {
        self.get(key).and_then(Value::as_u32)
    }

    pub fn get_str(&self, key: &str) -> Option<&str> {
        self.get(key).and_then(Value::as_str)
    }

    /// Header bytes including alignment padding.
    pub fn header_bytes(&self) -> u64 {
        self.data_offset
    }

    /// Sum of tensor sizes from the tensor table.
    pub fn tensor_bytes(&self) -> u64 {
        self.tensors.iter().map(|t| t.n_bytes).sum()
    }

    /// True when the file is shorter than the tensor table says, e.g. a partial download
    /// or a header-only test fixture.
    pub fn is_truncated(&self) -> bool {
        let end = self
            .tensors
            .iter()
            .map(|t| t.offset + t.n_bytes)
            .max()
            .unwrap_or(0);
        self.file_size < self.data_offset + end
    }
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{path}: {source}")]
    Io { path: PathBuf, source: io::Error },
    #[error("{path}: not a GGUF file (magic {found:#010x}); pass a .gguf model file")]
    BadMagic { path: PathBuf, found: u32 },
    #[error(
        "{path}: GGUF version {version} is not supported (loadmaster reads v2 and v3); re-convert with a current llama.cpp"
    )]
    UnsupportedVersion { path: PathBuf, version: u32 },
    #[error(
        "{path}: file ends inside the GGUF header at byte {offset}; the download is probably incomplete, re-download it"
    )]
    Truncated { path: PathBuf, offset: u64 },
    #[error("{path}: malformed GGUF header at byte {offset}: {reason}")]
    Malformed {
        path: PathBuf,
        offset: u64,
        reason: String,
    },
}

/// Parse errors before the path is known.
#[derive(Debug)]
enum RawError {
    Io(io::Error),
    BadMagic(u32),
    UnsupportedVersion(u32),
    Truncated(u64),
    Malformed(u64, String),
}

impl RawError {
    fn with_path(self, path: &Path) -> Error {
        let path = path.to_path_buf();
        match self {
            RawError::Io(source) => Error::Io { path, source },
            RawError::BadMagic(found) => Error::BadMagic { path, found },
            RawError::UnsupportedVersion(version) => Error::UnsupportedVersion { path, version },
            RawError::Truncated(offset) => Error::Truncated { path, offset },
            RawError::Malformed(offset, reason) => Error::Malformed {
                path,
                offset,
                reason,
            },
        }
    }
}

struct Reader<R> {
    inner: R,
    pos: u64,
}

impl<R: Read> Reader<R> {
    fn bytes<const N: usize>(&mut self) -> Result<[u8; N], RawError> {
        let mut buf = [0u8; N];
        self.inner.read_exact(&mut buf).map_err(|e| self.eof(e))?;
        self.pos += N as u64;
        Ok(buf)
    }

    fn eof(&self, e: io::Error) -> RawError {
        if e.kind() == io::ErrorKind::UnexpectedEof {
            RawError::Truncated(self.pos)
        } else {
            RawError::Io(e)
        }
    }

    fn malformed(&self, reason: impl Into<String>) -> RawError {
        RawError::Malformed(self.pos, reason.into())
    }

    fn u32(&mut self) -> Result<u32, RawError> {
        self.bytes().map(u32::from_le_bytes)
    }

    fn u64(&mut self) -> Result<u64, RawError> {
        self.bytes().map(u64::from_le_bytes)
    }

    fn skip(&mut self, n: u64) -> Result<(), RawError> {
        let copied =
            io::copy(&mut (&mut self.inner).take(n), &mut io::sink()).map_err(RawError::Io)?;
        self.pos += copied;
        if copied < n {
            return Err(RawError::Truncated(self.pos));
        }
        Ok(())
    }

    fn string(&mut self) -> Result<String, RawError> {
        let len = self.u64()?;
        if len > MAX_STRING {
            return Err(self.malformed(format!("string length {len} exceeds limit")));
        }
        let mut buf = vec![0u8; len as usize];
        self.inner.read_exact(&mut buf).map_err(|e| self.eof(e))?;
        self.pos += len;
        String::from_utf8(buf).map_err(|_| self.malformed("string is not valid UTF-8"))
    }

    fn value(&mut self, ty: u32) -> Result<Value, RawError> {
        Ok(match ty {
            0 => Value::U8(u8::from_le_bytes(self.bytes()?)),
            1 => Value::I8(i8::from_le_bytes(self.bytes()?)),
            2 => Value::U16(u16::from_le_bytes(self.bytes()?)),
            3 => Value::I16(i16::from_le_bytes(self.bytes()?)),
            4 => Value::U32(self.u32()?),
            5 => Value::I32(i32::from_le_bytes(self.bytes()?)),
            6 => Value::F32(f32::from_le_bytes(self.bytes()?)),
            7 => match self.bytes::<1>()?[0] {
                0 => Value::Bool(false),
                1 => Value::Bool(true),
                b => return Err(self.malformed(format!("invalid bool {b}"))),
            },
            8 => Value::String(self.string()?),
            9 => {
                let elem_type = self.u32()?;
                let len = self.u64()?;
                if elem_type == 9 {
                    return Err(self.malformed("nested arrays are not supported"));
                }
                if len > MAX_MATERIALISED_ARRAY {
                    self.skip_values(elem_type, len)?;
                    Value::LongArray { elem_type, len }
                } else {
                    let mut items = Vec::with_capacity(len as usize);
                    for _ in 0..len {
                        items.push(self.value(elem_type)?);
                    }
                    Value::Array(items)
                }
            }
            10 => Value::U64(self.u64()?),
            11 => Value::I64(i64::from_le_bytes(self.bytes()?)),
            12 => Value::F64(f64::from_le_bytes(self.bytes()?)),
            t => return Err(self.malformed(format!("unknown metadata value type {t}"))),
        })
    }

    fn skip_values(&mut self, elem_type: u32, len: u64) -> Result<(), RawError> {
        let width = match elem_type {
            0 | 1 | 7 => 1,
            2 | 3 => 2,
            4..=6 => 4,
            10..=12 => 8,
            8 => {
                for _ in 0..len {
                    let n = self.u64()?;
                    if n > MAX_STRING {
                        return Err(self.malformed(format!("string length {n} exceeds limit")));
                    }
                    self.skip(n)?;
                }
                return Ok(());
            }
            t => return Err(self.malformed(format!("unknown array element type {t}"))),
        };
        let n = len
            .checked_mul(width)
            .ok_or_else(|| self.malformed("array size overflows"))?;
        self.skip(n)
    }
}

fn parse<R: Read>(r: &mut Reader<R>, file_size: u64) -> Result<Gguf, RawError> {
    let magic = r.u32()?;
    if magic != MAGIC {
        return Err(RawError::BadMagic(magic));
    }
    let version = r.u32()?;
    if !(2..=3).contains(&version) {
        return Err(RawError::UnsupportedVersion(version));
    }
    let n_tensors = r.u64()?;
    let n_kv = r.u64()?;
    if n_tensors > MAX_TENSORS {
        return Err(r.malformed(format!("tensor count {n_tensors} exceeds limit")));
    }

    let mut metadata = BTreeMap::new();
    for _ in 0..n_kv {
        let key = r.string()?;
        let ty = r.u32()?;
        let value = r.value(ty)?;
        if metadata.insert(key.clone(), value).is_some() {
            return Err(r.malformed(format!("duplicate metadata key `{key}`")));
        }
    }

    let alignment = match metadata.get("general.alignment") {
        None => DEFAULT_ALIGNMENT,
        Some(v) => match v.as_u64() {
            Some(a) if a.is_power_of_two() => a,
            _ => return Err(r.malformed("general.alignment is not a power of two")),
        },
    };

    let mut tensors = Vec::with_capacity(n_tensors as usize);
    for _ in 0..n_tensors {
        let name = r.string()?;
        let n_dims = r.u32()?;
        if n_dims == 0 || n_dims > MAX_DIMS {
            return Err(r.malformed(format!("tensor `{name}` has {n_dims} dimensions")));
        }
        let mut dims = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            dims.push(r.u64()?);
        }
        let ty_id = r.u32()?;
        let ty = GgmlType::from_id(ty_id)
            .ok_or_else(|| r.malformed(format!("tensor `{name}` has unknown ggml type {ty_id}")))?;
        let offset = r.u64()?;
        let n_bytes = tensor_bytes(ty, &dims).ok_or_else(|| {
            r.malformed(format!(
                "tensor `{name}` has invalid shape {dims:?} for {}",
                ty.name()
            ))
        })?;
        tensors.push(TensorInfo {
            name,
            dims,
            ty,
            offset,
            n_bytes,
        });
    }

    let data_offset = r.pos.next_multiple_of(alignment);
    Ok(Gguf {
        path: PathBuf::new(),
        version,
        metadata,
        tensors,
        alignment,
        data_offset,
        file_size,
    })
}

fn tensor_bytes(ty: GgmlType, dims: &[u64]) -> Option<u64> {
    let row = ty.row_bytes(dims[0])?;
    dims[1..].iter().try_fold(row, |acc, &d| acc.checked_mul(d))
}

#[cfg(test)]
mod tests {
    use super::writer::GgufWriter;
    use super::*;

    fn sample() -> Vec<u8> {
        let mut w = GgufWriter::new();
        w.kv("general.architecture", Value::String("llama".into()));
        w.kv("llama.block_count", Value::U32(2));
        w.kv(
            "tokenizer.ggml.tokens",
            Value::Array((0..2000).map(|i| Value::String(format!("t{i}"))).collect()),
        );
        w.tensor("token_embd.weight", &[256, 10], GgmlType::Q4_K);
        w.tensor("blk.0.attn_q.weight", &[256, 256], GgmlType::F16);
        w.finish_with_data()
    }

    #[test]
    fn parses_written_file() {
        let bytes = sample();
        let g = Gguf::parse_bytes(&bytes).unwrap();
        assert_eq!(g.version, 3);
        assert_eq!(g.get_str("general.architecture"), Some("llama"));
        assert_eq!(g.get_u32("llama.block_count"), Some(2));
        assert_eq!(
            g.get("tokenizer.ggml.tokens").and_then(Value::array_len),
            Some(2000)
        );
        assert_eq!(g.tensors.len(), 2);
        assert_eq!(g.tensors[0].n_bytes, 144 * 10);
        assert_eq!(g.tensors[1].n_bytes, 256 * 256 * 2);
        assert_eq!(g.tensors[1].offset % g.alignment, 0);
        assert!(!g.is_truncated());
        assert_eq!(
            g.data_offset + g.tensor_bytes(),
            bytes.len() as u64 - padding_tail(&g)
        );
    }

    fn padding_tail(g: &Gguf) -> u64 {
        // the writer pads the final tensor to alignment, as llama.cpp does
        let last = g.tensors.last().unwrap();
        (last.offset + last.n_bytes).next_multiple_of(g.alignment) - (last.offset + last.n_bytes)
    }

    #[test]
    fn header_only_is_truncated() {
        let mut w = GgufWriter::new();
        w.tensor("blk.0.attn_q.weight", &[256, 256], GgmlType::F16);
        let g = Gguf::parse_bytes(&w.finish_header_only()).unwrap();
        assert!(g.is_truncated());
    }

    #[test]
    fn rejects_bad_magic() {
        let err = Gguf::parse_bytes(b"GGML\x03\0\0\0").unwrap_err();
        assert!(matches!(err, Error::BadMagic { .. }), "{err}");
    }

    #[test]
    fn short_header_is_truncated_error() {
        let bytes = sample();
        let err = Gguf::parse_bytes(&bytes[..40]).unwrap_err();
        assert!(matches!(err, Error::Truncated { .. }), "{err}");
        assert!(err.to_string().contains("re-download"));
    }

    #[test]
    fn rejects_bad_shape() {
        let mut w = GgufWriter::new();
        w.tensor_raw("x", &[100], GgmlType::Q4_K.id(), 0);
        let err = Gguf::parse_bytes(&w.finish_header_only()).unwrap_err();
        assert!(err.to_string().contains("invalid shape"), "{err}");
    }
}
