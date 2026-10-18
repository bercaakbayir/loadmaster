//! Minimal GGUF v3 writer for synthetic test fixtures. Output is deterministic: keys are
//! written in insertion order and tensor data, when present, is zero-filled.

use super::{DEFAULT_ALIGNMENT, GgmlType, MAGIC, Value, tensor_bytes};

#[derive(Default)]
pub struct GgufWriter {
    kvs: Vec<(String, Value)>,
    tensors: Vec<(String, Vec<u64>, u32, u64)>,
    next_offset: u64,
    data_bytes: u64,
}

impl GgufWriter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn kv(&mut self, key: &str, value: Value) -> &mut Self {
        self.kvs.push((key.to_owned(), value));
        self
    }

    /// Append a tensor at the next aligned offset.
    ///
    /// # Panics
    /// If `dims` is not a valid shape for `ty` (row length not a multiple of the block size).
    pub fn tensor(&mut self, name: &str, dims: &[u64], ty: GgmlType) -> &mut Self {
        let n_bytes = tensor_bytes(ty, dims)
            .unwrap_or_else(|| panic!("{name}: shape {dims:?} is invalid for {}", ty.name()));
        let offset = self.next_offset;
        self.tensors
            .push((name.to_owned(), dims.to_vec(), ty.id(), offset));
        self.data_bytes = offset + n_bytes;
        self.next_offset = self.data_bytes.next_multiple_of(DEFAULT_ALIGNMENT);
        self
    }

    /// Append a tensor entry verbatim, without validation. For malformed-input tests.
    pub fn tensor_raw(&mut self, name: &str, dims: &[u64], ty_id: u32, offset: u64) -> &mut Self {
        self.tensors
            .push((name.to_owned(), dims.to_vec(), ty_id, offset));
        self
    }

    pub fn finish_header_only(&self) -> Vec<u8> {
        let mut out = Vec::new();
        put_u32(&mut out, MAGIC);
        put_u32(&mut out, 3);
        put_u64(&mut out, self.tensors.len() as u64);
        put_u64(&mut out, self.kvs.len() as u64);
        for (key, value) in &self.kvs {
            put_str(&mut out, key);
            put_u32(&mut out, value_type(value));
            put_value(&mut out, value);
        }
        for (name, dims, ty, offset) in &self.tensors {
            put_str(&mut out, name);
            put_u32(&mut out, dims.len() as u32);
            for &d in dims {
                put_u64(&mut out, d);
            }
            put_u32(&mut out, *ty);
            put_u64(&mut out, *offset);
        }
        let aligned = (out.len() as u64).next_multiple_of(DEFAULT_ALIGNMENT);
        out.resize(aligned as usize, 0);
        out
    }

    /// Header plus zero-filled tensor data, padded to alignment.
    pub fn finish_with_data(&self) -> Vec<u8> {
        let mut out = self.finish_header_only();
        let len = out.len() as u64 + self.data_bytes.next_multiple_of(DEFAULT_ALIGNMENT);
        out.resize(len as usize, 0);
        out
    }
}

fn value_type(v: &Value) -> u32 {
    match v {
        Value::U8(_) => 0,
        Value::I8(_) => 1,
        Value::U16(_) => 2,
        Value::I16(_) => 3,
        Value::U32(_) => 4,
        Value::I32(_) => 5,
        Value::F32(_) => 6,
        Value::Bool(_) => 7,
        Value::String(_) => 8,
        Value::Array(_) | Value::LongArray { .. } => 9,
        Value::U64(_) => 10,
        Value::I64(_) => 11,
        Value::F64(_) => 12,
    }
}

fn put_value(out: &mut Vec<u8>, v: &Value) {
    match v {
        Value::U8(x) => out.push(*x),
        Value::I8(x) => out.extend(x.to_le_bytes()),
        Value::U16(x) => out.extend(x.to_le_bytes()),
        Value::I16(x) => out.extend(x.to_le_bytes()),
        Value::U32(x) => put_u32(out, *x),
        Value::I32(x) => out.extend(x.to_le_bytes()),
        Value::U64(x) => put_u64(out, *x),
        Value::I64(x) => out.extend(x.to_le_bytes()),
        Value::F32(x) => out.extend(x.to_le_bytes()),
        Value::F64(x) => out.extend(x.to_le_bytes()),
        Value::Bool(x) => out.push(u8::from(*x)),
        Value::String(s) => put_str(out, s),
        Value::Array(items) => {
            put_u32(out, items.first().map_or(4, value_type));
            put_u64(out, items.len() as u64);
            for item in items {
                put_value(out, item);
            }
        }
        Value::LongArray { .. } => panic!("LongArray cannot be written; use Value::Array"),
    }
}

fn put_u32(out: &mut Vec<u8>, v: u32) {
    out.extend(v.to_le_bytes());
}

fn put_u64(out: &mut Vec<u8>, v: u64) {
    out.extend(v.to_le_bytes());
}

fn put_str(out: &mut Vec<u8>, s: &str) {
    put_u64(out, s.len() as u64);
    out.extend(s.as_bytes());
}
