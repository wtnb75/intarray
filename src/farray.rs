use crate::core::PackedArrayCore;
use crate::error::ArrayError;
use log::debug;
use serde::de::{Deserialize, Deserializer, SeqAccess, Visitor};
use serde::ser::{Serialize, SerializeSeq, Serializer};
use std::{fmt, mem};

/// IEEE 754 half precision (exp=5, man=10, total=16)
pub const FLOAT16: (usize, usize) = (5, 10);
/// Brain Float 16 (exp=8, man=7, total=16)
pub const BFLOAT16: (usize, usize) = (8, 7);
/// IEEE 754 single precision (exp=8, man=23, total=32)
pub const FLOAT32: (usize, usize) = (8, 23);
/// IEEE 754 double precision (exp=11, man=52, total=64)
pub const FLOAT64: (usize, usize) = (11, 52);

/// Packed array of custom-precision floating-point values.
///
/// Each element uses `1 + exp_bits + man_bits` bits, packed into a `Vec<u64>`.
/// The interface type is `f64`. Denormal values are flushed to zero on encode.
#[derive(Clone, Debug, PartialEq)]
pub struct FloatArray {
    exp_bits: usize,
    man_bits: usize,
    total: usize,
    bpu: usize,
    bias: u64,
    exp_max: u64,
    length: usize,
    data: Vec<u64>,
}

/// Iterator for [`FloatArray`]
pub struct FloatIter<'a> {
    arr: &'a FloatArray,
    idx: usize,
    len: usize,
}

fn validate_params(exp_bits: usize, man_bits: usize) -> Result<(), ArrayError> {
    if !(2..=11).contains(&exp_bits)
        || !(1..=52).contains(&man_bits)
        || 1 + exp_bits + man_bits > 64
    {
        return Err(ArrayError::InvalidRange);
    }
    Ok(())
}

fn encode_float(v: f64, exp_bits: usize, man_bits: usize, bias: u64, exp_max: u64) -> u64 {
    let bits = v.to_bits();
    let sign = bits >> 63;
    let f64_exp = (bits >> 52) & 0x7FF;
    let f64_man = bits & ((1u64 << 52) - 1);

    if f64_exp == 0x7FF {
        let custom_man = if f64_man == 0 { 0u64 } else { 1u64 };
        (sign << (exp_bits + man_bits)) | (exp_max << man_bits) | custom_man
    } else if f64_exp == 0 {
        // Zero or denormal → flush to ±0
        sign << (exp_bits + man_bits)
    } else {
        let exp_biased = f64_exp as i64 - 1023 + bias as i64;
        if exp_biased >= exp_max as i64 {
            // Overflow → ±Inf
            (sign << (exp_bits + man_bits)) | (exp_max << man_bits)
        } else if exp_biased <= 0 {
            // Underflow → ±0
            sign << (exp_bits + man_bits)
        } else {
            let custom_exp = exp_biased as u64;
            let custom_man = f64_man >> (52 - man_bits);
            (sign << (exp_bits + man_bits)) | (custom_exp << man_bits) | custom_man
        }
    }
}

fn decode_float(raw: u64, exp_bits: usize, man_bits: usize, bias: u64, exp_max: u64) -> f64 {
    let sign = raw >> (exp_bits + man_bits);
    let custom_exp = (raw >> man_bits) & exp_max;
    let custom_man = raw & ((1u64 << man_bits) - 1);

    if custom_exp == exp_max {
        let f64_man = if custom_man == 0 { 0u64 } else { 1u64 << 51 };
        f64::from_bits((sign << 63) | (0x7FFu64 << 52) | f64_man)
    } else if custom_exp == 0 {
        f64::from_bits(sign << 63)
    } else {
        let f64_exp = custom_exp + 1023 - bias;
        let f64_man = custom_man << (52 - man_bits);
        f64::from_bits((sign << 63) | (f64_exp << 52) | f64_man)
    }
}

impl FloatArray {
    fn word_mask(&self) -> u64 {
        if self.total == 64 {
            u64::MAX
        } else {
            (1u64 << self.total) - 1
        }
    }

    fn raw_get_encoded(&self, i: usize) -> u64 {
        let wi = i / self.bpu;
        let bit_offset = (i % self.bpu) * self.total;
        (self.data[wi] >> bit_offset) & self.word_mask()
    }

    fn raw_set_encoded(&mut self, i: usize, encoded: u64) {
        let wi = i / self.bpu;
        let bit_offset = (i % self.bpu) * self.total;
        let mask = self.word_mask();
        self.data[wi] = (self.data[wi] & !(mask << bit_offset)) | (encoded << bit_offset);
    }

    fn resize(&mut self, len: usize) {
        self.length = len;
        self.data.resize(len.div_ceil(self.bpu), 0);
    }

    /// Creates a new `FloatArray` with all elements initialized to `+0.0`.
    pub fn new(exp_bits: usize, man_bits: usize, len: usize) -> Result<Self, ArrayError> {
        validate_params(exp_bits, man_bits)?;
        let total = 1 + exp_bits + man_bits;
        let bpu = 64 / total;
        let bias = (1u64 << (exp_bits - 1)) - 1;
        let exp_max = (1u64 << exp_bits) - 1;
        let word_count = len.div_ceil(bpu);
        debug!(
            "FloatArray::new exp_bits={} man_bits={} total={} bpu={} bias={} exp_max={}",
            exp_bits, man_bits, total, bpu, bias, exp_max
        );
        Ok(FloatArray {
            exp_bits,
            man_bits,
            total,
            bpu,
            bias,
            exp_max,
            length: len,
            data: vec![0u64; word_count],
        })
    }

    /// Creates from a `Vec<f64>`.
    pub fn new_with_vec(
        exp_bits: usize,
        man_bits: usize,
        vals: Vec<f64>,
    ) -> Result<Self, ArrayError> {
        let mut arr = Self::new(exp_bits, man_bits, 0)?;
        arr.extend(vals)?;
        Ok(arr)
    }

    /// Creates from an iterator.
    pub fn new_with_iter(
        exp_bits: usize,
        man_bits: usize,
        vals: impl IntoIterator<Item = f64>,
    ) -> Result<Self, ArrayError> {
        let mut arr = Self::new(exp_bits, man_bits, 0)?;
        arr.extend(vals)?;
        Ok(arr)
    }

    pub fn get(&self, i: usize) -> Result<f64, ArrayError> {
        if i >= self.length {
            return Err(ArrayError::OutOfBounds);
        }
        let raw = self.raw_get_encoded(i);
        Ok(decode_float(
            raw,
            self.exp_bits,
            self.man_bits,
            self.bias,
            self.exp_max,
        ))
    }

    pub fn set(&mut self, i: usize, v: f64) -> Result<(), ArrayError> {
        if i >= self.length {
            return Err(ArrayError::OutOfBounds);
        }
        let encoded = encode_float(v, self.exp_bits, self.man_bits, self.bias, self.exp_max);
        self.raw_set_encoded(i, encoded);
        Ok(())
    }

    pub fn push(&mut self, v: f64) -> Result<usize, ArrayError> {
        PackedArrayCore::core_push(self, v)
    }

    pub fn pop(&mut self) -> Result<f64, ArrayError> {
        PackedArrayCore::core_pop(self)
    }

    pub fn extend(&mut self, vals: impl IntoIterator<Item = f64>) -> Result<(), ArrayError> {
        PackedArrayCore::core_extend(self, vals)
    }

    /// Appends all elements from `other`.
    ///
    /// Fast path when formats match and `self` is word-aligned:
    /// raw words are copied directly without per-element get/set.
    pub fn extend_array(&mut self, other: &FloatArray) -> Result<(), ArrayError> {
        if self.exp_bits == other.exp_bits
            && self.man_bits == other.man_bits
            && self.length.is_multiple_of(self.bpu)
        {
            debug!("fast path: total={}, length={}", self.total, self.length);
            self.length += other.length;
            self.data.extend_from_slice(&other.data);
            return Ok(());
        }
        debug!("slow path: total={}, length={}", self.total, self.length);
        self.extend(other.iter())
    }

    pub fn len(&self) -> usize {
        self.length
    }

    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    pub fn exp_bits(&self) -> usize {
        self.exp_bits
    }

    pub fn man_bits(&self) -> usize {
        self.man_bits
    }

    pub fn capacity(&self) -> usize {
        self.data.len() * self.bpu
    }

    pub fn datasize(&self) -> usize {
        mem::size_of::<FloatArray>() + mem::size_of::<u64>() * self.data.capacity()
    }

    pub fn iter(&self) -> FloatIter<'_> {
        FloatIter {
            arr: self,
            idx: 0,
            len: self.length,
        }
    }

    pub fn sum(&self) -> Option<f64> {
        if self.length == 0 {
            return None;
        }
        Some(self.iter().fold(0.0f64, |acc, v| acc + v))
    }

    pub fn min(&self) -> Option<f64> {
        let mut it = self.iter();
        let first = it.next()?;
        Some(it.fold(first, f64::min))
    }

    pub fn max(&self) -> Option<f64> {
        let mut it = self.iter();
        let first = it.next()?;
        Some(it.fold(first, f64::max))
    }

    pub fn average(&self) -> Option<f64> {
        if self.length == 0 {
            return None;
        }
        Some(self.sum().unwrap() / self.length as f64)
    }
}

impl PackedArrayCore for FloatArray {
    type Item = f64;

    fn raw_get(&self, i: usize) -> Result<f64, ArrayError> {
        self.get(i)
    }

    fn raw_set(&mut self, i: usize, v: f64) -> Result<(), ArrayError> {
        self.set(i, v)
    }

    fn raw_len(&self) -> usize {
        self.length
    }

    fn raw_resize(&mut self, len: usize) {
        self.resize(len)
    }
}

impl<'a> Iterator for FloatIter<'a> {
    type Item = f64;

    fn next(&mut self) -> Option<f64> {
        if self.idx >= self.len {
            return None;
        }
        let v = self.arr.get(self.idx).unwrap();
        self.idx += 1;
        Some(v)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.len - self.idx;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for FloatIter<'_> {}

impl fmt::Display for FloatArray {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "[e{}m{}][{}]=",
            self.exp_bits, self.man_bits, self.length
        )?;
        let s = self
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(",");
        write!(f, "{}", s)
    }
}

impl Serialize for FloatArray {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut seq = serializer.serialize_seq(Some(self.length))?;
        for v in self.iter() {
            seq.serialize_element(&v)?;
        }
        seq.end()
    }
}

struct FloatArrayVisitor;

impl<'de> Visitor<'de> for FloatArrayVisitor {
    type Value = FloatArray;

    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "a sequence of floating-point numbers")
    }

    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let (exp_bits, man_bits) = FLOAT64;
        let mut arr = FloatArray::new(exp_bits, man_bits, 0)
            .map_err(|e| serde::de::Error::custom(e.to_string()))?;
        while let Some(v) = seq.next_element::<f64>()? {
            arr.push(v)
                .map_err(|e| serde::de::Error::custom(e.to_string()))?;
        }
        Ok(arr)
    }
}

/// Deserializes from a flat sequence of `f64` values.
///
/// **Note:** `exp_bits` and `man_bits` are not preserved.
/// The array is reconstructed with FLOAT64 precision (exp=11, man=52).
impl<'de> Deserialize<'de> for FloatArray {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_seq(FloatArrayVisitor)
    }
}

#[cfg(test)]
mod tests;
