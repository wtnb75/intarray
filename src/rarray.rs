use crate::core::PackedArrayCore;
use crate::error::ArrayError;
use log::debug;
use serde::de::{Deserialize, Deserializer, SeqAccess, Visitor};
use serde::ser::{Serialize, SerializeSeq, Serializer};
use std::{fmt, mem};

pub struct RadixIter<'a> {
    arr: &'a RadixArray,
    idx: usize,
    len: usize,
}

impl<'a> RadixIter<'a> {
    pub(crate) fn new(arr: &'a RadixArray, len: usize) -> Self {
        RadixIter { arr, idx: 0, len }
    }
}

impl Iterator for RadixIter<'_> {
    type Item = i64;

    fn next(&mut self) -> Option<i64> {
        if self.idx >= self.len {
            return None;
        }
        let val = self.arr.get(self.idx).unwrap();
        self.idx += 1;
        Some(val)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.len - self.idx;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for RadixIter<'_> {}

/// Packed integer array for values in the range [A, B].
///
/// Uses mixed-radix encoding: each u64 word stores `epu` elements as digits
/// in base K = B − A + 1.
#[derive(Clone, Debug, PartialEq)]
pub struct RadixArray {
    base: u64,
    offset: i64,
    max_val: i64,
    epu: usize,
    powers: Vec<u64>,
    length: usize,
    data: Vec<u64>,
}

fn validate_range(a: i64, b: i64) -> Result<u64, ArrayError> {
    let k = b as i128 - a as i128 + 1;
    if k < 2 || k > u64::MAX as i128 {
        return Err(ArrayError::InvalidRange);
    }
    Ok(k as u64)
}

/// Returns the largest `epu` such that `base^epu <= 2^64`.
pub(crate) fn calc_epu(base: u64) -> usize {
    debug_assert!(base >= 2);
    let limit: u128 = 1u128 << 64;
    let approx = (64.0 * std::f64::consts::LN_2 / (base as f64).ln()) as usize;
    let mut epu = approx.max(1);
    loop {
        let next = (base as u128).checked_pow((epu + 1) as u32).unwrap_or(u128::MAX);
        if next <= limit {
            epu += 1;
        } else {
            break;
        }
    }
    loop {
        let cur = (base as u128).checked_pow(epu as u32).unwrap_or(u128::MAX);
        if cur <= limit {
            break;
        }
        epu -= 1;
    }
    epu
}

fn calc_powers(base: u64, epu: usize) -> Vec<u64> {
    let mut p = vec![1u64; epu];
    for i in 1..epu {
        p[i] = p[i - 1] * base;
    }
    p
}

impl RadixArray {
    /// Create an array with values in `[a, b]` and `len` elements, initialized to `a`.
    pub fn new(a: i64, b: i64, len: usize) -> Result<Self, ArrayError> {
        let base = validate_range(a, b)?;
        let epu = calc_epu(base);
        let powers = calc_powers(base, epu);
        let word_count = len.div_ceil(epu);
        Ok(RadixArray {
            base,
            offset: a,
            max_val: b,
            epu,
            powers,
            length: len,
            data: vec![0u64; word_count],
        })
    }

    /// Create from a `Vec<i64>`. Atomic: returns `Err` if any value is out of `[a, b]`.
    pub fn new_with_vec(a: i64, b: i64, vals: Vec<i64>) -> Result<Self, ArrayError> {
        let mut arr = Self::new(a, b, 0)?;
        arr.extend(vals)?;
        Ok(arr)
    }

    /// Create from an iterator. Atomic: returns `Err` if any value is out of `[a, b]`.
    pub fn new_with_iter<I>(a: i64, b: i64, vals: I) -> Result<Self, ArrayError>
    where
        I: Iterator<Item = i64>,
    {
        let mut arr = Self::new(a, b, 0)?;
        arr.extend(vals)?;
        Ok(arr)
    }

    pub fn get(&self, i: usize) -> Result<i64, ArrayError> {
        if i >= self.length {
            return Err(ArrayError::OutOfBounds);
        }
        let (widx, dpos) = (i / self.epu, i % self.epu);
        let stored = (self.data[widx] / self.powers[dpos]) % self.base;
        Ok(stored as i64 + self.offset)
    }

    pub fn set(&mut self, i: usize, v: i64) -> Result<(), ArrayError> {
        if i >= self.length {
            return Err(ArrayError::OutOfBounds);
        }
        if v < self.offset {
            return Err(ArrayError::TooSmall);
        }
        if v > self.max_val {
            return Err(ArrayError::TooLarge);
        }
        let (widx, dpos) = (i / self.epu, i % self.epu);
        let p = self.powers[dpos];
        let old = (self.data[widx] / p) % self.base;
        let new_v = (v - self.offset) as u64;
        self.data[widx] = self.data[widx] - old * p + new_v * p;
        Ok(())
    }

    fn resize(&mut self, len: usize) {
        self.length = len;
        self.data.resize(len.div_ceil(self.epu), 0);
    }

    pub fn push(&mut self, v: i64) -> Result<usize, ArrayError> {
        PackedArrayCore::core_push(self, v)
    }

    pub fn pop(&mut self) -> Result<i64, ArrayError> {
        PackedArrayCore::core_pop(self)
    }

    pub fn extend<I>(&mut self, vals: I) -> Result<(), ArrayError>
    where
        I: IntoIterator<Item = i64>,
    {
        PackedArrayCore::core_extend(self, vals)
    }

    pub fn len(&self) -> usize {
        self.length
    }

    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// Append from another `RadixArray`.
    ///
    /// Fast path when bases and offsets match and `self` is word-aligned:
    /// raw words are copied directly without per-element get/set.
    pub fn extend_array(&mut self, vals: &RadixArray) -> Result<(), ArrayError> {
        if vals.base == self.base
            && vals.offset == self.offset
            && self.length.is_multiple_of(self.epu)
        {
            debug!("fast path: base={}, length={}", self.base, self.length);
            self.length += vals.length;
            self.data.extend_from_slice(&vals.data);
            return Ok(());
        }
        debug!("slow path: base={}, length={}", self.base, self.length);
        self.extend(vals.iter())
    }

    pub fn capacity(&self) -> usize {
        self.data.len() * self.epu
    }

    pub fn base(&self) -> u64 {
        self.base
    }

    pub fn range(&self) -> (i64, i64) {
        (self.offset, self.max_val)
    }

    pub fn datasize(&self) -> usize {
        mem::size_of::<RadixArray>()
            + mem::size_of::<u64>() * self.data.capacity()
            + mem::size_of::<u64>() * self.powers.capacity()
    }

    pub fn iter(&self) -> RadixIter<'_> {
        RadixIter::new(self, self.length)
    }

    pub fn sum(&self) -> Option<i128> {
        if self.length == 0 {
            return None;
        }
        Some(self.iter().fold(0i128, |acc, v| acc + v as i128))
    }

    pub fn min(&self) -> Option<i64> {
        self.iter().min()
    }

    pub fn max(&self) -> Option<i64> {
        self.iter().max()
    }

    pub fn average(&self) -> Option<f64> {
        if self.length == 0 {
            return None;
        }
        Some(self.sum().unwrap() as f64 / self.length as f64)
    }
}

impl PackedArrayCore for RadixArray {
    type Item = i64;

    fn raw_get(&self, i: usize) -> Result<i64, ArrayError> {
        self.get(i)
    }

    fn raw_set(&mut self, i: usize, v: i64) -> Result<(), ArrayError> {
        self.set(i, v)
    }

    fn raw_len(&self) -> usize {
        self.length
    }

    fn raw_resize(&mut self, len: usize) {
        self.resize(len)
    }
}

impl fmt::Display for RadixArray {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[{},{}][{}]=", self.offset, self.max_val, self.length)?;
        let s = self
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(",");
        write!(f, "{}", s)
    }
}

impl Serialize for RadixArray {
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

struct RadixArrayVisitor;

impl<'de> Visitor<'de> for RadixArrayVisitor {
    type Value = RadixArray;

    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "a sequence of integers")
    }

    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let hint = seq.size_hint().unwrap_or(0);
        let mut vals: Vec<i64> = Vec::with_capacity(hint);
        while let Some(v) = seq.next_element()? {
            vals.push(v);
        }

        // Re-infer range from min/max. Empty → default (0, 1).
        // All-same → expand upper bound (or lower if at i64::MAX).
        let (a, b) = if vals.is_empty() {
            (0i64, 1i64)
        } else {
            let &min = vals.iter().min().unwrap();
            let &max = vals.iter().max().unwrap();
            if min == max {
                if max < i64::MAX {
                    (min, min + 1)
                } else {
                    (min - 1, max)
                }
            } else {
                (min, max)
            }
        };

        RadixArray::new_with_vec(a, b, vals)
            .map_err(|e| serde::de::Error::custom(e.to_string()))
    }
}

/// Deserializes from a flat sequence of integers.
///
/// **Note:** Range `[A, B]` is not preserved. It is re-inferred from the
/// min and max of the sequence. An empty sequence deserializes with range `[0, 1]`.
impl<'de> Deserialize<'de> for RadixArray {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_seq(RadixArrayVisitor)
    }
}

#[cfg(test)]
mod tests;
