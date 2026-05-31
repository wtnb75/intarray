use crate::bits::{
    BitBlock, DEFAULT_K, read_biguint, read_elias_gamma, write_biguint, write_elias_gamma,
    zigzag_decode, zigzag_encode,
};
use crate::error::ArrayError;
use log::debug;
use num_bigint::BigInt;
use num_traits::{ToPrimitive, Zero};
use serde::de::{Deserialize, Deserializer, SeqAccess, Visitor};
use serde::ser::{Serialize, SerializeSeq, Serializer};
use std::{fmt, mem};

/// Packed array of arbitrary-precision integers using Elias gamma coding.
///
/// Elements are zigzag-encoded then stored as self-delimiting Elias gamma
/// bit strings, grouped into blocks of `k` elements each.
///
/// **Note on `PartialEq`:** two arrays are equal when they contain the same
/// elements in the same order, regardless of `k` or internal block structure.
/// This means a serde round-trip (which resets `k` to 64) still compares equal.
#[derive(Clone, Debug)]
pub struct VarIntArray {
    k: usize,
    blocks: Vec<BitBlock>,
    length: usize,
}

/// Iterator for [`VarIntArray`]
pub struct VarIntIter<'a> {
    arr: &'a VarIntArray,
    block_idx: usize,
    elem_in_block: usize,
    bit_pos: usize,
    remaining: usize,
}

// --- encode / decode one value ---

fn encode_into(block: &mut BitBlock, v: &BigInt) {
    let z = zigzag_encode(v);
    let b = z.bits() as usize;
    write_elias_gamma(&mut block.data, &mut block.bit_len, b + 1);
    write_biguint(&mut block.data, &mut block.bit_len, &z, b);
    block.count += 1;
}

fn decode_one(data: &[u8], bit_pos: &mut usize) -> BigInt {
    let bp1 = read_elias_gamma(data, bit_pos);
    let b = bp1 - 1;
    let z = read_biguint(data, bit_pos, b);
    zigzag_decode(z)
}

fn decode_block(block: &BitBlock) -> Vec<BigInt> {
    let mut bit_pos = 0;
    let mut elems = Vec::with_capacity(block.count);
    for _ in 0..block.count {
        elems.push(decode_one(&block.data, &mut bit_pos));
    }
    elems
}

fn reencode_block(elems: &[BigInt]) -> BitBlock {
    let mut block = BitBlock::empty();
    for elem in elems {
        encode_into(&mut block, elem);
    }
    block
}

impl VarIntArray {
    /// Creates an empty `VarIntArray` with block size `k`.
    pub fn new(k: usize) -> Result<Self, ArrayError> {
        if k == 0 {
            return Err(ArrayError::InvalidRange);
        }
        debug!("VarIntArray::new k={}", k);
        Ok(VarIntArray { k, blocks: vec![], length: 0 })
    }

    /// Creates from a `Vec<BigInt>`.
    pub fn new_with_vec(k: usize, vals: Vec<BigInt>) -> Result<Self, ArrayError> {
        let mut arr = Self::new(k)?;
        arr.extend(vals)?;
        Ok(arr)
    }

    /// Creates from an iterator.
    pub fn new_with_iter(
        k: usize,
        vals: impl IntoIterator<Item = BigInt>,
    ) -> Result<Self, ArrayError> {
        let mut arr = Self::new(k)?;
        arr.extend(vals)?;
        Ok(arr)
    }

    pub fn get(&self, i: usize) -> Result<BigInt, ArrayError> {
        if i >= self.length {
            return Err(ArrayError::OutOfBounds);
        }
        let block = &self.blocks[i / self.k];
        let target = i % self.k;
        let mut bit_pos = 0;
        for _ in 0..target {
            decode_one(&block.data, &mut bit_pos);
        }
        Ok(decode_one(&block.data, &mut bit_pos))
    }

    pub fn set(&mut self, i: usize, v: BigInt) -> Result<(), ArrayError> {
        if i >= self.length {
            return Err(ArrayError::OutOfBounds);
        }
        let block_idx = i / self.k;
        let mut elems = decode_block(&self.blocks[block_idx]);
        elems[i % self.k] = v;
        self.blocks[block_idx] = reencode_block(&elems);
        Ok(())
    }

    pub fn push(&mut self, v: BigInt) -> Result<usize, ArrayError> {
        if self.blocks.is_empty() || self.blocks.last().unwrap().count == self.k {
            self.blocks.push(BitBlock::empty());
        }
        encode_into(self.blocks.last_mut().unwrap(), &v);
        self.length += 1;
        Ok(self.length - 1)
    }

    pub fn pop(&mut self) -> Result<BigInt, ArrayError> {
        if self.blocks.is_empty() {
            return Err(ArrayError::Empty);
        }
        let mut elems = decode_block(self.blocks.last().unwrap());
        let ret = elems.pop().unwrap();
        if elems.is_empty() {
            self.blocks.pop();
        } else {
            *self.blocks.last_mut().unwrap() = reencode_block(&elems);
        }
        self.length -= 1;
        Ok(ret)
    }

    pub fn extend(&mut self, vals: impl IntoIterator<Item = BigInt>) -> Result<(), ArrayError> {
        for v in vals {
            self.push(v)?;
        }
        Ok(())
    }

    pub fn extend_array(&mut self, other: &VarIntArray) -> Result<(), ArrayError> {
        for v in other.iter() {
            self.push(v)?;
        }
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.length
    }

    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    pub fn block_size(&self) -> usize {
        self.k
    }

    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }

    pub fn datasize(&self) -> usize {
        mem::size_of::<VarIntArray>()
            + mem::size_of::<BitBlock>() * self.blocks.capacity()
            + self.blocks.iter().map(|b| b.data.capacity()).sum::<usize>()
    }

    pub fn iter(&self) -> VarIntIter<'_> {
        VarIntIter {
            arr: self,
            block_idx: 0,
            elem_in_block: 0,
            bit_pos: 0,
            remaining: self.length,
        }
    }

    pub fn sum(&self) -> Option<BigInt> {
        if self.length == 0 {
            return None;
        }
        Some(self.iter().fold(BigInt::zero(), |acc, v| acc + v))
    }

    pub fn min(&self) -> Option<BigInt> {
        let mut it = self.iter();
        let first = it.next()?;
        Some(it.fold(first, |a, b| if b < a { b } else { a }))
    }

    pub fn max(&self) -> Option<BigInt> {
        let mut it = self.iter();
        let first = it.next()?;
        Some(it.fold(first, |a, b| if b > a { b } else { a }))
    }

    /// Returns `None` if empty, or `NaN` if the sum cannot be represented as `f64`.
    pub fn average(&self) -> Option<f64> {
        if self.length == 0 {
            return None;
        }
        let s = self.sum()?;
        Some(s.to_f64().unwrap_or(f64::NAN) / self.length as f64)
    }
}

impl<'a> Iterator for VarIntIter<'a> {
    type Item = BigInt;

    fn next(&mut self) -> Option<BigInt> {
        if self.remaining == 0 {
            return None;
        }
        if self.elem_in_block == self.arr.blocks[self.block_idx].count {
            self.block_idx += 1;
            self.elem_in_block = 0;
            self.bit_pos = 0;
        }
        let v = decode_one(&self.arr.blocks[self.block_idx].data, &mut self.bit_pos);
        self.elem_in_block += 1;
        self.remaining -= 1;
        Some(v)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl ExactSizeIterator for VarIntIter<'_> {}

impl Eq for VarIntArray {}

impl PartialEq for VarIntArray {
    fn eq(&self, other: &Self) -> bool {
        self.length == other.length && self.iter().eq(other.iter())
    }
}

impl fmt::Display for VarIntArray {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[k={}][{}]=", self.k, self.length)?;
        let s = self.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",");
        write!(f, "{}", s)
    }
}

impl Serialize for VarIntArray {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut seq = serializer.serialize_seq(Some(self.length))?;
        for v in self.iter() {
            seq.serialize_element(&v.to_string())?;
        }
        seq.end()
    }
}

struct VarIntArrayVisitor;

impl<'de> Visitor<'de> for VarIntArrayVisitor {
    type Value = VarIntArray;

    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "a sequence of decimal integer strings")
    }

    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let mut arr = VarIntArray::new(DEFAULT_K)
            .map_err(|e| serde::de::Error::custom(e.to_string()))?;
        while let Some(s) = seq.next_element::<String>()? {
            let v: BigInt = s
                .parse()
                .map_err(|e: num_bigint::ParseBigIntError| serde::de::Error::custom(e.to_string()))?;
            arr.push(v)
                .map_err(|e| serde::de::Error::custom(e.to_string()))?;
        }
        Ok(arr)
    }
}

/// Deserializes from a flat sequence of decimal strings.
///
/// **Note:** `k` is not preserved; the array is reconstructed with `k = 64`.
impl<'de> Deserialize<'de> for VarIntArray {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_seq(VarIntArrayVisitor)
    }
}

#[cfg(test)]
mod tests;
