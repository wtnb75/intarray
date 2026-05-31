use crate::error::ArrayError;
use log::debug;
use num_bigint::{BigInt, BigUint, Sign};
use num_rational::Ratio;
use num_traits::Zero;
use serde::de::{Deserialize, Deserializer, SeqAccess, Visitor};
use serde::ser::{Serialize, SerializeSeq, Serializer};
use std::{fmt, mem};

const DEFAULT_K: usize = 64;

#[derive(Clone, Debug, PartialEq)]
struct BitBlock {
    data: Vec<u8>,
    bit_len: usize,
    count: usize,
}

impl BitBlock {
    fn empty() -> Self {
        BitBlock { data: vec![], bit_len: 0, count: 0 }
    }
}

/// Packed array of arbitrary-precision rationals using Elias gamma coding.
///
/// The numerator is zigzag-encoded then stored via extended Elias gamma;
/// the denominator (always ≥ 1) uses standard Elias gamma.
/// Elements are grouped into blocks of `k` each.
///
/// **Note on `PartialEq`:** two arrays are equal when they contain the same
/// elements in the same order, regardless of `k` or internal block structure.
#[derive(Clone, Debug)]
pub struct RatioArray {
    k: usize,
    blocks: Vec<BitBlock>,
    length: usize,
}

/// Iterator for [`RatioArray`]
pub struct RatioIter<'a> {
    arr: &'a RatioArray,
    block_idx: usize,
    elem_in_block: usize,
    bit_pos: usize,
    remaining: usize,
}

// --- bit I/O ---

fn read_bit(data: &[u8], bit_pos: usize) -> u8 {
    let byte_idx = bit_pos / 8;
    let bit_idx = 7 - (bit_pos % 8);
    (data[byte_idx] >> bit_idx) & 1
}

fn write_bit(data: &mut Vec<u8>, bit_len: &mut usize, bit: u8) {
    if (*bit_len).is_multiple_of(8) {
        data.push(0u8);
    }
    let byte_idx = *bit_len / 8;
    let bit_idx = 7 - (*bit_len % 8);
    data[byte_idx] |= bit << bit_idx;
    *bit_len += 1;
}

// --- zigzag ---

fn zigzag_encode(n: &BigInt) -> BigUint {
    match n.sign() {
        Sign::NoSign => BigUint::zero(),
        Sign::Plus => n.magnitude().clone() << 1,
        Sign::Minus => (n.magnitude().clone() << 1) - 1u32,
    }
}

fn zigzag_decode(z: BigUint) -> BigInt {
    if z.bit(0) {
        -BigInt::from((z + 1u32) / 2u32)
    } else {
        BigInt::from(z >> 1)
    }
}

// --- standard Elias gamma for usize n >= 1 (used for numerator's B+1) ---

fn write_elias_gamma(data: &mut Vec<u8>, bit_len: &mut usize, n: usize) {
    debug_assert!(n >= 1);
    let k = usize::BITS as usize - n.leading_zeros() as usize - 1;
    for _ in 0..k {
        write_bit(data, bit_len, 0);
    }
    write_bit(data, bit_len, 1);
    for i in (0..k).rev() {
        write_bit(data, bit_len, ((n >> i) & 1) as u8);
    }
}

fn read_elias_gamma(data: &[u8], bit_pos: &mut usize) -> usize {
    let mut k = 0usize;
    while read_bit(data, *bit_pos) == 0 {
        k += 1;
        *bit_pos += 1;
    }
    *bit_pos += 1; // skip "1"
    let mut lower = 0usize;
    for _ in 0..k {
        lower = (lower << 1) | (read_bit(data, *bit_pos) as usize);
        *bit_pos += 1;
    }
    (1 << k) + lower
}

// --- BigUint bit I/O (MSB first) ---

fn write_biguint(data: &mut Vec<u8>, bit_len: &mut usize, z: &BigUint, b: usize) {
    for i in (0..b).rev() {
        write_bit(data, bit_len, u8::from(z.bit(i as u64)));
    }
}

fn read_biguint(data: &[u8], bit_pos: &mut usize, b: usize) -> BigUint {
    let mut result = BigUint::zero();
    for _ in 0..b {
        result <<= 1u32;
        if read_bit(data, *bit_pos) == 1 {
            result |= BigUint::from(1u32);
        }
        *bit_pos += 1;
    }
    result
}

// --- BigUint Elias gamma for denominator (q >= 1) ---

fn write_denom(data: &mut Vec<u8>, bit_len: &mut usize, q: &BigUint) {
    debug_assert!(!q.is_zero());
    let k_bits = q.bits() as usize - 1; // floor(log2(q)), defined since q >= 1
    for _ in 0..k_bits {
        write_bit(data, bit_len, 0);
    }
    write_bit(data, bit_len, 1);
    for i in (0..k_bits).rev() {
        write_bit(data, bit_len, u8::from(q.bit(i as u64)));
    }
}

fn read_denom(data: &[u8], bit_pos: &mut usize) -> BigUint {
    let mut k_bits = 0usize;
    while read_bit(data, *bit_pos) == 0 {
        k_bits += 1;
        *bit_pos += 1;
    }
    *bit_pos += 1; // skip "1"
    let lower = read_biguint(data, bit_pos, k_bits);
    (BigUint::from(1u32) << k_bits) + lower
}

// --- encode / decode one value ---

fn encode_into(block: &mut BitBlock, v: &Ratio<BigInt>) {
    let p = v.numer().clone();
    // denom is always positive in a normalized Ratio<BigInt>
    let q = BigUint::try_from(v.denom().clone()).unwrap();

    let z = zigzag_encode(&p);
    let b = z.bits() as usize;
    write_elias_gamma(&mut block.data, &mut block.bit_len, b + 1);
    write_biguint(&mut block.data, &mut block.bit_len, &z, b);

    write_denom(&mut block.data, &mut block.bit_len, &q);

    block.count += 1;
}

fn decode_one(data: &[u8], bit_pos: &mut usize) -> Ratio<BigInt> {
    let bp1 = read_elias_gamma(data, bit_pos);
    let b = bp1 - 1;
    let z = read_biguint(data, bit_pos, b);
    let p = zigzag_decode(z);

    let q = read_denom(data, bit_pos);
    // stored values are already normalized; skip GCD recomputation
    Ratio::new_raw(p, BigInt::from(q))
}

fn decode_block(block: &BitBlock) -> Vec<Ratio<BigInt>> {
    let mut bit_pos = 0;
    let mut elems = Vec::with_capacity(block.count);
    for _ in 0..block.count {
        elems.push(decode_one(&block.data, &mut bit_pos));
    }
    elems
}

fn reencode_block(elems: &[Ratio<BigInt>]) -> BitBlock {
    let mut block = BitBlock::empty();
    for elem in elems {
        encode_into(&mut block, elem);
    }
    block
}

impl RatioArray {
    /// Creates an empty `RatioArray` with block size `k`.
    pub fn new(k: usize) -> Result<Self, ArrayError> {
        if k == 0 {
            return Err(ArrayError::InvalidRange);
        }
        debug!("RatioArray::new k={}", k);
        Ok(RatioArray { k, blocks: vec![], length: 0 })
    }

    /// Creates from a `Vec<Ratio<BigInt>>`.
    pub fn new_with_vec(k: usize, vals: Vec<Ratio<BigInt>>) -> Result<Self, ArrayError> {
        let mut arr = Self::new(k)?;
        arr.extend(vals)?;
        Ok(arr)
    }

    /// Creates from an iterator.
    pub fn new_with_iter(
        k: usize,
        vals: impl IntoIterator<Item = Ratio<BigInt>>,
    ) -> Result<Self, ArrayError> {
        let mut arr = Self::new(k)?;
        arr.extend(vals)?;
        Ok(arr)
    }

    pub fn get(&self, i: usize) -> Result<Ratio<BigInt>, ArrayError> {
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

    pub fn set(&mut self, i: usize, v: Ratio<BigInt>) -> Result<(), ArrayError> {
        if i >= self.length {
            return Err(ArrayError::OutOfBounds);
        }
        let block_idx = i / self.k;
        let mut elems = decode_block(&self.blocks[block_idx]);
        elems[i % self.k] = v;
        self.blocks[block_idx] = reencode_block(&elems);
        Ok(())
    }

    pub fn push(&mut self, v: Ratio<BigInt>) -> Result<usize, ArrayError> {
        if self.blocks.is_empty() || self.blocks.last().unwrap().count == self.k {
            self.blocks.push(BitBlock::empty());
        }
        encode_into(self.blocks.last_mut().unwrap(), &v);
        self.length += 1;
        Ok(self.length - 1)
    }

    pub fn pop(&mut self) -> Result<Ratio<BigInt>, ArrayError> {
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

    pub fn extend(&mut self, vals: impl IntoIterator<Item = Ratio<BigInt>>) -> Result<(), ArrayError> {
        for v in vals {
            self.push(v)?;
        }
        Ok(())
    }

    pub fn extend_array(&mut self, other: &RatioArray) -> Result<(), ArrayError> {
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
        mem::size_of::<RatioArray>()
            + mem::size_of::<BitBlock>() * self.blocks.capacity()
            + self.blocks.iter().map(|b| b.data.capacity()).sum::<usize>()
    }

    pub fn iter(&self) -> RatioIter<'_> {
        RatioIter {
            arr: self,
            block_idx: 0,
            elem_in_block: 0,
            bit_pos: 0,
            remaining: self.length,
        }
    }

    pub fn sum(&self) -> Option<Ratio<BigInt>> {
        if self.length == 0 {
            return None;
        }
        Some(self.iter().fold(Ratio::from_integer(BigInt::zero()), |acc, v| acc + v))
    }

    pub fn min(&self) -> Option<Ratio<BigInt>> {
        let mut it = self.iter();
        let first = it.next()?;
        Some(it.fold(first, |a, b| if b < a { b } else { a }))
    }

    pub fn max(&self) -> Option<Ratio<BigInt>> {
        let mut it = self.iter();
        let first = it.next()?;
        Some(it.fold(first, |a, b| if b > a { b } else { a }))
    }

    /// Returns exact rational average. `None` if empty.
    pub fn average(&self) -> Option<Ratio<BigInt>> {
        if self.length == 0 {
            return None;
        }
        let s = self.sum()?;
        Some(s / Ratio::from_integer(BigInt::from(self.length)))
    }
}

impl<'a> Iterator for RatioIter<'a> {
    type Item = Ratio<BigInt>;

    fn next(&mut self) -> Option<Ratio<BigInt>> {
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

impl ExactSizeIterator for RatioIter<'_> {}

impl PartialEq for RatioArray {
    fn eq(&self, other: &Self) -> bool {
        self.length == other.length && self.iter().eq(other.iter())
    }
}

impl fmt::Display for RatioArray {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[k={}][{}]=", self.k, self.length)?;
        let s = self.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",");
        write!(f, "{}", s)
    }
}

impl Serialize for RatioArray {
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

struct RatioArrayVisitor;

impl<'de> Visitor<'de> for RatioArrayVisitor {
    type Value = RatioArray;

    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "a sequence of rational number strings (\"p/q\" or \"p\")")
    }

    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let mut arr = RatioArray::new(DEFAULT_K)
            .map_err(|e| serde::de::Error::custom(e.to_string()))?;
        while let Some(s) = seq.next_element::<String>()? {
            let v = s
                .parse::<Ratio<BigInt>>()
                .map_err(|e| serde::de::Error::custom(e.to_string()))?;
            arr.push(v)
                .map_err(|e| serde::de::Error::custom(e.to_string()))?;
        }
        Ok(arr)
    }
}

/// Deserializes from a flat sequence of rational strings ("p/q" or "p").
///
/// **Note:** `k` is not preserved; the array is reconstructed with `k = 64`.
impl<'de> Deserialize<'de> for RatioArray {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_seq(RatioArrayVisitor)
    }
}

#[cfg(test)]
mod tests;
