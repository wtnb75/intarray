use crate::core::PackedArrayCore;
use crate::error::ArrayError;
use log::{debug, error};
use rand::RngExt;
use serde::de::{Deserialize, Deserializer, SeqAccess, Visitor};
use serde::ser::{Serialize, SerializeSeq, Serializer};
use std::ops::{AddAssign, MulAssign, Range, SubAssign};
use std::sync::OnceLock;
use std::{cmp, fmt, mem};

type Element = u64;
const ELEMENT_BITS: usize = mem::size_of::<Element>() * 8;
static MASK_ARRAY: OnceLock<[Element; ELEMENT_BITS / 2 - 1]> = OnceLock::new();

fn init_mask_array() -> [Element; ELEMENT_BITS / 2 - 1] {
    let mut arr = [0; ELEMENT_BITS / 2 - 1];
    for i in 1..(ELEMENT_BITS / 2) {
        let mask0 = ((1 as Element) << i) - 1;
        let mut mask: Element = mask0;
        for _j in 0..(ELEMENT_BITS / (i * 2)) {
            mask = mask.wrapping_shl(i as u32 * 2);
            mask |= mask0;
        }
        arr[i - 1] = mask;
    }
    for (i, j) in arr.iter().enumerate() {
        debug!("mask[{}+1]={:b}", i, j);
    }
    arr
}

fn get_mask(bits: usize) -> Element {
    let arr = MASK_ARRAY.get_or_init(init_mask_array);
    if (bits - 1) < arr.len() {
        arr[bits - 1]
    } else if bits == ELEMENT_BITS {
        !0
    } else {
        ((1 as Element) << bits) - 1
    }
}

fn get_masks(bits: usize) -> (Element, Element) {
    let res = get_mask(bits);
    (res, res.wrapping_shl(bits as u32))
}

trait ElementTrait {
    type E;

    fn val_expand(v: Self::E, bits: usize) -> Self::E;

    fn sum_bits(self, bits: usize) -> Option<Self::E>;

    fn add_bits(self, b: Self::E, bits: usize) -> Option<Self::E>;
    fn sub_bits(self, b: Self::E, bits: usize) -> Option<Self::E>;

    fn addval_bits(self, b: Self::E, bits: usize) -> Option<Self::E>;
    fn subval_bits(self, b: Self::E, bits: usize) -> Option<Self::E>;
    fn mulval_bits(self, b: Self::E, bits: usize) -> Option<Self::E>;

    fn ffs(self) -> usize;
}

impl ElementTrait for Element {
    type E = Element;

    fn val_expand(v: Self::E, bits: usize) -> Self::E {
        if bits == 0 {
            return 0;
        }
        let mut v1: Self::E = 0;
        for _ in 0..(ELEMENT_BITS / bits) {
            v1 <<= bits;
            v1 |= v;
        }
        v1
    }

    fn sum_bits(self, bits: usize) -> Option<Self::E> {
        if bits >= ELEMENT_BITS {
            debug!("return self: {}, bits={}", self, bits);
            return Some(self);
        }
        if bits == 0 {
            error!("invalid argument: bits={}", bits);
            return None;
        }
        let mask = get_mask(bits);
        let res = (self & mask) + ((self >> bits) & mask);
        res.sum_bits(bits * 2)
    }

    fn add_bits(self, b: Self::E, bits: usize) -> Option<Self::E> {
        if bits == 0 {
            error!("invalid argument: bits={}", bits);
            return None;
        }
        let (mask1, mask2) = get_masks(bits);
        let r1 = (self & mask1).wrapping_add(b & mask1);
        let r2 = (self & mask2).wrapping_add(b & mask2);
        if (r1 & mask2) != 0 || (r2 & mask1) != 0 {
            None
        } else {
            Some(r1 + r2)
        }
    }

    fn sub_bits(self, b: Self::E, bits: usize) -> Option<Self::E> {
        if bits == 0 {
            error!("invalid argument: bits={}", bits);
            return None;
        }
        let (mask1, mask2) = get_masks(bits);
        let r1 = (self & mask1).wrapping_sub(b & mask1);
        let r2 = (self & mask2).wrapping_sub(b & mask2);
        if (r1 & mask2) != 0 || (r2 & mask1) != 0 {
            None
        } else {
            Some(r1 + r2)
        }
    }

    fn mulval_bits(self, b: Self::E, bits: usize) -> Option<Self::E> {
        if bits == 0 {
            error!("invalid argument: bits={}", bits);
            return None;
        }
        let (mask1, mask2) = get_masks(bits);
        let r1 = (self & mask1).wrapping_mul(b);
        let r2 = (self & mask2).wrapping_mul(b);
        if (r1 & mask2) != 0 || (r2 & mask1) != 0 {
            None
        } else {
            Some(r1 + r2)
        }
    }

    fn addval_bits(self, b: Self::E, bits: usize) -> Option<Self::E> {
        self.add_bits(Self::val_expand(b, bits), bits)
    }

    fn subval_bits(self, b: Self::E, bits: usize) -> Option<Self::E> {
        self.sub_bits(Self::val_expand(b, bits), bits)
    }

    fn ffs(self) -> usize {
        ELEMENT_BITS - self.leading_zeros() as usize
    }
}

/// 1-64 bits unsigned integer array
pub struct IntArray {
    pub bits: usize,
    pub length: usize,
    data: Vec<Element>,
}

/// Iterator for IntArray
pub struct IntIter<'a> {
    range: Range<usize>,
    a: &'a IntArray,
}

impl IntArray {
    fn sizeval(b: usize, len: usize) -> (usize, usize) {
        assert!(
            b > 0 && b <= ELEMENT_BITS,
            "bits must be in 1..={}, got {}",
            ELEMENT_BITS,
            b
        );
        let bpd = ELEMENT_BITS / b;
        (bpd, len.div_ceil(bpd))
    }

    fn bits_for_max(max_val: Element) -> usize {
        max_val.ffs().max(1)
    }

    pub fn new(b: usize, len: usize) -> IntArray {
        let (bpd, cap) = IntArray::sizeval(b, len);
        debug!("bpd={}, cap={}", bpd, cap);
        IntArray {
            bits: b,
            length: len,
            data: vec![0; cap],
        }
    }

    pub fn new_with_vec(b: usize, vals: Vec<Element>) -> IntArray {
        let (bpd, cap) = IntArray::sizeval(b, vals.len());
        debug!("bpd={}, cap={}", bpd, cap);
        let mut res = IntArray {
            bits: b,
            length: vals.len(),
            data: vec![0; cap],
        };
        for (i, v) in vals.iter().enumerate() {
            res.set(i, *v).unwrap();
        }
        res
    }

    pub fn new_with_iter<I>(b: usize, vals: I) -> IntArray
    where
        I: Iterator<Item = Element>,
    {
        const UNIT: usize = 1024;
        let (bpd, cap) = IntArray::sizeval(b, UNIT);
        let mut cnt = 0usize;
        debug!("bpd={}, cap={}", bpd, cap);
        let mut res = IntArray {
            bits: b,
            length: UNIT,
            data: vec![0; cap],
        };
        for v in vals {
            if cnt >= res.length {
                res.resize(res.length + UNIT);
            }
            res.set(cnt, v).unwrap();
            cnt += 1
        }
        res.resize(cnt);
        res
    }

    pub fn subarray(&self, offset: usize, length: usize) -> IntArray {
        let mut res = IntArray::new(self.bits, length);
        let (bpd, _) = IntArray::sizeval(self.bits, 0);
        if offset.is_multiple_of(bpd) {
            debug!("fast path: offset={}, length={}", offset, length);
            for i in 0..(length / bpd) {
                res.data[i] = self.data[offset / bpd + i];
            }
            if !length.is_multiple_of(bpd) {
                let i = length / bpd;
                let mask = ((1 as Element) << (self.bits * (length % bpd))) - 1;
                res.data[i] = self.data[offset / bpd + i] & mask;
            }
        } else {
            debug!("slow path: offset={}, length={}", offset, length);
            for i in 0..length {
                res.set(i, self.get(offset + i).unwrap()).unwrap();
            }
        }
        res
    }

    pub fn datasize(&self) -> usize {
        mem::size_of::<IntArray>() + (ELEMENT_BITS / 8) * self.data.capacity()
    }

    pub fn get(&self, i: usize) -> Result<Element, ArrayError> {
        if self.length <= i {
            return Err(ArrayError::OutOfBounds);
        }
        let (_, idx, iv) = self.getoffset(i);
        let vv = self.data[idx];
        let res = (vv >> (iv * self.bits)) & self.max_value();
        Ok(res)
    }

    pub fn set(&mut self, i: usize, v: Element) -> Result<(), ArrayError> {
        if self.max_value() < v {
            return Err(ArrayError::TooLarge);
        }
        if self.length <= i {
            return Err(ArrayError::OutOfBounds);
        }
        let (_, idx, iv) = self.getoffset(i);
        let mask1 = (self.max_value()) << (iv * self.bits);
        let mask2 = v << (iv * self.bits);
        self.data[idx] = (self.data[idx] & (!mask1)) | mask2;
        Ok(())
    }

    pub fn resize(&mut self, len: usize) {
        let bpd = ELEMENT_BITS / self.bits;
        let cap = len.div_ceil(bpd);
        self.length = len;
        self.data.resize(cap, 0);
    }

    pub fn push(&mut self, v: Element) -> Result<usize, ArrayError> {
        PackedArrayCore::core_push(self, v)
    }

    pub fn pop(&mut self) -> Result<Element, ArrayError> {
        PackedArrayCore::core_pop(self)
    }

    pub fn extend<I>(&mut self, vals: I) -> Result<(), ArrayError>
    where
        I: IntoIterator<Item = Element>,
    {
        PackedArrayCore::core_extend(self, vals)
    }

    pub fn len(&self) -> usize {
        self.length
    }

    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    pub fn extend_array(&mut self, vals: &IntArray) -> Result<(), ArrayError> {
        let (bpd, _) = IntArray::sizeval(self.bits, 0);
        if vals.bits == self.bits && self.length.is_multiple_of(bpd) {
            debug!("fast path: bits={}, length={}", self.bits, self.length);
            self.length += vals.length;
            self.data.extend_from_slice(&vals.data);
            return Ok(());
        }
        debug!("slow path: bits={}, length={}", self.bits, self.length);
        self.extend(vals.iter())
    }

    pub fn shape(&self, bits: usize) -> IntArray {
        IntArray::new_with_iter(bits, self.iter())
    }

    pub fn shape_auto(&self) -> IntArray {
        let bits = IntArray::bits_for_max(self.iter().max().unwrap_or(0));
        IntArray::new_with_iter(bits, self.iter())
    }

    pub fn iter(&self) -> IntIter<'_> {
        IntIter {
            range: 0..self.length,
            a: self,
        }
    }

    pub fn capacity(&self) -> usize {
        let bpd = ELEMENT_BITS / self.bits;
        self.data.len() * bpd
    }

    pub fn max_value(&self) -> Element {
        if self.bits == ELEMENT_BITS {
            return Element::MAX;
        }
        ((1 as Element) << self.bits) - 1
    }

    pub fn max(&self) -> Option<Element> {
        self.iter().max()
    }

    pub fn min(&self) -> Option<Element> {
        self.iter().min()
    }

    pub fn average(&self) -> Option<f64> {
        if self.length == 0 {
            return None;
        }
        Some(self.sum().unwrap() as f64 / self.len() as f64)
    }

    pub fn add(&mut self, i: usize, v: Element) -> Result<(), ArrayError> {
        let n = self.get(i)?;
        let sum = n.checked_add(v).ok_or(ArrayError::TooLarge)?;
        self.set(i, sum)
    }

    pub fn sub(&mut self, i: usize, v: Element) -> Result<(), ArrayError> {
        let n = self.get(i)?;
        let diff = n.checked_sub(v).ok_or(ArrayError::TooSmall)?;
        self.set(i, diff)
    }

    pub fn incr_limit(&mut self, i: usize) -> Option<Element> {
        match self.get(i) {
            Ok(n) => {
                if n != self.max_value() {
                    self.set(i, n + 1).unwrap();
                    Some(n)
                } else {
                    None
                }
            }
            Err(_e) => None,
        }
    }

    pub fn decr_limit(&mut self, i: usize) -> Option<Element> {
        match self.get(i) {
            Ok(n) => {
                if n != 0 {
                    self.set(i, n - 1).unwrap();
                    Some(n)
                } else {
                    None
                }
            }
            Err(_e) => None,
        }
    }

    pub fn incr(&mut self, i: usize) -> Result<(), ArrayError> {
        self.add(i, 1)
    }

    pub fn decr(&mut self, i: usize) -> Result<(), ArrayError> {
        self.sub(i, 1)
    }

    pub fn sum(&self) -> Option<u128> {
        if self.bits > ELEMENT_BITS / 2 {
            return self.sum0();
        }
        let mut res: u128 = 0;
        for i in self.data.iter() {
            res += (*i).sum_bits(self.bits).unwrap() as u128;
            debug!("sum: {} -> {}", *i, res);
        }
        Some(res)
    }

    pub(crate) fn sum0(&self) -> Option<u128> {
        Some(self.iter().fold(0u128, |sum, a| sum + a as u128))
    }

    pub fn fill_random(&mut self) {
        if self.length == 0 {
            return;
        }
        let mut rng = rand::rng();
        if ELEMENT_BITS.is_multiple_of(self.bits) {
            for i in 0..(self.data.len() - 1) {
                self.data[i] = rng.random();
            }
        } else {
            let mvbits = ELEMENT_BITS - (ELEMENT_BITS % self.bits);
            let mvval = (1 as Element) << mvbits;
            for i in 0..(self.data.len() - 1) {
                self.data[i] = rng.random_range(0..mvval);
            }
        }
        let bpd = ELEMENT_BITS / self.bits;
        for i in (self.data.len() - 1) * bpd..self.length {
            self.set(i, rng.random_range(0..=self.max_value())).unwrap();
        }
    }

    fn getoffset(&self, i: usize) -> (usize, usize, usize) {
        let bpd = ELEMENT_BITS / self.bits;
        (bpd, i / bpd, i % bpd)
    }
}

impl PackedArrayCore for IntArray {
    type Item = Element;

    fn raw_get(&self, i: usize) -> Result<Element, ArrayError> {
        self.get(i)
    }

    fn raw_set(&mut self, i: usize, v: Element) -> Result<(), ArrayError> {
        self.set(i, v)
    }

    fn raw_len(&self) -> usize {
        self.length
    }

    fn raw_resize(&mut self, len: usize) {
        self.resize(len)
    }
}

impl<'a> Iterator for IntIter<'a> {
    type Item = Element;
    fn next(&mut self) -> Option<Element> {
        self.range.next().map(|i| self.a.get(i).unwrap())
    }

    fn count(self) -> usize {
        self.range.len()
    }
}

impl fmt::Display for IntArray {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}[{}]=", self.bits, self.length).unwrap();
        write!(
            f,
            "{}",
            self.iter()
                .map(|x| x.to_string())
                .collect::<Vec<String>>()
                .join(",")
        )
        .unwrap();
        Ok(())
    }
}

impl Serialize for IntArray {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut seq = serializer.serialize_seq(Some(self.length))?;
        self.iter().for_each(|x| seq.serialize_element(&x).unwrap());
        seq.end()
    }
}

struct IntArrayVisitor;

impl<'de> Visitor<'de> for IntArrayVisitor {
    type Value = IntArray;

    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "a sequence of unsigned integers")
    }

    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let hint = seq.size_hint().unwrap_or(0);
        let mut vals: Vec<Element> = Vec::with_capacity(hint);
        while let Some(v) = seq.next_element()? {
            vals.push(v);
        }
        let bits = IntArray::bits_for_max(vals.iter().copied().max().unwrap_or(0));
        Ok(IntArray::new_with_vec(bits, vals))
    }
}

/// Deserializes from a flat sequence of unsigned integers.
///
/// **Note:** Bit width is not preserved. The width is re-inferred from the
/// maximum value in the sequence (minimum 1). An all-zeros array always
/// deserializes with `bits = 1` regardless of the original width.
impl<'de> Deserialize<'de> for IntArray {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_seq(IntArrayVisitor)
    }
}

impl AddAssign<Element> for IntArray {
    fn add_assign(&mut self, v: Element) {
        let (bpd, _) = IntArray::sizeval(self.bits, 0);
        debug!("length={}, bits={}, bpd={}", self.length, self.bits, bpd);
        for i in 0..(self.length / bpd) {
            debug!("i={}, v={}", i, v);
            self.data[i] = self.data[i].addval_bits(v, self.bits).unwrap();
        }
        if !self.length.is_multiple_of(bpd) {
            let i = self.length / bpd;
            let mask = ((1 as Element) << (self.bits * (self.length % bpd))) - 1;
            debug!("mask={:x} ({} bits)", mask, mask.ffs());
            self.data[i] = self.data[i].addval_bits(v, self.bits).unwrap() & mask;
        }
    }
}

impl AddAssign<&'_ IntArray> for IntArray {
    fn add_assign(&mut self, v: &IntArray) {
        if self.bits == v.bits && self.length == v.length {
            debug!("fast path: bits={}, length={}", self.bits, self.length);
            let (bpd, _) = IntArray::sizeval(self.bits, 0);
            for i in 0..(self.length / bpd) {
                self.data[i] = self.data[i].add_bits(v.data[i], self.bits).unwrap();
            }
            if !self.length.is_multiple_of(bpd) {
                let i = self.length / bpd;
                let mask = ((1 as Element) << (self.bits * (self.length % bpd))) - 1;
                self.data[i] = self.data[i].add_bits(v.data[i], self.bits).unwrap() & mask;
            }
            return;
        }
        debug!(
            "slow path: bits={}/{}, length={}/{}",
            self.bits, v.bits, self.length, v.length
        );
        for i in 0..cmp::min(self.length, v.length) {
            self.set(i, self.get(i).unwrap() + v.get(i).unwrap())
                .unwrap();
        }
    }
}

impl SubAssign<Element> for IntArray {
    fn sub_assign(&mut self, v: Element) {
        let (bpd, _) = IntArray::sizeval(self.bits, 0);
        for i in 0..(self.length / bpd) {
            self.data[i] = self.data[i].subval_bits(v, self.bits).unwrap();
        }
        if !self.length.is_multiple_of(bpd) {
            let i = self.length / bpd;
            let mask = ((1 as Element) << (self.bits * (self.length % bpd))) - 1;
            self.data[i] = (self.data[i] | (!mask)).subval_bits(v, self.bits).unwrap() & mask;
        }
    }
}

impl SubAssign<&'_ IntArray> for IntArray {
    fn sub_assign(&mut self, v: &IntArray) {
        if self.bits == v.bits && self.length == v.length {
            debug!("fast path: bits={}, length={}", self.bits, self.length);
            let (bpd, _) = IntArray::sizeval(self.bits, 0);
            for i in 0..(self.length / bpd) {
                self.data[i] = self.data[i].sub_bits(v.data[i], self.bits).unwrap();
            }
            if !self.length.is_multiple_of(bpd) {
                let i = self.length / bpd;
                let mask = ((1 as Element) << (self.bits * (self.length % bpd))) - 1;
                self.data[i] = (self.data[i] | (!mask))
                    .sub_bits(v.data[i], self.bits)
                    .unwrap()
                    & mask;
            }
            return;
        }
        debug!(
            "slow path: bits={}/{}, length={}/{}",
            self.bits, v.bits, self.length, v.length
        );
        for i in 0..cmp::min(self.length, v.length) {
            self.sub(i, v.get(i).unwrap()).unwrap();
        }
    }
}

impl MulAssign<Element> for IntArray {
    fn mul_assign(&mut self, v: Element) {
        let (bpd, _) = IntArray::sizeval(self.bits, 0);
        for i in 0..(self.length / bpd) {
            self.data[i] = self.data[i].mulval_bits(v, self.bits).unwrap();
        }
        if !self.length.is_multiple_of(bpd) {
            let i = self.length / bpd;
            let mask = ((1 as Element) << (self.bits * (self.length % bpd))) - 1;
            self.data[i] = (self.data[i] & mask).mulval_bits(v, self.bits).unwrap() & mask;
        }
    }
}

impl Clone for IntArray {
    fn clone(&self) -> IntArray {
        let mut res = IntArray::new(self.bits, self.length);
        res.data.clone_from_slice(&self.data);
        res
    }
}

#[cfg(test)]
mod tests;
