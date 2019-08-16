#[macro_use]
extern crate log;

use rand::Rng;
use serde::ser::{Serialize, SerializeSeq, Serializer};
use std::ops::{AddAssign, MulAssign, Range, SubAssign};
use std::sync::Once;
use std::{cmp, fmt, mem};

type Element = u64;
// type Element = u128;
const ELEMENT_BITS: usize = mem::size_of::<Element>() * 8;
static INIT_MASK: Once = Once::new();
static mut MASK_ARRAY: [Element; ELEMENT_BITS / 2 - 1] = [0; ELEMENT_BITS / 2 - 1];

fn init_mask_fn() {
    unsafe {
        for i in 1..(ELEMENT_BITS / 2) {
            let mask0 = ((1 as Element) << i) - 1;
            let mut mask: Element = mask0;
            for _j in 0..(ELEMENT_BITS / (i * 2)) {
                mask = mask.wrapping_shl(i as u32 * 2);
                mask |= mask0;
            }
            MASK_ARRAY[i - 1] = mask;
        }
        for (i, j) in MASK_ARRAY.iter().enumerate() {
            debug!("mask[{}+1]={:b}", i, j);
        }
    }
}

fn get_mask(bits: usize) -> Element {
    INIT_MASK.call_once(|| init_mask_fn());
    unsafe {
        if (bits - 1) < MASK_ARRAY.len() {
            MASK_ARRAY[bits - 1]
        } else if bits == ELEMENT_BITS {
            !0
        } else {
            ((1 as Element) << bits) - 1
        }
    }
}

fn get_masks(bits: usize) -> (Element, Element) {
    let res = get_mask(bits);
    (res, res.wrapping_shl(bits as u32))
}

trait ElementTrait {
    type E;

    // utility
    fn val_expand(v: Self::E, bits: usize) -> Self::E;

    // a1+a2+a3+...
    fn sum_bits(self, bits: usize) -> Option<Self::E>;

    // (a1 OP b1), (a2 OP b2), (a3 OP b3),...
    fn add_bits(self, b: Self::E, bits: usize) -> Option<Self::E>;
    fn sub_bits(self, b: Self::E, bits: usize) -> Option<Self::E>;

    // (a1 OP b), (a2 OP b), (a3 OP b),...
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
    /// bit width
    pub bits: usize,
    /// number of element
    pub length: usize,
    /// data store
    data: Vec<Element>,
}

/// iterator for IntArray
pub struct IntIter<'a> {
    /// current position
    range: Range<usize>,
    /// refere to
    a: &'a IntArray,
}

impl IntArray {
    fn sizeval(b: usize, len: usize) -> (usize, usize) {
        let bpd = ELEMENT_BITS / b;
        return (bpd, (len + bpd - 1) / bpd);
    }
    pub fn new(b: usize, len: usize) -> IntArray {
        let (bpd, cap) = IntArray::sizeval(b, len);
        debug!("bpd={}, cap={}", bpd, cap);
        IntArray {
            bits: b,
            length: len,
            data: vec![0; cap as usize],
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

    pub fn new_with_iter<'a, I>(b: usize, vals: I) -> IntArray
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
            data: vec![0; cap as usize],
        };
        for v in vals {
            if cnt >= res.length {
                res.resize(res.length + UNIT);
            }
            res.set(cnt, v).unwrap();
            cnt += 1
        }
        // truncate
        res.resize(cnt);
        res
    }

    pub fn subarray<'a>(&'a self, offset: usize, length: usize) -> IntArray {
        let mut res = IntArray::new(self.bits, length);
        let (bpd, _) = IntArray::sizeval(self.bits, 0);
        if offset % bpd == 0 {
            // fast path
            debug!("fast path: offset={}, length={}", offset, length);
            for i in 0..(length / bpd) {
                res.data[i] = self.data[offset / bpd + i];
            }
            // rest
            if length % bpd != 0 {
                let i = length / bpd;
                let mask = (1 << (bpd * (length % bpd))) - 1;
                res.data[i] = self.data[offset / bpd + i] & (!mask);
            }
        } else {
            // slow path
            debug!("slow path: offset={}, length={}", offset, length);
            for i in 0..length {
                res.set(i, self.get(offset + i).unwrap()).unwrap();
            }
        }
        res
    }

    pub fn datasize(self) -> usize {
        mem::size_of::<IntArray>() + (ELEMENT_BITS / 8) * self.data.capacity()
    }

    pub fn push(&mut self, v: Element) -> Option<usize> {
        self.resize(self.length + 1);
        match self.set(self.length - 1, v) {
            Ok(_) => Some(self.length - 1),
            Err(_) => None,
        }
    }

    pub fn extend<I>(&mut self, vals: I)
    where
        I: Iterator<Item = Element>,
    {
        for v in vals {
            self.push(v).unwrap();
        }
    }

    pub fn extend_array(&mut self, vals: IntArray) {
        let (bpd, _) = IntArray::sizeval(self.bits, 0);
        if vals.bits == self.bits && self.length % bpd == 0 {
            // fast path
            debug!("fast path: bits={}, length={}", self.bits, self.length);
            self.resize(self.length + vals.length);
            self.data.extend(vals.data);
            return;
        }
        // slow path
        debug!("slow path: bits={}, length={}", self.bits, self.length);
        self.extend(vals.iter());
    }

    pub fn concat(&mut self, vals: IntArray) {
        self.extend_array(vals);
    }

    pub fn shape<'a>(self: &'a IntArray, bits: usize) -> IntArray {
        IntArray::new_with_iter(bits, self.iter())
    }

    pub fn shape_auto<'a>(self: &'a IntArray) -> IntArray {
        let mv = self.iter().max().unwrap();
        let bits = match mv.ffs() {
            0 => 1,
            n => n,
        };
        IntArray::new_with_iter(bits, self.iter())
    }

    pub fn pop(&mut self) -> Result<Element, String> {
        let res = self.get(self.length - 1);
        self.resize(self.length - 1);
        res
    }

    pub fn iter(&self) -> IntIter {
        IntIter {
            range: 0..self.length,
            a: &self,
        }
    }

    pub fn capacity(&self) -> usize {
        let bpd = ELEMENT_BITS / self.bits;
        self.data.len() * bpd
    }

    pub fn resize(&mut self, len: usize) {
        let bpd = ELEMENT_BITS / self.bits;
        let cap = (len + bpd - 1) / bpd;
        self.length = len;
        self.data.resize(cap, 0);
    }

    pub fn len(&self) -> usize {
        self.length
    }

    pub fn max_value(&self) -> Element {
        if self.bits == ELEMENT_BITS {
            return Element::max_value();
        }
        ((1 as Element) << self.bits) - 1
    }

    pub fn max(&self) -> Element {
        self.iter().max().unwrap()
    }

    pub fn min(&self) -> Element {
        self.iter().min().unwrap()
    }

    pub fn average(&self) -> f64 {
        self.sum().unwrap() as f64 / self.len() as f64
    }

    fn getoffset(&self, i: usize) -> (usize, usize, usize) {
        let bpd = ELEMENT_BITS / self.bits as usize;
        return (bpd, i / bpd, i % bpd);
    }

    pub fn get(&self, i: usize) -> Result<Element, String> {
        if self.length <= i {
            return Err("OB".to_owned());
        }
        // debug!("get {}/{}", i, self.capacity());
        let (_, idx, iv) = self.getoffset(i);
        let vv = self.data[idx];
        // debug!("idx={}, iv={}, vv={}", idx, iv, vv);
        let res = (vv >> (iv * self.bits as usize)) & self.max_value();
        Ok(res)
    }

    pub fn set(&mut self, i: usize, v: Element) -> Result<Element, String> {
        if self.max_value() < v {
            return Err("TooLarge".to_owned());
        }
        if self.length <= i {
            return Err("OutOfBounds".to_owned());
        }
        let (_, idx, iv) = self.getoffset(i);
        let mask1 = (self.max_value()) << (iv * self.bits);
        let mask2 = v << (iv * self.bits);
        let res = self.max_value() & (self.data[idx] >> (iv * self.bits));
        self.data[idx] = (self.data[idx] & (!mask1)) | mask2;
        Ok(res)
    }

    pub fn add(&mut self, i: usize, v: Element) -> Result<Element, String> {
        match self.get(i) {
            Ok(n) => self.set(i, n + v),
            Err(e) => return Err(e),
        }
    }

    pub fn sub(&mut self, i: usize, v: Element) -> Result<Element, String> {
        match self.get(i) {
            Ok(n) => self.set(i, n - v),
            Err(e) => return Err(e),
        }
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
                if n != 0 && n != self.max_value() {
                    self.set(i, n - 1).unwrap();
                    Some(n)
                } else {
                    None
                }
            }
            Err(_e) => None,
        }
    }

    pub fn incr(&mut self, i: usize) -> Result<Element, String> {
        self.add(i, 1)
    }

    pub fn decr(&mut self, i: usize) -> Result<Element, String> {
        self.sub(i, 1)
    }

    pub fn sum<'a>(&'a self) -> Option<Element> {
        let mut res = 0;
        if self.bits > ELEMENT_BITS / 2 {
            return self.sum0();
        }
        for i in self.data.iter() {
            res += (*i).sum_bits(self.bits).unwrap();
            debug!("sum: {} -> {}", *i, res);
        }
        Some(res)
    }

    pub fn sum0<'a>(&'a self) -> Option<Element> {
        return Some(self.iter().fold(0 as Element, |sum, a| sum + a));
    }

    pub fn fill_random(&mut self) {
        let mut rng = rand::thread_rng();
        if ELEMENT_BITS % self.bits == 0 {
            for i in 0..(self.data.len() - 1) {
                self.data[i] = rng.gen();
            }
        } else {
            let mvbits = ELEMENT_BITS - (ELEMENT_BITS % self.bits);
            let mvval = (1 as Element) << mvbits;
            for i in 0..(self.data.len() - 1) {
                self.data[i] = rng.gen_range(0, mvval);
            }
        }
        let bpd = ELEMENT_BITS / self.bits;
        for i in (self.data.len() - 1) * bpd..self.length {
            self.set(i, rng.gen_range(0, self.max_value())).unwrap();
        }
    }
}

impl<'a> Iterator for IntIter<'a> {
    type Item = Element;
    fn next(&mut self) -> Option<Element> {
        self.range.next().map(|i| self.a.get(i).unwrap())
    }

    fn count(self) -> usize {
        self.a.length
    }
}

impl fmt::Display for IntArray {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}[{}]=", self.bits, self.capacity()).unwrap();
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

impl AddAssign<u64> for IntArray {
    fn add_assign(&mut self, v: Element) {
        let (bpd, _) = IntArray::sizeval(self.bits, 0);
        for i in 0..(self.length / bpd) {
            self.data[i] = self.data[i].addval_bits(v, self.bits).unwrap();
        }
        if self.length % bpd != 0 {
            let i = self.length / bpd;
            let mask = ((1 as Element) << (bpd * (self.length % bpd))) - 1;
            self.data[i] = self.data[i].addval_bits(v, self.bits).unwrap() & mask;
        }
    }
}

impl AddAssign<IntArray> for IntArray {
    fn add_assign(&mut self, v: IntArray) {
        if self.bits == v.bits && self.length == v.length {
            // fast path
            debug!("fast path: bits={}, length={}", self.bits, self.length);
            let (bpd, _) = IntArray::sizeval(self.bits, 0);
            for i in 0..(self.length / bpd) {
                self.data[i] = self.data[i].add_bits(v.data[i], self.bits).unwrap();
            }
            if self.length % bpd != 0 {
                let i = self.length / bpd;
                let mask = ((1 as Element) << (bpd * (self.length % bpd))) - 1;
                self.data[i] = self.data[i].add_bits(v.data[i], self.bits).unwrap() & mask;
            }
            return;
        }
        // slow path
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
        if self.length % bpd != 0 {
            let i = self.length / bpd;
            let mask = ((1 as Element) << (bpd * (self.length % bpd))) - 1;
            self.data[i] = (self.data[i] | (!mask)).subval_bits(v, self.bits).unwrap() & mask;
        }
    }
}

impl SubAssign<IntArray> for IntArray {
    fn sub_assign(&mut self, v: IntArray) {
        if self.bits == v.bits && self.length == v.length {
            // fast path
            debug!("fast path: bits={}, length={}", self.bits, self.length);
            let (bpd, _) = IntArray::sizeval(self.bits, 0);
            for i in 0..(self.length / bpd) {
                self.data[i] = self.data[i].sub_bits(v.data[i], self.bits).unwrap();
            }
            if self.length % bpd != 0 {
                let i = self.length / bpd;
                let mask = ((1 as Element) << (bpd * (self.length % bpd))) - 1;
                self.data[i] = (self.data[i] | (!mask))
                    .sub_bits(v.data[i], self.bits)
                    .unwrap()
                    & mask;
            }
            return;
        }
        // slow path
        debug!(
            "slow path: bits={}/{}, length={}/{}",
            self.bits, v.bits, self.length, v.length
        );
        for i in 0..cmp::min(self.length, v.length) {
            self.sub(i, v.get(i).unwrap()).unwrap();
        }
    }
}

impl MulAssign<u64> for IntArray {
    fn mul_assign(&mut self, v: Element) {
        let (bpd, _) = IntArray::sizeval(self.bits, 0);
        for i in 0..(self.length / bpd) {
            self.data[i] = self.data[i].mulval_bits(v, self.bits).unwrap();
        }
        if self.length % bpd != 0 {
            let i = self.length / bpd;
            let mask = ((1 as Element) << (bpd * (self.length % bpd))) - 1;
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
