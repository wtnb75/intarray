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
        let (mask1, mask2) = get_masks(bits);
        let r1 = (self & mask1) + (b & mask1);
        let r2 = (self & mask2) + (b & mask2);
        if (r1 & mask2) != 0 || (r2 & mask1) != 0 {
            None
        } else {
            Some(r1 + r2)
        }
    }

    fn sub_bits(self, b: Self::E, bits: usize) -> Option<Self::E> {
        let (mask1, mask2) = get_masks(bits);
        let r1 = (self & mask1) - (b & mask1);
        let r2 = (self & mask2) - (b & mask2);
        if (r1 & mask2) != 0 || (r2 & mask1) != 0 {
            None
        } else {
            Some(r1 + r2)
        }
    }

    fn mulval_bits(self, b: Self::E, bits: usize) -> Option<Self::E> {
        let (mask1, mask2) = get_masks(bits);
        let r1 = (self & mask1) * b;
        let r2 = (self & mask2) * b;
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
    pub bits: u8,
    /// number of element
    pub length: usize,
    /// data store
    data: Vec<u64>,
}

/// iterator for IntArray
pub struct IntIter<'a> {
    /// current position
    range: Range<usize>,
    /// refere to
    a: &'a IntArray,
}

fn sum_mask(a: u64, bits: usize, mask: u64) -> u64 {
    let a1 = a & mask;
    let a2 = (a >> bits) & mask;
    debug!(
        "sum_mask(a={:064b}, bits={}, mask={:b}) {:x} + {:x} = {:x}",
        a,
        bits,
        mask,
        a1,
        a2,
        a1 + a2
    );
    a1 + a2
}

const MASK: [u64; 31] = [
    0x5555_5555_5555_5555, // 1
    0x3333_3333_3333_3333, // 2
    0b000111_000111_000111_000111_000111_000111_000111_000111_000111_000111_000111, // 3
    0x0f0f_0f0f_0f0f_0f0f, // 4
    0b1111_00000_11111_00000_11111_00000_11111_00000_11111_00000_11111_00000_11111, // 5
    0b1111_000000_111111_000000_111111_000000_111111_000000_111111_000000_111111, // 6
    0b0_1111111_0000000_1111111_0000000_1111111_0000000_1111111_0000000_1111111, // 7
    0x00ff_00ff_00ff_00ff, // 8
    0b111111111_000000000_111111111_000000000_111111111_000000000_111111111, // 9
    0b1111_0000000000_1111111111_0000000000_1111111111_0000000000_1111111111, // 10
    0b000000000_11111111111_00000000000_11111111111_00000000000_11111111111, // 11
    0b0000_111111111111_000000000000_111111111111_000000000000_111111111111, // 12
    0b111111111111_0000000000000_1111111111111_0000000000000_1111111111111, // 13
    0b11111111_00000000000000_11111111111111_00000000000000_11111111111111, // 14
    0b1111_000000000000000_111111111111111_000000000000000_111111111111111, // 15
    0b0000000000000000_1111111111111111_0000000000000000_1111111111111111, // 16
    0b000000000000_11111111111111111_00000000000000000_11111111111111111, // 17
    0b000000000_111111111111111111_000000000000000000_111111111111111111, // 18
    0b000000_1111111111111111111_0000000000000000000_1111111111111111111, // 19
    0b000_11111111111111111111_00000000000000000000_11111111111111111111, // 20
    0b0111111111111111111111_000000000000000000000_111111111111111111111, // 21
    0b11111111111111111111_0000000000000000000000_1111111111111111111111, // 22
    0b111111111111111111_00000000000000000000000_11111111111111111111111, // 23
    0b1111111111111111_000000000000000000000000_111111111111111111111111, // 24
    0b11111111111111_0000000000000000000000000_1111111111111111111111111, // 25
    0b111111111111_00000000000000000000000000_11111111111111111111111111, // 26
    0b1111111111_000000000000000000000000000_111111111111111111111111111, // 27
    0b11111111_0000000000000000000000000000_1111111111111111111111111111, // 28
    0b111111_00000000000000000000000000000_11111111111111111111111111111, // 29
    0b1111_000000000000000000000000000000_111111111111111111111111111111, // 30
    0b11_0000000000000000000000000000000_1111111111111111111111111111111, // 31
];

fn sum_64(a: u64, bits: usize) -> Option<u64> {
    if bits >= 64 {
        debug!("return self: bits={}, res={}", bits, a);
        return Some(a);
    }
    if bits == 0 {
        debug!("invalid argument: bits={}", bits);
        return None;
    }
    if (bits - 1) < MASK.len() {
        let res = sum_64(sum_mask(a, bits, MASK[bits - 1]), bits * 2);
        debug!(
            "return1: bits={}, mask={:b}, val={:b}, res={:b}",
            bits,
            MASK[bits - 1],
            a,
            res.unwrap(),
        );
        return res;
    }
    sum_64(sum_mask(a, bits, (1u64 << bits) - 1), bits * 2)
}

fn zebra_mask(v: u64, bits: u8) -> (u64, u64) {
    let mask_a = MASK[bits as usize - 1];
    let mut add_a = 0u64;
    for _ in 0..(64 / 2 / bits as usize) {
        add_a <<= bits * 2;
        add_a |= v;
    }
    (mask_a, add_a)
}

fn val2mask(v: u64, bits: u8) -> u64 {
    let mut v1 = 0u64;
    for _ in 0..(64 / bits as usize) {
        v1 <<= bits;
        v1 |= v;
    }
    v1
}

fn add_64(a: u64, b: u64, bits: u8) -> Option<u64> {
    let b1 = val2mask(b, bits);
    add2_64(a, b1, bits)
}

fn add2_64(a: u64, b: u64, bits: u8) -> Option<u64> {
    let mask1 = MASK[bits as usize - 1];
    let mask2 = mask1 << (bits as usize);
    let a1 = a & mask1;
    let a2 = a & mask2;
    let b1 = b & mask1;
    let b2 = b & mask2;
    let v1 = a1 + b1;
    let v2 = a2 + b2;
    if (v1 & mask2) != 0 || (v2 & mask1) != 0 {
        None
    } else {
        Some(v1 + v2)
    }
}

fn sub_64(a: u64, b: u64, bits: u8) -> Option<u64> {
    let b1 = val2mask(b, bits);
    sub2_64(a, b1, bits)
}

fn sub2_64(a: u64, b: u64, bits: u8) -> Option<u64> {
    let mask1 = MASK[bits as usize - 1];
    let mask2 = mask1 << (bits as usize);
    let a1 = a & mask1;
    let a2 = a & mask2;
    let b1 = b & mask1;
    let b2 = b & mask2;
    let v1 = a1 - b1;
    let v2 = a2 - b2;
    if (v1 & mask2) != 0 || (v2 & mask1) != 0 {
        None
    } else {
        Some(v1 + v2)
    }
}

fn mul_64(a: u64, b: u64, bits: u8) -> Option<u64> {
    let mask1 = (1 << (bits as usize)) - 1;
    let b1 = b & mask1;
    let (mask_a, _) = zebra_mask(b1, bits);
    let mask_b = mask_a << (bits as usize);
    let v1 = (a & mask_a) * b1;
    let v2 = (a & mask_b) * b1;
    if (v1 & mask_b) != 0 || (v2 & mask_a) != 0 {
        None
    } else {
        Some(v1 + v2)
    }
}

fn bits(a: u64) -> usize {
    64 - a.leading_zeros() as usize
}

impl IntArray {
    fn sizeval(b: u8, len: usize) -> (usize, usize) {
        let bpd = 64 / b as usize;
        return (bpd, (len + bpd - 1) / bpd);
    }
    pub fn new(b: u8, len: usize) -> IntArray {
        let (bpd, cap) = IntArray::sizeval(b, len);
        debug!("bpd={}, cap={}", bpd, cap);
        IntArray {
            bits: b,
            length: len,
            data: vec![0; cap as usize],
        }
    }

    pub fn new_with_vec(b: u8, vals: Vec<u64>) -> IntArray {
        let (bpd, cap) = IntArray::sizeval(b, vals.len());
        debug!("bpd={}, cap={}", bpd, cap);
        let mut res = IntArray {
            bits: b,
            length: vals.len(),
            data: vec![0; cap as usize],
        };
        for (i, v) in vals.iter().enumerate() {
            res.set(i, *v).unwrap();
        }
        res
    }

    pub fn new_with_iter<'a, I>(b: u8, vals: I) -> IntArray
    where
        I: Iterator<Item = u64>,
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
        mem::size_of::<IntArray>() + mem::size_of::<u64>() * self.data.capacity()
    }

    pub fn push(&mut self, v: u64) -> Option<usize> {
        self.resize(self.length + 1);
        match self.set(self.length - 1, v) {
            Ok(_) => Some(self.length - 1),
            Err(_) => None,
        }
    }

    pub fn extend<I>(&mut self, vals: I)
    where
        I: Iterator<Item = u64>,
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

    pub fn shape<'a>(self: &'a IntArray, bits: u8) -> IntArray {
        IntArray::new_with_iter(bits, self.iter())
    }

    pub fn shape_auto<'a>(self: &'a IntArray) -> IntArray {
        let mv = self.iter().max().unwrap();
        let bits = match bits(mv) as u8 {
            0 => 1,
            n => n,
        };
        IntArray::new_with_iter(bits, self.iter())
    }

    pub fn pop(&mut self) -> Result<u64, String> {
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
        let bpd = 64 / self.bits as usize;
        self.data.len() * bpd
    }

    pub fn resize(&mut self, len: usize) {
        let bpd = 64 / self.bits as usize;
        let cap = (len + bpd - 1) / bpd;
        self.length = len;
        self.data.resize(cap, 0);
    }

    pub fn len(&self) -> usize {
        self.length
    }

    pub fn max_value(&self) -> u64 {
        if self.bits == 64 {
            return u64::max_value();
        }
        (1u64 << self.bits) - 1
    }

    pub fn max(&self) -> u64 {
        self.iter().max().unwrap()
    }

    pub fn min(&self) -> u64 {
        self.iter().min().unwrap()
    }

    pub fn average(&self) -> f64 {
        self.sum().unwrap() as f64 / self.len() as f64
    }

    fn getoffset(&self, i: usize) -> (usize, usize, usize) {
        let bpd = 64 / self.bits as usize;
        return (bpd, i / bpd, i % bpd);
    }

    pub fn get(&self, i: usize) -> Result<u64, String> {
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

    pub fn set(&mut self, i: usize, v: u64) -> Result<u64, String> {
        if self.max_value() < v {
            return Err("TooLarge".to_owned());
        }
        if self.length <= i {
            return Err("OutOfBounds".to_owned());
        }
        let (_, idx, iv) = self.getoffset(i);
        let mask1 = (self.max_value()) << (iv * self.bits as usize);
        let mask2 = v << (iv * self.bits as usize);
        let res = self.max_value() & (self.data[idx] >> (iv * self.bits as usize));
        self.data[idx] = (self.data[idx] & (!mask1)) | mask2;
        Ok(res)
    }

    pub fn add(&mut self, i: usize, v: u64) -> Result<u64, String> {
        match self.get(i) {
            Ok(n) => self.set(i, n + v),
            Err(e) => return Err(e),
        }
    }

    pub fn sub(&mut self, i: usize, v: u64) -> Result<u64, String> {
        match self.get(i) {
            Ok(n) => self.set(i, n - v),
            Err(e) => return Err(e),
        }
    }

    pub fn incr_limit(&mut self, i: usize) -> Option<u64> {
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

    pub fn decr_limit(&mut self, i: usize) -> Option<u64> {
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

    pub fn incr(&mut self, i: usize) -> Result<u64, String> {
        self.add(i, 1)
    }

    pub fn decr(&mut self, i: usize) -> Result<u64, String> {
        self.sub(i, 1)
    }

    pub fn sum<'a>(&'a self) -> Option<u64> {
        let mut res = 0;
        if self.bits > 32 {
            return self.sum0();
        }
        for i in self.data.iter() {
            res += sum_64(*i, self.bits as usize).unwrap();
            debug!("sum: {} -> {}", *i, res);
        }
        Some(res)
    }

    pub fn sum0<'a>(&'a self) -> Option<u64> {
        return Some(self.iter().fold(0u64, |sum, a| sum + a as u64));
    }

    pub fn fill_random(&mut self) {
        let mut rng = rand::thread_rng();
        if 64 % self.bits == 0 {
            for i in 0..(self.data.len() - 1) {
                self.data[i] = rng.gen();
            }
        } else {
            let mvbits = 64 - (64 % self.bits);
            let mvval = 1u64 << mvbits;
            for i in 0..(self.data.len() - 1) {
                self.data[i] = rng.gen_range(0, mvval);
            }
        }
        let bpd = 64 / self.bits as usize;
        for i in (self.data.len() - 1) * bpd..self.length {
            self.set(i, rng.gen_range(0, self.max_value())).unwrap();
        }
    }
}

impl<'a> Iterator for IntIter<'a> {
    type Item = u64;
    fn next(&mut self) -> Option<u64> {
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
    fn add_assign(&mut self, v: u64) {
        let (bpd, _) = IntArray::sizeval(self.bits, 0);
        for i in 0..(self.length / bpd) {
            self.data[i] = add_64(self.data[i], v, self.bits).unwrap();
        }
        if self.length % bpd != 0 {
            let i = self.length / bpd;
            let mask = (1u64 << (bpd * (self.length % bpd))) - 1;
            self.data[i] = add_64(self.data[i] & (mask), v, self.bits).unwrap() & mask;
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
                self.data[i] = add2_64(self.data[i], v.data[i], self.bits).unwrap();
            }
            if self.length % bpd != 0 {
                let i = self.length / bpd;
                let mask = (1u64 << (bpd * (self.length % bpd))) - 1;
                self.data[i] = add2_64(self.data[i] & (mask), v.data[i], self.bits).unwrap() & mask;
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

impl SubAssign<u64> for IntArray {
    fn sub_assign(&mut self, v: u64) {
        let (bpd, _) = IntArray::sizeval(self.bits, 0);
        for i in 0..(self.length / bpd) {
            self.data[i] = sub_64(self.data[i], v, self.bits).unwrap();
        }
        if self.length % bpd != 0 {
            let i = self.length / bpd;
            let mask = (1u64 << (bpd * (self.length % bpd))) - 1;
            self.data[i] = sub_64(self.data[i] | (!mask), v, self.bits).unwrap() & mask;
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
                self.data[i] = sub2_64(self.data[i], v.data[i], self.bits).unwrap();
            }
            if self.length % bpd != 0 {
                let i = self.length / bpd;
                let mask = (1u64 << (bpd * (self.length % bpd))) - 1;
                self.data[i] =
                    sub2_64(self.data[i] | (!mask), v.data[i], self.bits).unwrap() & mask;
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
    fn mul_assign(&mut self, v: u64) {
        let (bpd, _) = IntArray::sizeval(self.bits, 0);
        for i in 0..(self.length / bpd) {
            self.data[i] = mul_64(self.data[i], v, self.bits).unwrap();
        }
        if self.length % bpd != 0 {
            let i = self.length / bpd;
            let mask = (1u64 << (bpd * (self.length % bpd))) - 1;
            self.data[i] = mul_64(self.data[i] & (mask), v, self.bits).unwrap() & mask;
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
