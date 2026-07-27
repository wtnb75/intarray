use num_bigint::{BigInt, BigUint, Sign};
use num_traits::Zero;

pub(crate) const DEFAULT_K: usize = 64;

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct BitBlock {
    pub(crate) data: Vec<u8>,
    pub(crate) bit_len: usize,
    pub(crate) count: usize,
}

impl BitBlock {
    pub(crate) fn empty() -> Self {
        BitBlock {
            data: vec![],
            bit_len: 0,
            count: 0,
        }
    }
}

pub(crate) fn read_bit(data: &[u8], bit_pos: usize) -> u8 {
    let byte_idx = bit_pos / 8;
    let bit_idx = 7 - (bit_pos % 8);
    (data[byte_idx] >> bit_idx) & 1
}

pub(crate) fn write_bit(data: &mut Vec<u8>, bit_len: &mut usize, bit: u8) {
    if (*bit_len).is_multiple_of(8) {
        data.push(0u8);
    }
    let byte_idx = *bit_len / 8;
    let bit_idx = 7 - (*bit_len % 8);
    data[byte_idx] |= bit << bit_idx;
    *bit_len += 1;
}

pub(crate) fn zigzag_encode(n: &BigInt) -> BigUint {
    match n.sign() {
        Sign::NoSign => BigUint::zero(),
        Sign::Plus => n.magnitude().clone() << 1,
        Sign::Minus => (n.magnitude().clone() << 1) - 1u32,
    }
}

pub(crate) fn zigzag_decode(z: BigUint) -> BigInt {
    if z.bit(0) {
        -BigInt::from((z + 1u32) / 2u32)
    } else {
        BigInt::from(z >> 1)
    }
}

// k = floor(log2(n)): number of bits in n's representation minus 1
pub(crate) fn write_elias_gamma(data: &mut Vec<u8>, bit_len: &mut usize, n: usize) {
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

pub(crate) fn read_elias_gamma(data: &[u8], bit_pos: &mut usize) -> usize {
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

pub(crate) fn write_biguint(data: &mut Vec<u8>, bit_len: &mut usize, z: &BigUint, b: usize) {
    for i in (0..b).rev() {
        write_bit(data, bit_len, u8::from(z.bit(i as u64)));
    }
}

pub(crate) fn read_biguint(data: &[u8], bit_pos: &mut usize, b: usize) -> BigUint {
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
