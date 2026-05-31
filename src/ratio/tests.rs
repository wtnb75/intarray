use super::*;
use crate::error::ArrayError;
use num_bigint::BigInt;
use num_rational::Ratio;
use test_log::test;

fn r(p: i64, q: i64) -> Ratio<BigInt> {
    Ratio::new(BigInt::from(p), BigInt::from(q))
}

fn ri(n: i64) -> Ratio<BigInt> {
    Ratio::from_integer(BigInt::from(n))
}

// --- validation ---

#[test]
fn invalid_k_zero() {
    assert_eq!(RatioArray::new(0), Err(ArrayError::InvalidRange));
}

#[test]
fn valid_k_one() {
    assert!(RatioArray::new(1).is_ok());
}

#[test]
fn valid_k_64() {
    assert!(RatioArray::new(64).is_ok());
}

// --- construction ---

#[test]
fn new_empty() {
    let arr = RatioArray::new(4).unwrap();
    assert_eq!(arr.len(), 0);
    assert!(arr.is_empty());
    assert_eq!(arr.block_count(), 0);
    assert_eq!(arr.block_size(), 4);
}

#[test]
fn new_with_vec_basic() {
    let arr = RatioArray::new_with_vec(4, vec![r(0, 1), r(1, 2), r(-1, 3)]).unwrap();
    assert_eq!(arr.len(), 3);
    assert_eq!(arr.get(0).unwrap(), ri(0));
    assert_eq!(arr.get(1).unwrap(), r(1, 2));
    assert_eq!(arr.get(2).unwrap(), r(-1, 3));
}

#[test]
fn new_with_iter_basic() {
    let arr = RatioArray::new_with_iter(4, [r(0, 1), r(1, 2), r(-1, 3)]).unwrap();
    assert_eq!(arr.len(), 3);
    assert_eq!(arr.get(2).unwrap(), r(-1, 3));
}

// --- encoding correctness: pin specific bit patterns ---

#[test]
fn spec_block_example() {
    // [0, 1/2, -1] in k=3 block:
    //   "11" + "01110010" + "01011" = 15 bits
    //   data[0] = 0b11011100 = 0xDC
    //   data[1] = 0b10010110 = 0x96
    let arr = RatioArray::new_with_vec(3, vec![ri(0), r(1, 2), ri(-1)]).unwrap();
    assert_eq!(arr.block_count(), 1);
    let block = &arr.blocks[0];
    assert_eq!(block.count, 3);
    assert_eq!(block.bit_len, 15);
    assert_eq!(block.data[0], 0xDC);
    assert_eq!(block.data[1], 0x96);
}

#[test]
fn encode_decode_zero() {
    let mut arr = RatioArray::new(64).unwrap();
    arr.push(ri(0)).unwrap();
    assert_eq!(arr.get(0).unwrap(), ri(0));
}

#[test]
fn encode_decode_one() {
    let mut arr = RatioArray::new(64).unwrap();
    arr.push(ri(1)).unwrap();
    assert_eq!(arr.get(0).unwrap(), ri(1));
}

#[test]
fn encode_decode_negative_one() {
    let mut arr = RatioArray::new(64).unwrap();
    arr.push(ri(-1)).unwrap();
    assert_eq!(arr.get(0).unwrap(), ri(-1));
}

#[test]
fn encode_decode_fraction_one_half() {
    let mut arr = RatioArray::new(64).unwrap();
    arr.push(r(1, 2)).unwrap();
    assert_eq!(arr.get(0).unwrap(), r(1, 2));
}

#[test]
fn encode_decode_negative_fraction() {
    let mut arr = RatioArray::new(64).unwrap();
    arr.push(r(-1, 3)).unwrap();
    assert_eq!(arr.get(0).unwrap(), r(-1, 3));
}

#[test]
fn encode_decode_22_over_7() {
    let mut arr = RatioArray::new(64).unwrap();
    arr.push(r(22, 7)).unwrap();
    assert_eq!(arr.get(0).unwrap(), r(22, 7));
}

#[test]
fn push_normalizes_fraction() {
    // 2/4 is reduced to 1/2
    let mut arr = RatioArray::new(64).unwrap();
    arr.push(Ratio::new(BigInt::from(2), BigInt::from(4))).unwrap();
    assert_eq!(arr.get(0).unwrap(), r(1, 2));
}

#[test]
fn encode_decode_large_integer() {
    let big = BigInt::from(i64::MAX);
    let v = Ratio::from_integer(big.clone());
    let mut arr = RatioArray::new(64).unwrap();
    arr.push(v).unwrap();
    assert_eq!(arr.get(0).unwrap(), Ratio::from_integer(big));
}

// --- get / set ---

#[test]
fn get_out_of_bounds() {
    let arr = RatioArray::new(4).unwrap();
    assert_eq!(arr.get(0), Err(ArrayError::OutOfBounds));
}

#[test]
fn get_out_of_bounds_usize_max() {
    let arr = RatioArray::new(4).unwrap();
    assert_eq!(arr.get(usize::MAX), Err(ArrayError::OutOfBounds));
}

#[test]
fn set_basic() {
    let mut arr = RatioArray::new_with_vec(4, vec![ri(1), ri(2), ri(3)]).unwrap();
    arr.set(1, r(1, 2)).unwrap();
    assert_eq!(arr.get(0).unwrap(), ri(1));
    assert_eq!(arr.get(1).unwrap(), r(1, 2));
    assert_eq!(arr.get(2).unwrap(), ri(3));
    assert_eq!(arr.len(), 3);
}

#[test]
fn set_out_of_bounds() {
    let mut arr = RatioArray::new(4).unwrap();
    assert_eq!(arr.set(0, ri(1)), Err(ArrayError::OutOfBounds));
}

// --- push / pop ---

#[test]
fn push_returns_index() {
    let mut arr = RatioArray::new(4).unwrap();
    assert_eq!(arr.push(ri(0)).unwrap(), 0);
    assert_eq!(arr.push(r(1, 2)).unwrap(), 1);
    assert_eq!(arr.push(ri(-1)).unwrap(), 2);
}

#[test]
fn pop_returns_last() {
    let mut arr = RatioArray::new_with_vec(4, vec![ri(1), r(1, 2), ri(-1)]).unwrap();
    assert_eq!(arr.pop().unwrap(), ri(-1));
    assert_eq!(arr.len(), 2);
    assert_eq!(arr.pop().unwrap(), r(1, 2));
    assert_eq!(arr.pop().unwrap(), ri(1));
    assert!(arr.is_empty());
}

#[test]
fn pop_empty_error() {
    let mut arr = RatioArray::new(4).unwrap();
    assert_eq!(arr.pop(), Err(ArrayError::Empty));
}

#[test]
fn push_pop_block_boundary() {
    let mut arr = RatioArray::new(2).unwrap();
    arr.push(ri(1)).unwrap();
    arr.push(ri(2)).unwrap();
    arr.push(ri(3)).unwrap(); // starts second block
    assert_eq!(arr.block_count(), 2);
    assert_eq!(arr.pop().unwrap(), ri(3));
    assert_eq!(arr.block_count(), 1);
}

#[test]
fn k1_single_ratio() {
    let mut arr = RatioArray::new(1).unwrap();
    arr.push(r(1, 2)).unwrap();
    assert_eq!(arr.get(0).unwrap(), r(1, 2));
    assert_eq!(arr.block_count(), 1);
    arr.push(r(3, 4)).unwrap();
    assert_eq!(arr.block_count(), 2);
}

// --- extend ---

#[test]
fn extend_basic() {
    let mut arr = RatioArray::new(4).unwrap();
    arr.extend([ri(1), r(1, 2), ri(-1)]).unwrap();
    assert_eq!(arr.len(), 3);
    assert_eq!(arr.get(1).unwrap(), r(1, 2));
}

#[test]
fn extend_array_basic() {
    let a = RatioArray::new_with_vec(4, vec![ri(1), r(1, 2)]).unwrap();
    let mut b = RatioArray::new(8).unwrap();
    b.extend_array(&a).unwrap();
    assert_eq!(b.len(), 2);
    assert_eq!(b.get(0).unwrap(), ri(1));
    assert_eq!(b.get(1).unwrap(), r(1, 2));
}

#[test]
fn extend_array_empty_source() {
    let a = RatioArray::new(4).unwrap();
    let mut b = RatioArray::new_with_vec(4, vec![ri(1)]).unwrap();
    b.extend_array(&a).unwrap();
    assert_eq!(b.len(), 1);
}

// --- iterator ---

#[test]
fn iter_basic() {
    let arr = RatioArray::new_with_vec(4, vec![ri(0), r(1, 2), ri(-1)]).unwrap();
    let collected: Vec<_> = arr.iter().collect();
    assert_eq!(collected, vec![ri(0), r(1, 2), ri(-1)]);
}

#[test]
fn iter_exact_size() {
    let arr = RatioArray::new_with_vec(4, vec![ri(1), r(1, 2), ri(-1)]).unwrap();
    let mut it = arr.iter();
    assert_eq!(it.len(), 3);
    it.next();
    assert_eq!(it.len(), 2);
}

#[test]
fn iter_all_full_blocks() {
    // k=2, 4 elements → 2 full blocks
    let arr = RatioArray::new_with_vec(2, vec![ri(1), r(1, 2), ri(-1), r(3, 4)]).unwrap();
    assert_eq!(arr.block_count(), 2);
    let collected: Vec<_> = arr.iter().collect();
    assert_eq!(collected, vec![ri(1), r(1, 2), ri(-1), r(3, 4)]);
}

#[test]
fn iter_empty() {
    let arr = RatioArray::new(4).unwrap();
    assert_eq!(arr.iter().next(), None);
}

// --- stats ---

#[test]
fn sum_empty() {
    let arr = RatioArray::new(4).unwrap();
    assert_eq!(arr.sum(), None);
}

#[test]
fn sum_basic() {
    // 1/2 + 1/3 + 1/6 = 1
    let arr = RatioArray::new_with_vec(4, vec![r(1, 2), r(1, 3), r(1, 6)]).unwrap();
    assert_eq!(arr.sum().unwrap(), ri(1));
}

#[test]
fn min_max_basic() {
    let arr = RatioArray::new_with_vec(4, vec![r(1, 3), r(1, 2), r(1, 6)]).unwrap();
    assert_eq!(arr.min().unwrap(), r(1, 6));
    assert_eq!(arr.max().unwrap(), r(1, 2));
}

#[test]
fn min_max_empty() {
    let arr = RatioArray::new(4).unwrap();
    assert_eq!(arr.min(), None);
    assert_eq!(arr.max(), None);
}

#[test]
fn average_basic() {
    // average of [1, 2, 3] = 2
    let arr = RatioArray::new_with_vec(4, vec![ri(1), ri(2), ri(3)]).unwrap();
    assert_eq!(arr.average().unwrap(), ri(2));
}

#[test]
fn average_fraction_result() {
    // average of [1/2, 1/2] = 1/2
    let arr = RatioArray::new_with_vec(4, vec![r(1, 2), r(1, 2)]).unwrap();
    assert_eq!(arr.average().unwrap(), r(1, 2));
}

#[test]
fn average_exact_rational() {
    // average of [1, 2] = 3/2 (exact, no floating-point error)
    let arr = RatioArray::new_with_vec(4, vec![ri(1), ri(2)]).unwrap();
    assert_eq!(arr.average().unwrap(), r(3, 2));
}

#[test]
fn average_empty() {
    let arr = RatioArray::new(4).unwrap();
    assert_eq!(arr.average(), None);
}

// --- display ---

#[test]
fn display_format() {
    let arr = RatioArray::new_with_vec(64, vec![ri(0), r(1, 2), r(-1, 3), r(22, 7)]).unwrap();
    assert_eq!(arr.to_string(), "[k=64][4]=0,1/2,-1/3,22/7");
}

#[test]
fn display_integers_no_slash() {
    // integers display as "n" not "n/1"
    let arr = RatioArray::new_with_vec(4, vec![ri(1), ri(-2), ri(3)]).unwrap();
    assert_eq!(arr.to_string(), "[k=4][3]=1,-2,3");
}

// --- serde ---

#[test]
fn serde_json_format() {
    use serde_json;
    let arr = RatioArray::new_with_vec(4, vec![ri(0), r(1, 2), r(-1, 3), r(22, 7)]).unwrap();
    let json = serde_json::to_string(&arr).unwrap();
    assert_eq!(json, r#"["0","1/2","-1/3","22/7"]"#);
}

#[test]
fn serde_json_round_trip() {
    use serde_json;
    let orig = RatioArray::new_with_vec(4, vec![ri(0), r(1, 2), r(-1, 3), r(22, 7)]).unwrap();
    let json = serde_json::to_string(&orig).unwrap();
    let loaded: RatioArray = serde_json::from_str(&json).unwrap();
    assert_eq!(orig, loaded);
}

#[test]
fn serde_json_integer_strings() {
    // integers serialize as "p" not "p/1"
    use serde_json;
    let arr = RatioArray::new_with_vec(4, vec![ri(1), ri(-2)]).unwrap();
    let json = serde_json::to_string(&arr).unwrap();
    assert_eq!(json, r#"["1","-2"]"#);
}

#[test]
fn serde_invalid_string_returns_error() {
    use serde_json;
    let result: Result<RatioArray, _> = serde_json::from_str(r#"["not_a_ratio"]"#);
    assert!(result.is_err());
}

// --- equality ---

#[test]
fn clone_and_eq() {
    let a = RatioArray::new_with_vec(4, vec![ri(1), r(1, 2), ri(-1)]).unwrap();
    let b = a.clone();
    assert_eq!(a, b);
}

#[test]
fn eq_ignores_k_and_block_structure() {
    let a = RatioArray::new_with_vec(2, vec![ri(1), r(1, 2), ri(-1)]).unwrap();
    let b = RatioArray::new_with_vec(8, vec![ri(1), r(1, 2), ri(-1)]).unwrap();
    assert_eq!(a, b);
}

#[test]
fn serde_round_trip_eq() {
    use serde_json;
    let orig = RatioArray::new_with_vec(4, vec![ri(0), r(1, 2), ri(-1)]).unwrap();
    let json = serde_json::to_string(&orig).unwrap();
    let loaded: RatioArray = serde_json::from_str(&json).unwrap(); // k=64
    assert_eq!(orig, loaded);
}

// --- datasize ---

#[test]
fn datasize_nonzero() {
    let arr = RatioArray::new_with_vec(4, vec![ri(1), r(1, 2)]).unwrap();
    assert!(arr.datasize() > 0);
}

#[test]
fn datasize_grows_with_elements() {
    let a = RatioArray::new_with_vec(64, vec![ri(1)]).unwrap();
    let b = RatioArray::new_with_vec(64, vec![ri(1); 100]).unwrap();
    assert!(b.datasize() > a.datasize());
}
