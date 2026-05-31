use super::*;
use crate::error::ArrayError;
use num_bigint::BigInt;
use test_log::test;

fn bi(n: i64) -> BigInt {
    BigInt::from(n)
}

// --- validation ---

#[test]
fn invalid_k_zero() {
    assert_eq!(VarIntArray::new(0), Err(ArrayError::InvalidRange));
}

#[test]
fn valid_k_one() {
    assert!(VarIntArray::new(1).is_ok());
}

#[test]
fn valid_k_64() {
    assert!(VarIntArray::new(64).is_ok());
}

// --- construction ---

#[test]
fn new_empty() {
    let arr = VarIntArray::new(4).unwrap();
    assert_eq!(arr.len(), 0);
    assert!(arr.is_empty());
    assert_eq!(arr.block_count(), 0);
    assert_eq!(arr.block_size(), 4);
}

#[test]
fn new_with_vec_basic() {
    let arr = VarIntArray::new_with_vec(4, vec![bi(0), bi(-1), bi(1)]).unwrap();
    assert_eq!(arr.len(), 3);
    assert_eq!(arr.get(0).unwrap(), bi(0));
    assert_eq!(arr.get(1).unwrap(), bi(-1));
    assert_eq!(arr.get(2).unwrap(), bi(1));
}

#[test]
fn new_with_iter_basic() {
    let arr = VarIntArray::new_with_iter(4, [bi(0), bi(-1), bi(1)]).unwrap();
    assert_eq!(arr.len(), 3);
    assert_eq!(arr.get(2).unwrap(), bi(1));
}

// --- encoding correctness: specific values ---

#[test]
fn encode_decode_zero() {
    let mut arr = VarIntArray::new(64).unwrap();
    arr.push(bi(0)).unwrap();
    assert_eq!(arr.get(0).unwrap(), bi(0));
}

#[test]
fn encode_decode_positive_negative() {
    let vals = [0i64, -1, 1, -2, 2, 127, -128, 1000];
    let mut arr = VarIntArray::new(64).unwrap();
    for &v in &vals {
        arr.push(bi(v)).unwrap();
    }
    for (i, &v) in vals.iter().enumerate() {
        assert_eq!(arr.get(i).unwrap(), bi(v), "index {i}");
    }
}

#[test]
fn encode_decode_large_bigint() {
    let big: BigInt = "123456789012345678901234567890".parse().unwrap();
    let neg_big: BigInt = "-987654321098765432109876543210".parse().unwrap();
    let mut arr = VarIntArray::new(64).unwrap();
    arr.push(big.clone()).unwrap();
    arr.push(neg_big.clone()).unwrap();
    assert_eq!(arr.get(0).unwrap(), big);
    assert_eq!(arr.get(1).unwrap(), neg_big);
}

// --- k=1 edge cases ---

#[test]
fn k1_single_zero() {
    let mut arr = VarIntArray::new(1).unwrap();
    arr.push(bi(0)).unwrap();
    assert_eq!(arr.block_count(), 1);
    let v: Vec<_> = arr.iter().collect();
    assert_eq!(v, vec![bi(0)]);
}

#[test]
fn k1_multiple_pushes_each_own_block() {
    let arr = VarIntArray::new_with_vec(1, vec![bi(1), bi(2), bi(3)]).unwrap();
    assert_eq!(arr.block_count(), 3);
    let v: Vec<_> = arr.iter().collect();
    assert_eq!(v, vec![bi(1), bi(2), bi(3)]);
}

// --- block layout verification ---

#[test]
fn spec_block_example() {
    // From the spec: [0, -1, 1, -2] in one block → 15 bits → data=[0xAB, 0x9E]
    let mut arr = VarIntArray::new(4).unwrap();
    arr.push(bi(0)).unwrap();
    arr.push(bi(-1)).unwrap();
    arr.push(bi(1)).unwrap();
    arr.push(bi(-2)).unwrap();

    let block = &arr.blocks[0];
    assert_eq!(block.bit_len, 15);
    assert_eq!(block.data[0], 0b10101011); // 0xAB
    assert_eq!(block.data[1], 0b10011110); // 0x9E
}

// --- metadata ---

#[test]
fn datasize_nonzero() {
    let empty = VarIntArray::new(4).unwrap();
    assert!(empty.datasize() > 0); // struct overhead always present
    let arr = VarIntArray::new_with_vec(4, vec![bi(1), bi(2)]).unwrap();
    assert!(arr.datasize() > empty.datasize());
}

// --- get / set ---

#[test]
fn out_of_bounds_get() {
    let arr = VarIntArray::new(4).unwrap();
    assert_eq!(arr.get(0), Err(ArrayError::OutOfBounds));
}

#[test]
fn out_of_bounds_usize_max() {
    let arr = VarIntArray::new_with_vec(4, vec![bi(1)]).unwrap();
    assert_eq!(arr.get(usize::MAX), Err(ArrayError::OutOfBounds));
    let mut arr2 = VarIntArray::new_with_vec(4, vec![bi(1)]).unwrap();
    assert_eq!(arr2.set(usize::MAX, bi(99)), Err(ArrayError::OutOfBounds));
}

#[test]
fn out_of_bounds_set() {
    let mut arr = VarIntArray::new(4).unwrap();
    assert_eq!(arr.set(0, bi(1)), Err(ArrayError::OutOfBounds));
}

#[test]
fn set_replaces_value() {
    let mut arr = VarIntArray::new_with_vec(4, vec![bi(1), bi(2), bi(3)]).unwrap();
    arr.set(1, bi(99)).unwrap();
    assert_eq!(arr.get(0).unwrap(), bi(1));
    assert_eq!(arr.get(1).unwrap(), bi(99));
    assert_eq!(arr.get(2).unwrap(), bi(3));
    assert_eq!(arr.len(), 3);
}

#[test]
fn set_across_blocks() {
    // k=2: first block has [0,1], second has [2,3]
    let mut arr = VarIntArray::new_with_vec(2, vec![bi(0), bi(1), bi(2), bi(3)]).unwrap();
    assert_eq!(arr.block_count(), 2);
    arr.set(2, bi(99)).unwrap();
    assert_eq!(arr.get(2).unwrap(), bi(99));
    assert_eq!(arr.get(3).unwrap(), bi(3));
}

// --- push / pop ---

#[test]
fn push_returns_index() {
    let mut arr = VarIntArray::new(4).unwrap();
    assert_eq!(arr.push(bi(10)).unwrap(), 0);
    assert_eq!(arr.push(bi(20)).unwrap(), 1);
    assert_eq!(arr.push(bi(30)).unwrap(), 2);
}

#[test]
fn pop_basic() {
    let mut arr = VarIntArray::new_with_vec(4, vec![bi(1), bi(2), bi(3)]).unwrap();
    assert_eq!(arr.pop().unwrap(), bi(3));
    assert_eq!(arr.len(), 2);
    assert_eq!(arr.pop().unwrap(), bi(2));
    assert_eq!(arr.len(), 1);
}

#[test]
fn pop_empty() {
    let mut arr = VarIntArray::new(4).unwrap();
    assert_eq!(arr.pop(), Err(ArrayError::Empty));
}

#[test]
fn push_fills_block_then_new_block() {
    // k=2: elements 0,1 fill block 0; element 2 starts block 1
    let mut arr = VarIntArray::new(2).unwrap();
    arr.push(bi(0)).unwrap();
    arr.push(bi(1)).unwrap();
    assert_eq!(arr.block_count(), 1);
    arr.push(bi(2)).unwrap();
    assert_eq!(arr.block_count(), 2);
    assert_eq!(arr.get(2).unwrap(), bi(2));
}

#[test]
fn pop_removes_empty_block() {
    // k=2: one full block [0,1]
    let mut arr = VarIntArray::new_with_vec(2, vec![bi(0), bi(1)]).unwrap();
    assert_eq!(arr.block_count(), 1);
    arr.pop().unwrap(); // block becomes [0]
    assert_eq!(arr.block_count(), 1);
    arr.pop().unwrap(); // block becomes empty → removed
    assert_eq!(arr.block_count(), 0);
    assert!(arr.is_empty());
}

#[test]
fn pop_across_block_boundary() {
    // k=2: blocks = [[0,1], [2]]
    let mut arr = VarIntArray::new_with_vec(2, vec![bi(0), bi(1), bi(2)]).unwrap();
    assert_eq!(arr.pop().unwrap(), bi(2));
    assert_eq!(arr.block_count(), 1); // second block removed
    assert_eq!(arr.pop().unwrap(), bi(1));
    assert_eq!(arr.pop().unwrap(), bi(0));
    assert!(arr.is_empty());
}

// --- iter ---

#[test]
fn iter_empty() {
    let arr = VarIntArray::new(4).unwrap();
    let v: Vec<_> = arr.iter().collect();
    assert!(v.is_empty());
}

#[test]
fn iter_basic() {
    let vals = vec![bi(0), bi(-1), bi(1), bi(-2), bi(1000)];
    let arr = VarIntArray::new_with_vec(4, vals.clone()).unwrap();
    let collected: Vec<_> = arr.iter().collect();
    assert_eq!(collected, vals);
}

#[test]
fn iter_across_blocks() {
    // k=2: 5 elements → 3 blocks
    let vals: Vec<_> = (0i64..5).map(bi).collect();
    let arr = VarIntArray::new_with_vec(2, vals.clone()).unwrap();
    assert_eq!(arr.block_count(), 3);
    let collected: Vec<_> = arr.iter().collect();
    assert_eq!(collected, vals);
}

#[test]
fn iter_all_full_blocks() {
    // k=2: 4 elements → exactly 2 full blocks; iterator must not over-advance
    let vals: Vec<_> = (0i64..4).map(bi).collect();
    let arr = VarIntArray::new_with_vec(2, vals.clone()).unwrap();
    assert_eq!(arr.block_count(), 2);
    let collected: Vec<_> = arr.iter().collect();
    assert_eq!(collected, vals);
}

#[test]
fn iter_size_hint() {
    let arr = VarIntArray::new_with_vec(4, vec![bi(1), bi(2), bi(3)]).unwrap();
    let mut it = arr.iter();
    assert_eq!(it.size_hint(), (3, Some(3)));
    it.next();
    assert_eq!(it.size_hint(), (2, Some(2)));
}

#[test]
fn iter_exact_size() {
    let arr = VarIntArray::new_with_vec(4, vec![bi(1), bi(2), bi(3)]).unwrap();
    let it = arr.iter();
    assert_eq!(it.len(), 3);
}

// --- statistics ---

#[test]
fn empty_stats() {
    let arr = VarIntArray::new(4).unwrap();
    assert_eq!(arr.sum(), None);
    assert_eq!(arr.min(), None);
    assert_eq!(arr.max(), None);
    assert_eq!(arr.average(), None);
}

#[test]
fn stats_basic() {
    let arr =
        VarIntArray::new_with_vec(4, vec![bi(-2), bi(1), bi(3), bi(-1)]).unwrap();
    assert_eq!(arr.sum().unwrap(), bi(1));
    assert_eq!(arr.min().unwrap(), bi(-2));
    assert_eq!(arr.max().unwrap(), bi(3));
    assert_eq!(arr.average().unwrap(), 0.25f64);
}

#[test]
fn stats_single_element() {
    let arr = VarIntArray::new_with_vec(4, vec![bi(42)]).unwrap();
    assert_eq!(arr.sum().unwrap(), bi(42));
    assert_eq!(arr.min().unwrap(), bi(42));
    assert_eq!(arr.max().unwrap(), bi(42));
    assert_eq!(arr.average().unwrap(), 42.0f64);
}

#[test]
fn average_large_value_is_finite_or_nan() {
    // 10^400 exceeds f64 max (~1.8×10^308); average should be ±infinity or NaN, not panic
    let huge: BigInt = std::iter::repeat('9').take(400).collect::<String>().parse().unwrap();
    let arr = VarIntArray::new_with_vec(64, vec![huge]).unwrap();
    let avg = arr.average().unwrap();
    assert!(avg.is_nan() || avg.is_infinite());
}

// --- clone / eq ---

#[test]
fn clone_and_eq() {
    let a = VarIntArray::new_with_vec(4, vec![bi(1), bi(-2), bi(3)]).unwrap();
    let b = a.clone();
    assert_eq!(a, b);
}

#[test]
fn eq_ignores_k_and_block_structure() {
    // same elements but different k → must be equal
    let a = VarIntArray::new_with_vec(2, vec![bi(1), bi(-2), bi(3)]).unwrap();
    let b = VarIntArray::new_with_vec(4, vec![bi(1), bi(-2), bi(3)]).unwrap();
    assert_eq!(a, b);
}

#[test]
fn serde_round_trip_eq() {
    // serde resets k to 64; the original had k=4 — they must still be equal
    let orig = VarIntArray::new_with_vec(4, vec![bi(0), bi(-1), bi(1)]).unwrap();
    let json = serde_json::to_string(&orig).unwrap();
    let loaded: VarIntArray = serde_json::from_str(&json).unwrap();
    assert_eq!(orig, loaded);
}

// --- extend / extend_array ---

#[test]
fn extend_basic() {
    let mut arr = VarIntArray::new(4).unwrap();
    arr.extend(vec![bi(1), bi(2), bi(3)]).unwrap();
    assert_eq!(arr.len(), 3);
    assert_eq!(arr.get(2).unwrap(), bi(3));
}

#[test]
fn extend_array_different_k() {
    let a = VarIntArray::new_with_vec(2, vec![bi(10), bi(20), bi(30)]).unwrap();
    let mut b = VarIntArray::new(4).unwrap();
    b.extend_array(&a).unwrap();
    assert_eq!(b.len(), 3);
    assert_eq!(b.block_size(), 4);
    assert_eq!(b.get(1).unwrap(), bi(20));
}

#[test]
fn extend_array_empty_source() {
    let mut a = VarIntArray::new_with_vec(4, vec![bi(1)]).unwrap();
    let empty = VarIntArray::new(2).unwrap();
    a.extend_array(&empty).unwrap();
    assert_eq!(a.len(), 1);
    assert_eq!(a.get(0).unwrap(), bi(1));
}

// --- display ---

#[test]
fn display_format() {
    let arr = VarIntArray::new_with_vec(64, vec![bi(0), bi(-1), bi(1)]).unwrap();
    let s = arr.to_string();
    assert!(s.starts_with("[k=64][3]="));
    assert!(s.contains("0,-1,1"));
}

#[test]
fn display_large_value() {
    let big: BigInt = "123456789012345678901234567890".parse().unwrap();
    let arr = VarIntArray::new_with_vec(64, vec![big]).unwrap();
    assert!(arr.to_string().contains("123456789012345678901234567890"));
}

// --- serde ---

#[test]
fn serde_json_format_is_flat_string_array() {
    let arr = VarIntArray::new_with_vec(4, vec![bi(-1), bi(2)]).unwrap();
    let json = serde_json::to_string(&arr).unwrap();
    assert_eq!(json, r#"["-1","2"]"#);
}

#[test]
fn serde_invalid_string_returns_error() {
    let result: Result<VarIntArray, _> = serde_json::from_str(r#"["not_a_number"]"#);
    assert!(result.is_err());
}

#[test]
fn serde_round_trip() {
    let arr = VarIntArray::new_with_vec(4, vec![bi(0), bi(-1), bi(1), bi(1000)]).unwrap();
    let json = serde_json::to_string(&arr).unwrap();
    let arr2: VarIntArray = serde_json::from_str(&json).unwrap();
    assert_eq!(arr2.len(), 4);
    assert_eq!(arr2.get(0).unwrap(), bi(0));
    assert_eq!(arr2.get(1).unwrap(), bi(-1));
    assert_eq!(arr2.get(3).unwrap(), bi(1000));
    assert_eq!(arr2.block_size(), 64); // default k after deserialize
}

#[test]
fn serde_large_bigint() {
    let big: BigInt = "99999999999999999999999999999999999999999".parse().unwrap();
    let arr = VarIntArray::new_with_vec(64, vec![big.clone()]).unwrap();
    let json = serde_json::to_string(&arr).unwrap();
    let arr2: VarIntArray = serde_json::from_str(&json).unwrap();
    assert_eq!(arr2.get(0).unwrap(), big);
}
