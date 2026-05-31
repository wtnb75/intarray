use super::*;
use crate::error::ArrayError;
use test_log::test;

// --- epu calculation ---

#[test]
fn epu_base2() {
    assert_eq!(calc_epu(2), 64);
}

#[test]
fn epu_base3() {
    assert_eq!(calc_epu(3), 40);
}

#[test]
fn epu_base4() {
    assert_eq!(calc_epu(4), 32);
}

#[test]
fn epu_base10() {
    assert_eq!(calc_epu(10), 19);
}

#[test]
fn epu_base_max() {
    assert_eq!(calc_epu(u64::MAX), 1);
}

// --- Construction ---

#[test]
fn new_basic() {
    let v = RadixArray::new(-1, 1, 5).unwrap();
    assert_eq!(v.len(), 5);
    assert_eq!(v.base(), 3);
    assert_eq!(v.range(), (-1, 1));
    for i in 0..5 {
        assert_eq!(v.get(i).unwrap(), -1);
    }
}

#[test]
fn new_unsigned_range() {
    let v = RadixArray::new(0, 9, 10).unwrap();
    assert_eq!(v.base(), 10);
    assert_eq!(v.len(), 10);
}

#[test]
fn new_invalid_range() {
    assert_eq!(RadixArray::new(1, 0, 0), Err(ArrayError::InvalidRange));
    assert_eq!(RadixArray::new(5, 5, 0), Err(ArrayError::InvalidRange));
    assert_eq!(
        RadixArray::new(i64::MIN, i64::MAX, 0),
        Err(ArrayError::InvalidRange)
    );
}

#[test]
fn new_with_vec_basic() {
    let v = RadixArray::new_with_vec(-1, 1, vec![-1, 0, 1, -1, 0]).unwrap();
    assert_eq!(v.len(), 5);
    assert_eq!(v.get(0).unwrap(), -1);
    assert_eq!(v.get(1).unwrap(), 0);
    assert_eq!(v.get(2).unwrap(), 1);
    assert_eq!(v.get(3).unwrap(), -1);
    assert_eq!(v.get(4).unwrap(), 0);
}

#[test]
fn new_with_vec_out_of_range() {
    assert_eq!(
        RadixArray::new_with_vec(-1, 1, vec![-1, 0, 2]),
        Err(ArrayError::TooLarge)
    );
    assert_eq!(
        RadixArray::new_with_vec(-1, 1, vec![-1, -2]),
        Err(ArrayError::TooSmall)
    );
}

#[test]
fn new_with_iter_basic() {
    let v = RadixArray::new_with_iter(0, 2, 0..3i64).unwrap();
    assert_eq!(v.len(), 3);
    assert_eq!(v.get(0).unwrap(), 0);
    assert_eq!(v.get(1).unwrap(), 1);
    assert_eq!(v.get(2).unwrap(), 2);
}

// --- get / set ---

#[test]
fn get_set_basic() {
    let mut v = RadixArray::new(0, 2, 5).unwrap();
    for i in 0..5usize {
        v.set(i, (i % 3) as i64).unwrap();
    }
    for i in 0..5usize {
        assert_eq!(v.get(i).unwrap(), (i % 3) as i64);
    }
}

#[test]
fn get_set_negative_range() {
    let mut v = RadixArray::new(-10, 10, 3).unwrap();
    v.set(0, -10).unwrap();
    v.set(1, 0).unwrap();
    v.set(2, 10).unwrap();
    assert_eq!(v.get(0).unwrap(), -10);
    assert_eq!(v.get(1).unwrap(), 0);
    assert_eq!(v.get(2).unwrap(), 10);
}

#[test]
fn epu_word_boundary() {
    let epu = calc_epu(3); // 40
    let mut v = RadixArray::new(0, 2, epu + 2).unwrap();
    v.set(epu - 1, 2).unwrap();
    v.set(epu, 1).unwrap();
    v.set(epu + 1, 0).unwrap();
    assert_eq!(v.get(epu - 1).unwrap(), 2);
    assert_eq!(v.get(epu).unwrap(), 1);
    assert_eq!(v.get(epu + 1).unwrap(), 0);
}

#[test]
fn overwrite_element() {
    let mut v = RadixArray::new(0, 2, 5).unwrap();
    v.set(2, 2).unwrap();
    assert_eq!(v.get(2).unwrap(), 2);
    v.set(2, 0).unwrap();
    assert_eq!(v.get(2).unwrap(), 0);
    v.set(2, 1).unwrap();
    assert_eq!(v.get(2).unwrap(), 1);
    assert_eq!(v.get(1).unwrap(), 0);
    assert_eq!(v.get(3).unwrap(), 0);
}

#[test]
fn error_types() {
    let mut v = RadixArray::new(-1, 1, 3).unwrap();
    assert_eq!(v.get(10), Err(ArrayError::OutOfBounds));
    assert_eq!(v.set(10, 0), Err(ArrayError::OutOfBounds));
    assert_eq!(v.set(0, 2), Err(ArrayError::TooLarge));
    assert_eq!(v.set(0, -2), Err(ArrayError::TooSmall));
    let mut empty = RadixArray::new(-1, 1, 0).unwrap();
    assert_eq!(empty.pop(), Err(ArrayError::Empty));
}

// --- push / pop ---

#[test]
fn push_pop_basic() {
    let mut v = RadixArray::new(0, 9, 0).unwrap();
    assert_eq!(v.push(3).unwrap(), 0);
    assert_eq!(v.push(7).unwrap(), 1);
    assert_eq!(v.push(5).unwrap(), 2);
    assert_eq!(v.len(), 3);
    assert_eq!(v.pop().unwrap(), 5);
    assert_eq!(v.pop().unwrap(), 7);
    assert_eq!(v.pop().unwrap(), 3);
    assert_eq!(v.pop(), Err(ArrayError::Empty));
}

#[test]
fn push_too_large() {
    let mut v = RadixArray::new(-1, 1, 0).unwrap();
    assert_eq!(v.push(2), Err(ArrayError::TooLarge));
    assert_eq!(v.len(), 0);
}

#[test]
fn push_too_small() {
    let mut v = RadixArray::new(-1, 1, 0).unwrap();
    assert_eq!(v.push(-2), Err(ArrayError::TooSmall));
    assert_eq!(v.len(), 0);
}

// --- extend ---

#[test]
fn extend_basic() {
    let mut v = RadixArray::new(-1, 1, 0).unwrap();
    v.extend(vec![-1i64, 0, 1, -1, 0]).unwrap();
    assert_eq!(v.len(), 5);
    assert_eq!(v.get(2).unwrap(), 1);
}

#[test]
fn extend_atomic_rollback() {
    let mut v = RadixArray::new(-1, 1, 0).unwrap();
    v.extend(vec![-1i64, 0]).unwrap();
    assert_eq!(v.len(), 2);
    assert_eq!(v.extend(vec![-1i64, 0, 2]), Err(ArrayError::TooLarge));
    assert_eq!(v.len(), 2);
    assert_eq!(v.get(0).unwrap(), -1);
    assert_eq!(v.get(1).unwrap(), 0);
}

// --- extend_array ---

#[test]
fn extend_array_slow_path_different_base() {
    let mut dst = RadixArray::new(0, 2, 0).unwrap(); // K=3
    let src = RadixArray::new_with_vec(0, 1, vec![0i64, 1, 0, 1]).unwrap(); // K=2
    dst.extend_array(&src).unwrap();
    assert_eq!(dst.len(), 4);
    assert_eq!(dst.get(1).unwrap(), 1);
}

#[test]
fn extend_array_slow_path_different_offset() {
    let epu = calc_epu(3);
    let mut dst = RadixArray::new(0, 2, epu).unwrap();
    for i in 0..epu {
        dst.set(i, (i % 3) as i64).unwrap();
    }
    let src = RadixArray::new_with_vec(-1, 1, vec![0i64, 1, 0]).unwrap();
    dst.extend_array(&src).unwrap();
    assert_eq!(dst.len(), epu + 3);
    assert_eq!(dst.get(epu).unwrap(), 0);
    assert_eq!(dst.get(epu + 1).unwrap(), 1);
    assert_eq!(dst.get(epu + 2).unwrap(), 0);
    assert_ne!(src.offset, dst.offset);
}

#[test]
fn extend_array_fast_path() {
    let epu = calc_epu(3);
    let mut dst = RadixArray::new(0, 2, epu).unwrap();
    for i in 0..epu {
        dst.set(i, (i % 3) as i64).unwrap();
    }
    let src = RadixArray::new_with_vec(0, 2, vec![1i64, 2, 0]).unwrap();
    dst.extend_array(&src).unwrap();
    assert_eq!(dst.len(), epu + 3);
    assert_eq!(dst.get(epu).unwrap(), 1);
    assert_eq!(dst.get(epu + 1).unwrap(), 2);
    assert_eq!(dst.get(epu + 2).unwrap(), 0);
    for i in 0..epu {
        assert_eq!(dst.get(i).unwrap(), (i % 3) as i64);
    }
}

#[test]
fn extend_array_too_large() {
    let mut dst = RadixArray::new(0, 1, 0).unwrap();
    let src = RadixArray::new_with_vec(0, 2, vec![0i64, 1, 2]).unwrap();
    assert_eq!(dst.extend_array(&src), Err(ArrayError::TooLarge));
    assert_eq!(dst.len(), 0);
}

// --- large array across many word boundaries ---

#[test]
fn many_elements_correctness() {
    let n = 200usize;
    let mut v = RadixArray::new(0, 2, n).unwrap();
    for i in 0..n {
        v.set(i, (i % 3) as i64).unwrap();
    }
    for i in 0..n {
        assert_eq!(v.get(i).unwrap(), (i % 3) as i64, "mismatch at index {}", i);
    }
}

// --- statistics ---

#[test]
fn statistics_basic() {
    let v = RadixArray::new_with_vec(-2, 2, vec![-2i64, -1, 0, 1, 2]).unwrap();
    assert_eq!(v.sum().unwrap(), 0i128);
    assert_eq!(v.min().unwrap(), -2);
    assert_eq!(v.max().unwrap(), 2);
    assert_eq!(v.average().unwrap(), 0.0);
}

#[test]
fn statistics_empty() {
    let v = RadixArray::new(-1, 1, 0).unwrap();
    assert_eq!(v.sum(), None);
    assert_eq!(v.min(), None);
    assert_eq!(v.max(), None);
    assert_eq!(v.average(), None);
}

#[test]
fn statistics_single() {
    let v = RadixArray::new_with_vec(0, 9, vec![7i64]).unwrap();
    assert_eq!(v.sum().unwrap(), 7i128);
    assert_eq!(v.min().unwrap(), 7);
    assert_eq!(v.max().unwrap(), 7);
    assert_eq!(v.average().unwrap(), 7.0);
}

// --- iterator ---

#[test]
fn iter_collect() {
    let v = RadixArray::new_with_vec(-1, 1, vec![-1i64, 0, 1]).unwrap();
    let got: Vec<i64> = v.iter().collect();
    assert_eq!(got, vec![-1, 0, 1]);
}

#[test]
fn iter_size_hint() {
    let v = RadixArray::new_with_vec(0, 2, vec![0i64, 1, 2]).unwrap();
    let mut iter = v.iter();
    assert_eq!(iter.size_hint(), (3, Some(3)));
    iter.next();
    assert_eq!(iter.size_hint(), (2, Some(2)));
    iter.next();
    iter.next();
    assert_eq!(iter.size_hint(), (0, Some(0)));
}

#[test]
fn iter_len() {
    let v = RadixArray::new_with_vec(0, 2, vec![0i64, 1, 2]).unwrap();
    assert_eq!(v.iter().len(), 3);
}

// --- clone ---

#[test]
fn clone_independence() {
    let v1 = RadixArray::new_with_vec(-1, 1, vec![-1i64, 0, 1]).unwrap();
    let mut v2 = v1.clone();
    v2.set(0, 1).unwrap();
    assert_eq!(v1.get(0).unwrap(), -1);
    assert_eq!(v2.get(0).unwrap(), 1);
}

// --- Display ---

#[test]
fn display_basic() {
    let v = RadixArray::new_with_vec(-1, 1, vec![-1i64, 0, 1]).unwrap();
    assert_eq!(v.to_string(), "[-1,1][3]=-1,0,1");
}

#[test]
fn display_empty() {
    let v = RadixArray::new(-1, 1, 0).unwrap();
    assert_eq!(v.to_string(), "[-1,1][0]=");
}

// --- Serde ---

#[test]
fn serde_roundtrip_json() {
    let v = RadixArray::new_with_vec(-1, 1, vec![-1i64, 0, 1, -1, 0]).unwrap();
    let json = serde_json::to_string(&v).unwrap();
    assert_eq!(json, "[-1,0,1,-1,0]");
    let v2: RadixArray = serde_json::from_str(&json).unwrap();
    assert_eq!(v2.len(), v.len());
    for i in 0..v.len() {
        assert_eq!(v2.get(i).unwrap(), v.get(i).unwrap());
    }
    assert_eq!(v2.range(), (-1, 1));
}

#[test]
fn serde_roundtrip_yaml() {
    let v = RadixArray::new_with_vec(0, 9, vec![3i64, 1, 4, 1, 5]).unwrap();
    let yaml = serde_yaml::to_string(&v).unwrap();
    let v2: RadixArray = serde_yaml::from_str(&yaml).unwrap();
    assert_eq!(v2.len(), v.len());
    for i in 0..v.len() {
        assert_eq!(v2.get(i).unwrap(), v.get(i).unwrap());
    }
}

#[test]
fn serde_empty() {
    let v = RadixArray::new(-1, 1, 0).unwrap();
    let json = serde_json::to_string(&v).unwrap();
    assert_eq!(json, "[]");
    let v2: RadixArray = serde_json::from_str(&json).unwrap();
    assert_eq!(v2.len(), 0);
    assert_eq!(v2.range(), (0, 1));
}

#[test]
fn serde_all_same_value() {
    let v = RadixArray::new_with_vec(3, 5, vec![3i64, 3, 3]).unwrap();
    let json = serde_json::to_string(&v).unwrap();
    let v2: RadixArray = serde_json::from_str(&json).unwrap();
    assert_eq!(v2.len(), 3);
    assert_eq!(v2.range(), (3, 4));
    for i in 0..3 {
        assert_eq!(v2.get(i).unwrap(), 3);
    }
}

#[test]
fn serde_range_loss() {
    let v = RadixArray::new_with_vec(-5, 5, vec![0i64, 1, 0]).unwrap();
    let json = serde_json::to_string(&v).unwrap();
    let v2: RadixArray = serde_json::from_str(&json).unwrap();
    assert_eq!(v2.range(), (0, 1));
}

// --- K=2 correctness ---

#[test]
fn base2_correctness() {
    let n = 128usize;
    let mut v = RadixArray::new(0, 1, n).unwrap();
    assert_eq!(v.base(), 2);
    assert_eq!(v.epu, 64);
    for i in 0..n {
        v.set(i, (i % 2) as i64).unwrap();
    }
    for i in 0..n {
        assert_eq!(v.get(i).unwrap(), (i % 2) as i64);
    }
}

// --- capacity / datasize ---

#[test]
fn capacity_and_datasize() {
    let v = RadixArray::new(0, 2, 5).unwrap(); // K=3, epu=40 → 1 word
    assert_eq!(v.capacity(), 40);
    assert!(v.datasize() > 0);
}
