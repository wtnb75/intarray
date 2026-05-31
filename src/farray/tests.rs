use super::*;
use crate::error::ArrayError;
use test_log::test;

// --- validation ---

#[test]
fn invalid_exp_bits_too_small() {
    assert_eq!(FloatArray::new(1, 10, 0), Err(ArrayError::InvalidRange));
}

#[test]
fn invalid_exp_bits_too_large() {
    assert_eq!(FloatArray::new(12, 10, 0), Err(ArrayError::InvalidRange));
}

#[test]
fn invalid_man_bits_zero() {
    assert_eq!(FloatArray::new(5, 0, 0), Err(ArrayError::InvalidRange));
}

#[test]
fn invalid_man_bits_too_large() {
    assert_eq!(FloatArray::new(5, 53, 0), Err(ArrayError::InvalidRange));
}

#[test]
fn valid_standard_formats() {
    let (e, m) = FLOAT16;
    assert!(FloatArray::new(e, m, 0).is_ok());
    let (e, m) = BFLOAT16;
    assert!(FloatArray::new(e, m, 0).is_ok());
    let (e, m) = FLOAT32;
    assert!(FloatArray::new(e, m, 0).is_ok());
    let (e, m) = FLOAT64;
    assert!(FloatArray::new(e, m, 0).is_ok());
}

// --- construction ---

#[test]
fn new_initializes_to_zero() {
    let arr = FloatArray::new(5, 10, 4).unwrap();
    assert_eq!(arr.len(), 4);
    assert_eq!(arr.exp_bits(), 5);
    assert_eq!(arr.man_bits(), 10);
    for i in 0..4 {
        assert_eq!(arr.get(i).unwrap(), 0.0f64);
    }
}

#[test]
fn new_with_vec() {
    let arr = FloatArray::new_with_vec(8, 23, vec![1.0, 2.0, 3.0]).unwrap();
    assert_eq!(arr.len(), 3);
}

#[test]
fn new_with_iter() {
    let arr = FloatArray::new_with_iter(8, 23, [1.0f64, 2.0, 3.0]).unwrap();
    assert_eq!(arr.len(), 3);
    assert_eq!(arr.get(1).unwrap(), 2.0);
}

// --- get / set ---

#[test]
fn out_of_bounds() {
    let mut arr = FloatArray::new(5, 10, 2).unwrap();
    assert_eq!(arr.get(2), Err(ArrayError::OutOfBounds));
    assert_eq!(arr.set(2, 1.0), Err(ArrayError::OutOfBounds));
}

#[test]
fn float64_round_trip_exact() {
    let (e, m) = FLOAT64;
    let mut arr = FloatArray::new(e, m, 0).unwrap();
    let vals = [std::f64::consts::PI, std::f64::consts::E, 1.0, -1.0, 0.0];
    for v in vals {
        arr.push(v).unwrap();
    }
    for (i, v) in vals.iter().enumerate() {
        assert_eq!(arr.get(i).unwrap(), *v);
    }
}

#[test]
fn float16_normal_values() {
    let (e, m) = FLOAT16;
    let mut arr = FloatArray::new(e, m, 0).unwrap();
    arr.push(1.0).unwrap();
    arr.push(-1.0).unwrap();
    arr.push(0.5).unwrap();
    // FLOAT16 has ~3 decimal digits of precision
    assert!((arr.get(0).unwrap() - 1.0).abs() < 1e-3);
    assert!((arr.get(1).unwrap() + 1.0).abs() < 1e-3);
    assert!((arr.get(2).unwrap() - 0.5).abs() < 1e-4);
}

// --- special values ---

#[test]
fn positive_infinity() {
    let mut arr = FloatArray::new(5, 10, 0).unwrap();
    arr.push(f64::INFINITY).unwrap();
    assert_eq!(arr.get(0).unwrap(), f64::INFINITY);
}

#[test]
fn negative_infinity() {
    let mut arr = FloatArray::new(5, 10, 0).unwrap();
    arr.push(f64::NEG_INFINITY).unwrap();
    assert_eq!(arr.get(0).unwrap(), f64::NEG_INFINITY);
}

#[test]
fn nan_round_trip() {
    let mut arr = FloatArray::new(5, 10, 0).unwrap();
    arr.push(f64::NAN).unwrap();
    assert!(arr.get(0).unwrap().is_nan());
}

#[test]
fn positive_zero() {
    let mut arr = FloatArray::new(5, 10, 0).unwrap();
    arr.push(0.0f64).unwrap();
    let v = arr.get(0).unwrap();
    assert_eq!(v, 0.0);
    assert!(v.is_sign_positive());
}

#[test]
fn negative_zero() {
    let mut arr = FloatArray::new(5, 10, 0).unwrap();
    arr.push(-0.0f64).unwrap();
    let v = arr.get(0).unwrap();
    assert_eq!(v, 0.0);
    assert!(v.is_sign_negative());
}

#[test]
fn denormal_flushes_to_zero() {
    let denormal = f64::from_bits(1u64); // smallest positive denormal
    let mut arr = FloatArray::new(5, 10, 0).unwrap();
    arr.push(denormal).unwrap();
    assert_eq!(arr.get(0).unwrap(), 0.0);
}

#[test]
fn overflow_becomes_inf() {
    // 1e300 overflows FLOAT16 exponent range
    let mut arr = FloatArray::new(5, 10, 0).unwrap();
    arr.push(1e300f64).unwrap();
    assert_eq!(arr.get(0).unwrap(), f64::INFINITY);
}

#[test]
fn underflow_becomes_zero() {
    let mut arr = FloatArray::new(5, 10, 0).unwrap();
    arr.push(1e-300f64).unwrap();
    assert_eq!(arr.get(0).unwrap(), 0.0);
}

// --- push / pop ---

#[test]
fn push_pop() {
    let mut arr = FloatArray::new(5, 10, 0).unwrap();
    assert_eq!(arr.push(1.0).unwrap(), 0);
    assert_eq!(arr.push(2.0).unwrap(), 1);
    assert_eq!(arr.len(), 2);
    assert!((arr.pop().unwrap() - 2.0).abs() < 1e-3);
    assert_eq!(arr.len(), 1);
}

#[test]
fn pop_empty() {
    let mut arr = FloatArray::new(5, 10, 0).unwrap();
    assert_eq!(arr.pop(), Err(ArrayError::Empty));
}

// --- extend_array ---

#[test]
fn extend_array_fast_path() {
    // bpu=4 for FLOAT16; 4 elements is word-aligned
    let (e, m) = FLOAT16;
    let a = FloatArray::new_with_vec(e, m, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let mut b = FloatArray::new(e, m, 0).unwrap();
    b.extend_array(&a).unwrap();
    assert_eq!(b.len(), 4);
    assert!((b.get(0).unwrap() - 1.0).abs() < 1e-3);
    assert!((b.get(3).unwrap() - 4.0).abs() < 1e-3);
}

#[test]
fn extend_array_slow_path_different_format() {
    let a = FloatArray::new_with_vec(5, 10, vec![1.0, 2.0]).unwrap();
    let mut b = FloatArray::new(8, 23, 0).unwrap();
    b.extend_array(&a).unwrap();
    assert_eq!(b.len(), 2);
    assert!((b.get(0).unwrap() - 1.0).abs() < 1e-3);
}

#[test]
fn extend_array_slow_path_same_format_unaligned() {
    // bpu=4 for FLOAT16; 3 elements is not word-aligned → slow path
    let (e, m) = FLOAT16;
    let a = FloatArray::new_with_vec(e, m, vec![1.0, 2.0]).unwrap();
    let mut b = FloatArray::new_with_vec(e, m, vec![0.5, 0.25, 0.125]).unwrap(); // len=3, not multiple of bpu=4
    b.extend_array(&a).unwrap();
    assert_eq!(b.len(), 5);
    assert!((b.get(3).unwrap() - 1.0).abs() < 1e-3);
    assert!((b.get(4).unwrap() - 2.0).abs() < 1e-3);
}

// --- statistics ---

#[test]
fn empty_stats() {
    let arr = FloatArray::new(5, 10, 0).unwrap();
    assert_eq!(arr.sum(), None);
    assert_eq!(arr.min(), None);
    assert_eq!(arr.max(), None);
    assert_eq!(arr.average(), None);
}

#[test]
fn stats_basic() {
    let (e, m) = FLOAT64;
    let arr = FloatArray::new_with_vec(e, m, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    assert_eq!(arr.sum().unwrap(), 10.0);
    assert_eq!(arr.min().unwrap(), 1.0);
    assert_eq!(arr.max().unwrap(), 4.0);
    assert_eq!(arr.average().unwrap(), 2.5);
}

#[test]
fn min_max_ignores_nan() {
    let (e, m) = FLOAT64;
    let arr = FloatArray::new_with_vec(e, m, vec![1.0, f64::NAN, 3.0]).unwrap();
    assert_eq!(arr.min().unwrap(), 1.0);
    assert_eq!(arr.max().unwrap(), 3.0);
}

#[test]
fn sum_propagates_nan() {
    let (e, m) = FLOAT64;
    let arr = FloatArray::new_with_vec(e, m, vec![1.0, f64::NAN]).unwrap();
    assert!(arr.sum().unwrap().is_nan());
}

// --- display ---

#[test]
fn display_format() {
    let arr = FloatArray::new_with_vec(5, 10, vec![]).unwrap();
    assert!(arr.to_string().starts_with("[e5m10][0]="));
}

// --- serde ---

#[test]
fn serde_round_trip() {
    let (e, m) = FLOAT64;
    let arr = FloatArray::new_with_vec(e, m, vec![1.0, 2.0, 3.0]).unwrap();
    let json = serde_json::to_string(&arr).unwrap();
    let arr2: FloatArray = serde_json::from_str(&json).unwrap();
    assert_eq!(arr2.len(), 3);
    assert_eq!(arr2.get(0).unwrap(), 1.0);
    assert_eq!(arr2.exp_bits(), 11);
    assert_eq!(arr2.man_bits(), 52);
}
