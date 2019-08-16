use super::*;
use serde_derive::Serialize;
use serde_json;
use serde_yaml;
use std::time::Instant;
use test_env_log::test;

#[test]
fn construct() {
    let v = IntArray::new(2, 10);
    info!("{}", v);
    info!("max={}", v.max_value());
    info!("idx[0]={}", v.get(0).unwrap());
}

#[test]
fn construct_withvec() {
    let v = IntArray::new_with_vec(2, vec![0, 1, 2, 3]);
    assert_eq!(v.get(0).unwrap(), 0);
    assert_eq!(v.get(1).unwrap(), 1);
    assert_eq!(v.get(2).unwrap(), 2);
    assert_eq!(v.get(3).unwrap(), 3);
}

#[test]
fn construct_withiter() {
    let v = IntArray::new_with_iter(2, vec![0, 1, 2, 3].iter().map(|a| *a));
    assert_eq!(v.get(0).unwrap(), 0);
    assert_eq!(v.get(1).unwrap(), 1);
    assert_eq!(v.get(2).unwrap(), 2);
    assert_eq!(v.get(3).unwrap(), 3);
    assert_eq!(v.length, 4);
}

#[test]
fn construct_withiter2() {
    let v = IntArray::new_with_iter(20, 0..(1 << 16));
    for i in 0..v.length {
        assert_eq!(v.get(i).unwrap(), i as u64);
    }
    assert_eq!(v.length, 1 << 16);
}

#[test]
#[should_panic]
fn construct_withvec_bad() {
    let _ = IntArray::new_with_vec(2, vec![0, 1, 2, 3, 4, 5]);
}

#[test]
fn modify() {
    let mut v = IntArray::new(2, 10);
    v.set(3, 1).unwrap();
    v.set(6, 2).unwrap();
    v.set(9, 3).unwrap();
    assert_eq!(v.to_string(), "2[32]=0,0,0,1,0,0,2,0,0,3".to_string())
}

#[test]
#[should_panic]
fn large() {
    let mut v = IntArray::new(2, 3);
    v.set(0, 5).unwrap();
}

#[test]
#[should_panic]
fn out_of_bounds() {
    let mut v = IntArray::new(2, 3);
    v.set(3, 1).unwrap();
}

#[test]
fn extend() {
    let mut v = IntArray::new(2, 3);
    v.resize(10);
    v.set(3, 1).unwrap();
    assert_eq!(v.to_string(), "2[32]=0,0,0,1,0,0,0,0,0,0".to_string())
}

#[test]
fn extend_fast() {
    let mut v1 = IntArray::new(2, 3);
    let v2 = IntArray::new(10, 20);
    // slow path
    v1.extend_array(v2);
    assert_eq!(v1.length, 23);
    assert_eq!(v1.sum().unwrap(), 0);

    let mut v3 = IntArray::new(2, 128);
    let v4 = IntArray::new(2, 10);
    v3.extend_array(v4);
    assert_eq!(v3.length, 138);
    assert_eq!(v3.sum().unwrap(), 0);
}

#[test]
fn iterate1() {
    let mut v = IntArray::new(2, 3);
    v.set(1, 1).unwrap();
    assert_eq!(v.iter().count(), 3);
    v.resize(10);
    assert_eq!(v.iter().count(), 10)
}

#[test]
fn iterate2() {
    let mut v = IntArray::new(2, 3);
    v.set(0, 0).unwrap();
    v.set(1, 1).unwrap();
    v.set(2, 2).unwrap();
    for (i, j) in v.iter().enumerate() {
        assert_eq!(i, j as usize)
    }
}

#[test]
fn iterate3() {
    let mut v = IntArray::new(2, 3);
    v.set(0, 0).unwrap();
    v.set(1, 1).unwrap();
    v.set(2, 2).unwrap();
    let res = v.iter().collect::<Vec<u64>>();
    assert_eq!(vec![0, 1, 2], res);
}

#[test]
fn addsub() {
    let mut v = IntArray::new(3, 3);
    for i in 0..v.length {
        v.incr(i).unwrap();
    }
    assert_eq!(v.to_string(), "3[21]=1,1,1".to_string());

    for i in 0..v.length {
        v.add(i, i as u64).unwrap();
    }
    assert_eq!(v.to_string(), "3[21]=1,2,3".to_string());

    for i in 0..v.length {
        v.decr(i).unwrap();
    }
    assert_eq!(v.to_string(), "3[21]=0,1,2".to_string());
}

#[test]
fn sum_1() {
    let mut v = IntArray::new(7, 120);
    for i in 0..v.length {
        v.incr(i).unwrap();
    }
    assert_eq!(v.sum0().unwrap(), v.length as u64);
    assert_eq!(v.sum().unwrap(), v.length as u64);

    for i in 0..v.length {
        v.add(i, i as u64).unwrap();
    }
    assert_eq!(v.sum(), v.sum0());

    for i in 0..v.length {
        v.decr(i).unwrap();
    }
    assert_eq!(v.sum(), v.sum0());
}

#[test]
fn clone1() {
    let mut v1 = IntArray::new(3, 3);
    v1.set(1, 1).unwrap();
    let mut v2 = v1.clone();
    v2.set(2, 2).unwrap();
    assert_eq!(v1.get(2).unwrap(), 0);
    assert_eq!(v2.get(2).unwrap(), 2);
    assert_eq!(v2.get(1).unwrap(), 1);
    assert_eq!(v1.bits, v2.bits);
    assert_eq!(v1.length, v2.length);
}

#[derive(Serialize)]
struct Ex {
    a: u32,
    v: IntArray,
}

#[test]
fn json() {
    let mut x = Ex {
        a: 10,
        v: IntArray::new(3, 3),
    };
    x.v.incr(1).unwrap();
    let serialized = serde_json::to_string(&x).unwrap();
    assert_eq!(serialized, r#"{"a":10,"v":[0,1,0]}"#.to_string())
}

#[test]
fn yaml() {
    let mut x = Ex {
        a: 10,
        v: IntArray::new(3, 3),
    };
    x.v.incr(1).unwrap();
    let serialized = serde_yaml::to_string(&x).unwrap();
    assert_eq!(
        serialized,
        r#"---
a: 10
v:
  - 0
  - 1
  - 0"#
            .to_string()
    )
}

#[test]
fn u64test() {
    println!("min={}, max={}", u64::min_value(), u64::max_value());
    let mv: f64 = u64::max_value() as f64;
    for i in 2..64 {
        println!("log({}, max) = {}", i, mv.log(i as f64));
    }
}

#[test]
fn sumx() {
    let mut rng = rand::thread_rng();
    let bits: usize = rng.gen_range(1, 20);
    let entries: usize = rng.gen_range(1 * 1024 * 1024, (64 / bits as usize) * 1024 * 1024);
    let mut v = IntArray::new(bits, entries);
    let maxv = v.max_value();
    info!(
        "{} bits, {} entries, {}*8, max={}",
        bits,
        entries,
        v.data.len(),
        maxv
    );
    v.fill_random();
    info!("initialize done.");
    let start = Instant::now();
    let sum1 = v.sum().unwrap();
    let end = start.elapsed();
    info!(
        "sum1={}, elapsed={}.{:03}",
        sum1,
        end.as_secs(),
        end.subsec_nanos() / 1_000_000
    );
    let start2 = Instant::now();
    let sum2 = v.sum0().unwrap();
    let end2 = start2.elapsed();
    info!(
        "sum2={}, elapsed={}.{:03}",
        sum2,
        end2.as_secs(),
        end2.subsec_nanos() / 1_000_000
    );
    assert_eq!(sum1, sum2);
}

#[test]
fn limit_incdec() {
    let mut v = IntArray::new(2, 10);
    v.incr_limit(1);
    assert_eq!(v.get(1).unwrap(), 1);
    v.decr_limit(1);
    assert_eq!(v.get(1).unwrap(), 0);
    v.incr_limit(1);
    assert_eq!(v.get(1).unwrap(), 1);
    v.incr_limit(1);
    assert_eq!(v.get(1).unwrap(), 2);
    v.incr_limit(1);
    assert_eq!(v.get(1).unwrap(), 3);
    v.incr_limit(1);
    assert_eq!(v.get(1).unwrap(), 3);
    v.decr_limit(1);
    assert_eq!(v.get(1).unwrap(), 3);
}

#[test]
fn test_bits() {
    assert_eq!(0.ffs(), 0);
    assert_eq!(1.ffs(), 1);
    assert_eq!(5.ffs(), 3);
    assert_eq!(1023.ffs(), 10);
    assert_eq!(1024.ffs(), 11);
    assert_eq!(65535.ffs(), 16);
    assert_eq!(8192.ffs(), 14);
    assert_eq!(0x1234_5678_9abc.ffs(), 45);
    assert_eq!(0xf000_0000_0000_0000.ffs(), 64);
}

#[test]
fn pushpop() {
    let mut v = IntArray::new_with_vec(2, vec![0, 1, 2]);
    v.push(3);
    assert_eq!(v.pop().unwrap(), 3);
    assert_eq!(v.pop().unwrap(), 2);
    assert_eq!(v.pop().unwrap(), 1);
    assert_eq!(v.pop().unwrap(), 0);
}

#[test]
fn cat() {
    let mut v1 = IntArray::new_with_vec(2, vec![0, 1, 2]);
    let v2 = IntArray::new_with_vec(3, vec![0, 1, 2]);
    v1.concat(v2);
    assert_eq!(v1.length, 6);
    assert_eq!(v1.get(3).unwrap(), 0);
}

#[test]
fn shape() {
    let v1 = IntArray::new_with_vec(10, vec![0, 1, 2, 0, 1, 2]);
    let v2 = v1.shape(3);
    // assert_eq!(v2.length, v1.length);
    assert_eq!(v2.bits, 3);
    assert_eq!(v2.len(), v1.len());
    let v3 = v1.shape_auto();
    assert_eq!(v3.bits, 2);
    assert_eq!(v3.len(), v1.len());
}

#[test]
fn assign_int() {
    let mut v1 = IntArray::new_with_vec(10, vec![0, 1, 2, 0, 1, 2, 3]);
    v1 += 10;
    assert_eq!(v1.get(0).unwrap(), 10);
    assert_eq!(v1.get(2).unwrap(), 12);
    v1 -= 5;
    assert_eq!(v1.get(1).unwrap(), 6);
    assert_eq!(v1.get(3).unwrap(), 5);
    v1.resize(10);
    println!("v1={}", v1);
    assert_eq!(v1.sum().unwrap(), 44);
    v1 *= 3;
    assert_eq!(v1.sum().unwrap(), 44 * 3);
}

#[test]
fn assign_array() {
    let mut v1 = IntArray::new_with_vec(10, vec![0, 1, 2, 0, 1, 2]);
    let v2 = IntArray::new_with_vec(10, vec![2, 1, 0, 2, 1, 0]);
    v1 += v2;
    assert_eq!(v1.max(), 2);
    assert_eq!(v1.min(), 2);
    v1 -= IntArray::new_with_vec(10, vec![2, 1, 0, 2, 1, 0]);
    assert_eq!(v1.get(0).unwrap(), 0);
    assert_eq!(v1.get(2).unwrap(), 2);
    v1 += IntArray::new_with_vec(3, vec![2, 1, 0, 2, 1, 0]);
    assert_eq!(v1.max(), 2);
    assert_eq!(v1.min(), 2);
    v1 -= IntArray::new_with_vec(5, vec![2, 1, 0, 2, 1, 0, 2]);
    assert_eq!(v1.get(0).unwrap(), 0);
    assert_eq!(v1.get(2).unwrap(), 2);
}

#[test]
fn maxmin() {
    let v1 = IntArray::new_with_vec(10, vec![0, 1, 2, 0, 1, 2]);
    assert_eq!(v1.max(), 2);
    assert_eq!(v1.min(), 0);
}

#[test]
fn test_subarray() {
    let v1 = IntArray::new_with_vec(10, vec![0, 1, 2, 0, 1, 2]);
    // slow path
    let v2 = v1.subarray(2, 3);
    assert_eq!(v2.get(0).unwrap(), 2);
    assert_eq!(v2.get(1).unwrap(), 0);
    assert_eq!(v2.sum().unwrap(), 3);
    assert_eq!(v2.length, 3);

    // fast path
    let v3 = IntArray::new_with_iter(10, 0..64);
    let v4 = v3.subarray(6, 18);
    assert_eq!(v4.get(0).unwrap(), 6);
    assert_eq!(v4.get(1).unwrap(), 7);
    assert_eq!(v4.sum().unwrap(), 261);
    assert_eq!(v4.length, 18);
}

#[test]
fn add64() {
    let v1 = 0x1234_5678_90ab_cdef;
    let v2 = 2;
    println!("v={}", v1.addval_bits(v2, 8).unwrap());
}

#[test]
fn zerowidth() {
    assert_eq!(0.subval_bits(0, 0), None);
}

#[test]
fn add2_64_overflow() {
    assert_eq!(1.addval_bits(1, 1), None);
    assert_eq!(3.addval_bits(1, 2), None);
}

#[test]
fn sub2_64_underflow() {
    assert_eq!(4.subval_bits(1, 1), None);
    assert_eq!(0xff00.subval_bits(1, 2), None);
}

#[test]
fn getmask() {
    if ELEMENT_BITS == 128 {
        assert_eq!(
            get_mask(1) as u128,
            0x5555_5555_5555_5555_5555_5555_5555_5555
        );
        assert_eq!(
            get_mask(2) as u128,
            0x3333_3333_3333_3333_3333_3333_3333_3333
        );
        assert_eq!(
            get_mask(24) as u128,
            0x0000_ffffff_000000_ffffff_000000_ffffff
        );
        assert_eq!(
            get_mask(32) as u128,
            0x0000_0000_ffff_ffff_0000_0000_ffff_ffff
        );
        assert_eq!(
            get_mask(48) as u128,
            0xffff_ffff_0000_0000_0000_ffff_ffff_ffff
        );
        assert_eq!(get_mask(64) as u128, 0xffff_ffff_ffff_ffff);
    }
    if ELEMENT_BITS == 64 {
        assert_eq!(get_mask(1) as u64, 0x5555_5555_5555_5555);
        assert_eq!(get_mask(2) as u64, 0x3333_3333_3333_3333);
        assert_eq!(
            get_mask(24) as u64,
            0b1111111111111111_000000000000000000000000_111111111111111111111111
        );
        assert_eq!(get_mask(32) as u64, 0xffff_ffff);
        assert_eq!(get_mask(48) as u64, 0xffff_ffff_ffff);
        assert_eq!(get_mask(63) as u64, 0x7fff_ffff_ffff_ffff);
        assert_eq!(get_mask(64) as u64, 0xffff_ffff_ffff_ffff);
    }
    if ELEMENT_BITS == 32 {
        assert_eq!(get_mask(1) as u32, 0x5555_5555);
        assert_eq!(get_mask(2) as u32, 0x3333_3333);
        assert_eq!(get_mask(24) as u32, 0b111111111111111111111111);
        assert_eq!(get_mask(32) as u32, 0xffff_ffff);
    }
}

#[test]
fn element_sum() {
    let a = 0x1234;
    assert_eq!(a.sum_bits(1).unwrap(), 5);
    assert_eq!(a.sum_bits(2).unwrap(), 7);
    assert_eq!(a.sum_bits(4).unwrap(), 10);
    assert_eq!(a.sum_bits(8).unwrap(), 70);
}

#[test]
fn element_add() {
    let a = 0x1234;
    let b = 0x4321;
    assert_eq!(a.add_bits(b, 4).unwrap(), 0x5555);
    assert_eq!(a.add_bits(b, 8).unwrap(), 0x5555);

    assert_eq!(a.addval_bits(1, 4).unwrap() & 0xffff, 0x2345);
    assert_eq!(a.addval_bits(1, 8).unwrap() & 0xffff, 0x1335);

    assert_eq!(a.addval_bits(2, 4).unwrap() & 0xffff, 0x3456);
    assert_eq!(a.addval_bits(2, 8).unwrap() & 0xffff, 0x1436);
}

/*
    #[bench]
    fn getbench1(b: &mut Bencher) {
        let mut v = IntArray::new(1, 1024 * 1024);
        for i in 0..v.length {
            if i % 2 == 0 {
                continue;
            }
            v.set(i, 1);
        }
        b.iter(|| {
            for i in 0..v.length / 3 {
                v.get(i * 3);
            }
        })
    }
*/
