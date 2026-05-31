# intarray

Memory-efficient packed arrays for Rust. Five types covering unsigned integers, signed integers, floating-point, arbitrary-precision integers, and exact rationals — all with no per-element heap allocation.

- **`IntArray`** — N-bit unsigned integers packed into `Vec<u64>`. Store 1–64 bits per element with no per-element overhead.
- **`RadixArray`** — Signed integers in an arbitrary range `[A, B]`, packed using mixed-radix encoding. Maximizes elements per 64-bit word for any value range.
- **`FloatArray`** — Floating-point values stored at a custom bit precision (exponent + mantissa bits). Supports FLOAT16, BFLOAT16, FLOAT32, FLOAT64, and any user-defined format.
- **`VarIntArray`** — Arbitrary-precision integers (`num_bigint::BigInt`) compressed with Elias gamma + zigzag encoding. Block-based storage for random access.
- **`RatioArray`** — Exact rational numbers (`num_rational::Ratio<BigInt>`) with no rounding error. Numerator and denominator are each Elias-gamma encoded.

## When to use

| | `IntArray` | `RadixArray` | `FloatArray` | `VarIntArray` | `RatioArray` |
|---|---|---|---|---|---|
| Value type | `u64` | `i64` | `f64` | `BigInt` | `Ratio<BigInt>` |
| Precision | fixed bits | fixed range | custom exp+man | arbitrary | arbitrary |
| Bounded values | yes | yes | no | no | no |
| Exact arithmetic | — | — | no | yes | yes |
| Best for | bit-width known at design time | range known at runtime | compact floats | large unbounded ints | rounding-free rationals |

All types: memory is the bottleneck, no per-element heap allocation.

## Installation

```toml
[dependencies]
intarray = "0.4"
```

## Error type

All types share a single error enum:

```rust
pub enum ArrayError {
    OutOfBounds,   // index ≥ array length
    TooLarge,      // value exceeds upper bound
    TooSmall,      // value is below lower bound
    Empty,         // pop() on empty array
    InvalidRange,  // invalid construction parameters
}
```

`ArrayError` implements `std::error::Error` and `Display`.

---

## IntArray

Stores `u64` values using exactly `bits` bits per element. Suitable when all values fit in a known fixed bit width.

### Quick start

```rust
use intarray::{IntArray, ArrayError};

// 7-bit unsigned integers, 1000 elements (pre-allocated, zero-filled)
let mut v = IntArray::new(7, 1000);

v.set(0, 100).unwrap();       // v[0] = 100
v.set(1, 127).unwrap();       // max for 7 bits
assert_eq!(v.get(0).unwrap(), 100);

// Out-of-range returns Err, never panics
assert_eq!(v.set(0, 128), Err(ArrayError::TooLarge));

v.push(42).unwrap();          // append
```

### Construction

```rust
// Pre-allocated, zero-filled
let v = IntArray::new(4, 100);

// From a Vec
let v = IntArray::new_with_vec(4, vec![1u64, 2, 3, 4]).unwrap();

// From an iterator
let v = IntArray::new_with_iter(4, 0..16u64).unwrap();

// Infer minimum bit width from data
let v = IntArray::new_with_vec(8, vec![0u64, 1, 2, 3]).unwrap();
let compact = v.shape_auto();  // bits = 2 (max value = 3, needs 2 bits)
```

### Element access

```rust
let mut v = IntArray::new(4, 10);  // max value = 15

v.get(5).unwrap();                 // → 0
v.set(5, 15).unwrap();             // ok
v.set(5, 16).unwrap_err();         // TooLarge
v.get(10).unwrap_err();            // OutOfBounds

v.push(7).unwrap();                // append, returns new index
v.pop().unwrap();                  // remove last, returns value

v.incr(5).unwrap();                // v[5] += 1
v.decr(5).unwrap();                // v[5] -= 1
v.add(5, 3).unwrap();              // v[5] += 3
v.sub(5, 3).unwrap();              // v[5] -= 3
```

`incr_limit` / `decr_limit` clamp at the boundary and return `None` at the edge:

```rust
v.incr_limit(5);  // → Some(old_value) or None if already at max
v.decr_limit(5);  // → Some(old_value) or None if already at 0
```

### Bulk operations

`push`, `extend`, and `extend_array` are all atomic: on error, the array is left unchanged.

```rust
let mut v = IntArray::new(4, 0);

v.extend(vec![1u64, 2, 3]).unwrap();

let other = IntArray::new_with_vec(4, vec![4u64, 5, 6]).unwrap();
v.extend_array(&other).unwrap();   // fast path when bits and alignment match
```

### Arithmetic operators

Element-wise `+=`, `-=`, `*=` on a scalar `u64` or another `IntArray`:

```rust
let mut a = IntArray::new_with_vec(8, vec![10u64, 20, 30]).unwrap();
a += 5u64;    // [15, 25, 35]

let b = IntArray::new_with_vec(8, vec![1u64, 2, 3]).unwrap();
a += &b;      // [16, 27, 38]
```

### Iteration and statistics

```rust
let v = IntArray::new_with_vec(8, vec![3u64, 1, 4, 1, 5, 9]).unwrap();

for x in v.iter() { println!("{}", x); }

v.sum().unwrap();      // → 23u128
v.min().unwrap();      // → 1
v.max().unwrap();      // → 9
v.average().unwrap();  // → 3.833...
```

### Shape / reshape

```rust
let v = IntArray::new_with_vec(16, vec![0u64, 1, 1000]).unwrap();

let v10 = v.shape(10);       // reshape to 10 bits
let compact = v.shape_auto(); // minimum bits for max value (10 bits for 1000)

let sub = v.subarray(1, 2);  // elements [1..3) — zero-copy when aligned
```

### Memory layout

```
v.len();       // number of elements
v.capacity();  // allocated capacity in elements (rounded to word boundary)
v.datasize();  // total size in bytes
```

Each `u64` word holds `64 / bits` elements. For example, 4-bit integers pack 16 per word; a 100-element array uses 7 words (56 bytes of data).

---

## RadixArray

Stores `i64` values in a fixed range `[A, B]` using mixed-radix (base-K) encoding, where K = B − A + 1. Each `u64` word holds `floor(64·ln2 / ln(K))` elements — maximally dense for any value range.

### Quick start

```rust
use intarray::{RadixArray, ArrayError};

// Values in [0, 9] (10 possible values), 5 elements
let mut v = RadixArray::new(0, 9, 5).unwrap();

v.set(0, 7).unwrap();
assert_eq!(v.get(0).unwrap(), 7);

// Out-of-range returns Err
assert_eq!(v.set(0, 10), Err(ArrayError::TooLarge));
assert_eq!(v.set(0, -1), Err(ArrayError::TooSmall));

v.push(3).unwrap();  // → index 5
```

### Construction

```rust
// Pre-allocated, values initialized to A
let v = RadixArray::new(-10, 10, 100).unwrap();

// From a Vec — atomic (Err if any value out of range)
let v = RadixArray::new_with_vec(-5, 5, vec![1, -2, 3]).unwrap();

// From an iterator
let v = RadixArray::new_with_iter(0, 255, 0..=255i64).unwrap();
```

### Element access

```rust
let mut v = RadixArray::new(-100, 100, 10).unwrap();

v.get(0).unwrap();            // → -100 (initialized to A)
v.set(0, 42).unwrap();
v.set(0, 101).unwrap_err();   // TooLarge
v.set(0, -101).unwrap_err();  // TooSmall

v.push(-50).unwrap();         // append, returns new index
v.pop().unwrap();             // remove last, returns value
```

### Bulk operations

```rust
let mut v = RadixArray::new(0, 9, 0).unwrap();

v.extend(vec![1i64, 2, 3]).unwrap();

// Extend from another RadixArray
let other = RadixArray::new_with_vec(0, 9, vec![4i64, 5, 6]).unwrap();
v.extend_array(&other).unwrap();  // fast path when ranges match and alignment holds
```

### Iteration and statistics

```rust
let v = RadixArray::new_with_vec(-5, 5, vec![-3i64, 0, 2, -1]).unwrap();

for x in v.iter() { println!("{}", x); }

v.sum().unwrap();      // → -2i128
v.min().unwrap();      // → -3
v.max().unwrap();      // → 2
v.average().unwrap();  // → -0.5
```

### Range info

```rust
v.base();          // K = B − A + 1
v.range();         // (A, B) as (i64, i64)
v.len();
v.capacity();      // allocated capacity in elements
v.datasize();      // total size in bytes
```

### Packing efficiency

| Range size K | elements per word |
|---|---|
| 2 | 64 |
| 10 | 19 |
| 256 | 8 |
| 65536 | 4 |
| 2³² | 2 |

---

## FloatArray

Stores `f64` values at a reduced precision defined by `(exp_bits, man_bits)`. Each value is re-encoded into the custom format on write and decoded back to `f64` on read. Four predefined formats are provided as constants.

| Constant | exp | man | total bits | notes |
|---|---|---|---|---|
| `FLOAT64` | 11 | 52 | 64 | standard `f64` |
| `FLOAT32` | 8 | 23 | 32 | standard `f32` |
| `FLOAT16` | 5 | 10 | 16 | IEEE 754 half |
| `BFLOAT16` | 8 | 7 | 16 | Google Brain float |

### Quick start

```rust
use intarray::{FloatArray, FLOAT32};

// 32-bit floats, block size 64
let mut v = FloatArray::new(FLOAT32.0, FLOAT32.1, 0).unwrap();

v.push(3.14).unwrap();
v.push(-1.0).unwrap();
assert!((v.get(0).unwrap() - 3.14f64).abs() < 1e-6);

v.sum().unwrap();      // → ~2.14
v.min().unwrap();      // → ~-1.0
v.average().unwrap();  // → ~1.07
```

### Construction

```rust
use intarray::{FloatArray, FLOAT16, FLOAT32, FLOAT64, BFLOAT16};

// Using a predefined format
let v = FloatArray::new(FLOAT32.0, FLOAT32.1, 100).unwrap();

// Custom format: 6-bit exponent, 9-bit mantissa (16 bits total)
let v = FloatArray::new(6, 9, 0).unwrap();

// From a Vec
let v = FloatArray::new_with_vec(FLOAT32.0, FLOAT32.1, vec![1.0, 2.0, 3.0]).unwrap();

// From an iterator
let v = FloatArray::new_with_iter(FLOAT16.0, FLOAT16.1, [0.5, 1.0, 1.5]).unwrap();
```

### Element access

```rust
let mut v = FloatArray::new(FLOAT32.0, FLOAT32.1, 4).unwrap();

v.set(0, 1.5).unwrap();
v.get(0).unwrap();         // → ~1.5
v.push(42.0).unwrap();     // append, returns new index
v.pop().unwrap();          // remove last, returns value
```

### Iteration and statistics

```rust
let v = FloatArray::new_with_vec(FLOAT32.0, FLOAT32.1, vec![1.0, 2.0, 3.0]).unwrap();

for x in v.iter() { println!("{}", x); }

v.sum().unwrap();      // → ~6.0
v.min().unwrap();      // → ~1.0
v.max().unwrap();      // → ~3.0
v.average().unwrap();  // → ~2.0
```

### Metadata

```rust
v.len();
v.bits_per_unit();    // = 1 + exp_bits + man_bits
v.datasize();         // total size in bytes
```

---

## VarIntArray

Stores arbitrary-precision signed integers (`num_bigint::BigInt`) using Elias gamma + zigzag encoding. Elements are grouped into blocks of `k` for O(1) amortized `push` and O(k) `get`/`set`.

Memory usage scales with the actual values stored: small values (near zero) use fewer bits.

### Quick start

```rust
use intarray::VarIntArray;
use num_bigint::BigInt;

let mut v = VarIntArray::new(64).unwrap();  // block size = 64

v.push(BigInt::from(0)).unwrap();
v.push(BigInt::from(-1)).unwrap();
v.push(BigInt::from(i64::MAX)).unwrap();
v.push("123456789012345678901234567890".parse::<BigInt>().unwrap()).unwrap();

assert_eq!(v.get(1).unwrap(), BigInt::from(-1));
assert_eq!(v.len(), 4);
```

### Construction

```rust
use num_bigint::BigInt;

let v = VarIntArray::new(64).unwrap();

let v = VarIntArray::new_with_vec(32, vec![
    BigInt::from(1),
    BigInt::from(-2),
    BigInt::from(1000),
]).unwrap();

let v = VarIntArray::new_with_iter(64, (0i64..100).map(BigInt::from)).unwrap();
```

### Element access

```rust
let mut v = VarIntArray::new_with_vec(4, vec![
    BigInt::from(1), BigInt::from(2), BigInt::from(3),
]).unwrap();

v.get(0).unwrap();                        // → BigInt::from(1)
v.set(1, BigInt::from(-99)).unwrap();     // decode block, replace, re-encode
v.push(BigInt::from(42)).unwrap();        // append
v.pop().unwrap();                         // remove last
```

### Iteration and statistics

```rust
let v = VarIntArray::new_with_vec(64, vec![
    BigInt::from(10), BigInt::from(-3), BigInt::from(7),
]).unwrap();

for x in v.iter() { println!("{}", x); }

v.sum().unwrap();      // → BigInt::from(14)
v.min().unwrap();      // → BigInt::from(-3)
v.max().unwrap();      // → BigInt::from(10)
v.average().unwrap();  // → ~4.666... (f64, None if empty)
```

### Metadata

```rust
v.len();
v.block_size();    // k
v.block_count();   // number of blocks
v.datasize();      // total size in bytes
```

---

## RatioArray

Stores exact rational numbers (`num_rational::Ratio<BigInt>`) with no rounding error. The numerator is zigzag + Elias gamma encoded; the denominator (always ≥ 1) uses standard Elias gamma. Integers (denominator = 1) cost only 1 extra bit over `VarIntArray`.

`average()` returns an exact `Ratio<BigInt>`, not a float.

### Quick start

```rust
use intarray::RatioArray;
use num_bigint::BigInt;
use num_rational::Ratio;

let mut v = RatioArray::new(64).unwrap();

v.push(Ratio::from_integer(BigInt::from(0))).unwrap();
v.push(Ratio::new(BigInt::from(1), BigInt::from(2))).unwrap();   // 1/2
v.push(Ratio::new(BigInt::from(-1), BigInt::from(3))).unwrap();  // -1/3
v.push(Ratio::new(BigInt::from(22), BigInt::from(7))).unwrap();  // 22/7

assert_eq!(v.get(1).unwrap(), Ratio::new(BigInt::from(1), BigInt::from(2)));
assert_eq!(v.len(), 4);
```

### Construction

```rust
use num_bigint::BigInt;
use num_rational::Ratio;

let v = RatioArray::new(64).unwrap();

let v = RatioArray::new_with_vec(32, vec![
    Ratio::from_integer(BigInt::from(1)),
    Ratio::new(BigInt::from(1), BigInt::from(2)),
]).unwrap();

let v = RatioArray::new_with_iter(64, [
    Ratio::from_integer(BigInt::from(0)),
    Ratio::new(BigInt::from(3), BigInt::from(4)),
]).unwrap();
```

### Element access

```rust
let mut v = RatioArray::new_with_vec(4, vec![
    Ratio::from_integer(BigInt::from(1)),
    Ratio::from_integer(BigInt::from(2)),
]).unwrap();

v.get(0).unwrap();    // → 1/1
v.set(0, Ratio::new(BigInt::from(3), BigInt::from(7))).unwrap();
v.push(Ratio::new(BigInt::from(1), BigInt::from(6))).unwrap();
v.pop().unwrap();
```

### Iteration and statistics

```rust
use num_bigint::BigInt;
use num_rational::Ratio;

let v = RatioArray::new_with_vec(4, vec![
    Ratio::new(BigInt::from(1), BigInt::from(2)),   // 1/2
    Ratio::new(BigInt::from(1), BigInt::from(3)),   // 1/3
    Ratio::new(BigInt::from(1), BigInt::from(6)),   // 1/6
]).unwrap();

for x in v.iter() { println!("{}", x); }

v.sum().unwrap();      // → Ratio = 1  (exact: 1/2 + 1/3 + 1/6 = 1)
v.min().unwrap();      // → 1/6
v.max().unwrap();      // → 1/2
v.average().unwrap();  // → Ratio = 1/3  (exact, no float rounding)
```

### Metadata

```rust
v.len();
v.block_size();    // k
v.block_count();   // number of blocks
v.datasize();      // total size in bytes
```

---

## Serialization (serde)

All types serialize as flat sequences. Internal parameters (bit width, range, block size) are not preserved; they are re-inferred or reset to defaults on deserialization.

```rust
use serde_json;

// IntArray → JSON array of u64
let v = IntArray::new_with_vec(4, vec![1u64, 2, 3]).unwrap();
let json = serde_json::to_string(&v).unwrap();   // "[1,2,3]"
let v2: IntArray = serde_json::from_str(&json).unwrap();
// Bit width re-inferred from max value on deserialize.

// RadixArray → JSON array of i64
let r = RadixArray::new_with_vec(-5, 5, vec![-1i64, 0, 2]).unwrap();
let json = serde_json::to_string(&r).unwrap();   // "[-1,0,2]"
let r2: RadixArray = serde_json::from_str(&json).unwrap();
// Range [A, B] re-inferred from min/max on deserialize.

// FloatArray → JSON array of f64
let f = FloatArray::new_with_vec(FLOAT32.0, FLOAT32.1, vec![1.0, 2.0]).unwrap();
let json = serde_json::to_string(&f).unwrap();   // "[1.0,2.0]"
let f2: FloatArray = serde_json::from_str(&json).unwrap();
// Format defaults to FLOAT64 on deserialize.

// VarIntArray → JSON array of decimal strings (arbitrary precision)
let vi = VarIntArray::new_with_vec(4, vec![BigInt::from(-1), BigInt::from(2)]).unwrap();
let json = serde_json::to_string(&vi).unwrap();  // "["-1","2"]"
let vi2: VarIntArray = serde_json::from_str(&json).unwrap();
// Block size k defaults to 64 on deserialize.

// RatioArray → JSON array of "p/q" strings (integers as "p")
let ra = RatioArray::new_with_vec(4, vec![
    Ratio::new(BigInt::from(1), BigInt::from(2)),
    Ratio::from_integer(BigInt::from(-3)),
]).unwrap();
let json = serde_json::to_string(&ra).unwrap();  // "["1/2","-3"]"
let ra2: RatioArray = serde_json::from_str(&json).unwrap();
// Block size k defaults to 64 on deserialize.
```

`PartialEq` for `VarIntArray` and `RatioArray` compares elements only (ignoring block size `k`), so a serde round-trip always compares equal.

## MSRV

Rust 1.87 (uses `usize::is_multiple_of`, stabilized in 1.87).
