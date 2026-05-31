# intarray

Memory-efficient packed integer arrays for Rust.

- **`IntArray`** — N-bit unsigned integers packed into `Vec<u64>`. Store 1–64 bits per element with no per-element overhead.
- **`RadixArray`** — Signed integers in an arbitrary range `[A, B]`, packed using mixed-radix encoding. Maximizes elements per 64-bit word for any value range.

## When to use

| | `IntArray` | `RadixArray` |
|---|---|---|
| Value type | `u64` (unsigned) | `i64` (signed) |
| Range constraint | 1–64 bits | any `[A, B]` |
| Best for | bit-width known at design time | range known at runtime |

Both types: memory is the bottleneck, values are bounded, no per-element allocation.

## Installation

```toml
[dependencies]
intarray = "0.3"
```

## Error type

Both types share a single error enum:

```rust
pub enum ArrayError {
    OutOfBounds,   // index ≥ array length
    TooLarge,      // value exceeds upper bound
    TooSmall,      // value is below lower bound
    Empty,         // pop() on empty array
    InvalidRange,  // RadixArray: K < 2 or K > u64::MAX
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

## Serialization (serde)

Both types serialize as flat sequences.

```rust
use serde_json;

// IntArray → [u64, ...]
let v = IntArray::new_with_vec(4, vec![1u64, 2, 3]).unwrap();
let json = serde_json::to_string(&v).unwrap();   // "[1,2,3]"
let v2: IntArray = serde_json::from_str(&json).unwrap();
// Bit width re-inferred from max value on deserialize.

// RadixArray → [i64, ...]
let r = RadixArray::new_with_vec(-5, 5, vec![-1i64, 0, 2]).unwrap();
let json = serde_json::to_string(&r).unwrap();   // "[-1,0,2]"
let r2: RadixArray = serde_json::from_str(&json).unwrap();
// Range [A, B] re-inferred from min/max on deserialize.
```

## MSRV

Rust 1.87 (uses `usize::is_multiple_of`, stabilized in 1.87).
