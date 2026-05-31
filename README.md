# intarray

Memory-efficient packed integer array for Rust. Stores integers of 1–64 bits each, tightly packed into `Vec<u64>` with no per-element overhead.

## When to use

- Fixed-width small integers (e.g. 4-bit counters, 10-bit indices, 20-bit values)
- Memory is the bottleneck and values fit in fewer than 64 bits
- You know the maximum value at construction time

## Installation

```toml
[dependencies]
intarray = "0.2"
```

## Quick start

```rust
use intarray::{IntArray, IntArrayError};

// 7-bit unsigned integers, 1000 elements (pre-allocated)
let mut v = IntArray::new(7, 1000);

v.set(0, 100).unwrap();       // v[0] = 100
v.set(1, 127).unwrap();       // v[1] = 127 (max for 7 bits)
assert_eq!(v.get(0).unwrap(), 100);

// Out-of-range value returns Err, never panics
assert_eq!(v.set(0, 128), Err(IntArrayError::TooLarge));

// Append elements
v.push(42).unwrap();
```

## Construction

```rust
// Pre-allocated, zero-filled
let v = IntArray::new(4, 100);                        // 4-bit, 100 elements

// From a Vec
let v = IntArray::new_with_vec(4, vec![1, 2, 3, 4]);

// From an iterator
let v = IntArray::new_with_iter(4, 0..16u64);

// Infer bit width from data (same as shape_auto)
let v = IntArray::new_with_vec(8, vec![0u64, 1, 2, 3]);
let compact = v.shape_auto();  // bits = 2 (max value = 3, needs 2 bits)
```

## Element access

All access methods return `Result<_, IntArrayError>`:

| Error | When |
|---|---|
| `OutOfBounds` | index ≥ length |
| `TooLarge` | value exceeds 2^bits − 1 |
| `Empty` | `pop()` on empty array |

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

`incr_limit` / `decr_limit` clamp at the boundary and return `None` instead of overflowing:

```rust
v.incr_limit(5);  // → Some(old_value) if v[5] < max, None if already at max
v.decr_limit(5);  // → Some(old_value) if v[5] > 0, None if already at 0 or max
```

## Bulk operations

`push`, `extend`, `extend_array`, and `concat` are all atomic: on error, the array is left unchanged.

```rust
let mut v = IntArray::new(4, 0);

// Extend from an iterator of u64
v.extend(vec![1u64, 2, 3]).unwrap();

// Extend from another IntArray (fast path when bits and alignment match)
let other = IntArray::new_with_vec(4, vec![4, 5, 6]);
v.extend_array(other).unwrap();

// concat is an alias for extend_array
let more = IntArray::new_with_vec(4, vec![7, 8]);
v.concat(more).unwrap();
```

## Arithmetic operators

Element-wise arithmetic via `+=`, `-=`, `*=`. Works on both a scalar `u64` and another `IntArray`:

```rust
let mut a = IntArray::new_with_vec(8, vec![10, 20, 30]);
a += 5u64;   // [15, 25, 35]

let b = IntArray::new_with_vec(8, vec![1, 2, 3]);
a += b;      // [16, 27, 38]
```

## Iteration and statistics

```rust
let v = IntArray::new_with_vec(8, vec![3, 1, 4, 1, 5, 9]);

for x in v.iter() { println!("{}", x); }

v.sum().unwrap();      // → 23
v.min().unwrap();      // → 1
v.max().unwrap();      // → 9
v.average().unwrap();  // → 3.8333...
```

## Shape / reshape

```rust
let v = IntArray::new_with_vec(16, vec![0, 1, 1000]);

// Reshape to a different bit width (copies all elements)
let v8 = v.shape(10);

// Infer minimum bit width from max value
let compact = v.shape_auto();  // bits = 10 (needs 10 bits for 1000)

// Subarray (slice without copying when aligned)
let sub = v.subarray(1, 2);   // elements [1..3)
```

## Serialization (serde)

`IntArray` implements `Serialize` and `Deserialize` as a flat sequence of `u64`.

```rust
use serde_json;

let v = IntArray::new_with_vec(4, vec![1, 2, 3]);
let json = serde_json::to_string(&v).unwrap();   // "[1,2,3]"

let v2: IntArray = serde_json::from_str(&json).unwrap();
// Note: bit width is re-inferred from the max value (bits = 2 for max=3).
// All-zeros arrays always deserialize with bits = 1.
```

## Memory layout

```rust
let v = IntArray::new(4, 100);

v.len();       // 100  — number of elements
v.capacity();  // 112  — allocated capacity in elements (rounded to word boundary)
v.datasize();  // size in bytes (consumes v)
```

Each `u64` word holds `64 / bits` elements. For example, 4-bit integers pack 16 per word; a 100-element array uses 7 words (56 bytes of data).

## Error type

```rust
pub enum IntArrayError {
    OutOfBounds,  // index ≥ array length
    TooLarge,     // value does not fit in the configured bit width
    Empty,        // pop() on an empty array
}
```

`IntArrayError` implements `std::error::Error` and `Display`.

## MSRV

Rust 1.87 (uses `usize::is_multiple_of`, stabilized in 1.87).
