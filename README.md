# Memory Efficient Integer Array

## Usage

Cargo.toml

```
[dependencies]
intarray = "0.1.0"
```

```rust
use intarray;

let mut v = intarray::IntArray::new(7, 999);  // 7-bit unsigned integer, 999 entries
v.set(10, 20).unwrap();                       // v[10] = 20
v.get(10).unwrap();                           // get v[10]
```
