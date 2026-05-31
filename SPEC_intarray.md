# IntArray 仕様

## 概要

`IntArray` は、1〜64 ビットの符号なし整数を `Vec<u64>` にビット詰めして格納するパック配列である。
要素型は `u64`（内部型エイリアス `Element`）。

メモリ効率を最大化するため、各要素は `bits` ビットだけを消費し、
複数の要素を1つの `u64` ワードに隙間なく詰め込む。

---

## パッキング方式（ビット詰め）

```
bpd = 64 / bits          // elements per word（小数切り捨て）
word_count = ceil(len / bpd)
```

要素 `i` の格納位置：
```
word_index = i / bpd
bit_position = i % bpd
bit_offset = bit_position * bits

get:  (data[word_index] >> bit_offset) & max_value
set:  data[word_index] = (data[word_index] & !(max_value << bit_offset))
                        | (v << bit_offset)
```

- `max_value = (1u64 << bits) - 1`（bits < 64 の場合）、または `u64::MAX`（bits == 64 の場合）
- `bits` が 64 を割り切れない場合、各ワードの上位ビットは未使用（常に 0）
- 最終ワードも同様に部分的に使用される場合がある

### パッキング効率

| bits | bpd | 効率 |
|---|---|---|
| 1 | 64 | 100% |
| 4 | 16 | 100% |
| 7 | 9 | 98.4%（63/64 bit 使用） |
| 8 | 8 | 100% |
| 10 | 6 | 93.8% |
| 16 | 4 | 100% |
| 32 | 2 | 100% |
| 64 | 1 | 100% |

---

## 構成パラメータの制約

`bits` は `1 ≤ bits ≤ 64` を満たさなければならない。
違反した場合、コンストラクタは **panic** する（`Result` を返さない）。

---

## API

### 構築

```rust
// 全要素を 0 で初期化。bits が範囲外なら panic。
IntArray::new(bits: usize, len: usize) -> IntArray

// Vec<u64> から構築。値が max_value を超えると panic。
IntArray::new_with_vec(bits: usize, vals: Vec<u64>) -> IntArray

// イテレータから構築。値が max_value を超えると panic。
// 内部で 1024 要素単位にバッファし、最後に resize する。
IntArray::new_with_iter(bits: usize, vals: impl Iterator<Item=u64>) -> IntArray
```

> **注意**: `new_with_vec` / `new_with_iter` は `Result` を返さず、
> 範囲外の値が渡された場合は panic する。
> 呼び出し側で事前にバリデーションするか、`push` / `extend`（Result を返す）を使うこと。

### 要素アクセス

```rust
get(i: usize) -> Result<u64, ArrayError>   // Err(OutOfBounds)
set(i: usize, v: u64) -> Result<(), ArrayError>  // Err(OutOfBounds) / Err(TooLarge)
max_value() -> u64                          // 格納可能な最大値 = 2^bits - 1
```

### インクリメント／デクリメント

```rust
incr(i: usize) -> Result<(), ArrayError>   // v[i] += 1。max_value で Err(TooLarge)
decr(i: usize) -> Result<(), ArrayError>   // v[i] -= 1。0 で Err(TooSmall)
add(i: usize, v: u64) -> Result<(), ArrayError>
sub(i: usize, v: u64) -> Result<(), ArrayError>

// 境界でエラーではなく None を返す（クランプ）
incr_limit(i: usize) -> Option<u64>   // 成功時は変更前の値を返す。既に max なら None
decr_limit(i: usize) -> Option<u64>   // 成功時は変更前の値を返す。既に 0 なら None
```

### スタック操作

```rust
push(v: u64) -> Result<usize, ArrayError>   // 戻り値は追加した要素のインデックス
pop() -> Result<u64, ArrayError>            // Err(Empty)
```

`push` は `v > max_value` のとき `Err(TooLarge)` を返し、配列は変化しない。

### 一括操作

```rust
extend(vals: impl IntoIterator<Item=u64>) -> Result<(), ArrayError>
extend_array(other: &IntArray) -> Result<(), ArrayError>
```

`extend` は atomic: 途中でエラーが発生した場合、元の長さまでロールバックする。

`extend_array` の fast path: `self.bits == other.bits` かつ `self.length % bpd == 0` のとき、
raw ワードをそのままコピーする（要素ごとの get/set をスキップ）。
それ以外は slow path（要素ごとに `get` → `set`）。

### 形状変換

```rust
shape(bits: usize) -> IntArray      // 異なるビット幅に変換（全要素コピー）
shape_auto() -> IntArray            // max 値から最小ビット幅を自動推定して変換
subarray(offset: usize, length: usize) -> IntArray  // 部分配列の抽出
```

`subarray` の fast path: `offset % bpd == 0` のとき raw ワードコピー。

### 統計

```rust
iter() -> IntIter        // インデックス 0 から順に u64 を返す
sum() -> Option<u128>    // None if empty
min() -> Option<u64>     // None if empty
max() -> Option<u64>     // None if empty
average() -> Option<f64> // None if empty
```

`sum` は `bits ≤ 32` のとき SIMD ライクなワード単位の並列加算で高速化する（後述）。

### 乱数充填

```rust
fill_random(&mut self)
```

全要素を `[0, max_value]` の一様乱数で埋める。
`bits` が 64 を割り切る場合は `bpd - 1` ワードを丸ごとランダム生成（高速）。
最終ワードは要素ごとに設定する。

### メタデータ

```rust
len() -> usize
is_empty() -> bool
capacity() -> usize    // data.len() * bpd（確保済みの要素数）
datasize() -> usize    // 構造体 + Vec データのバイト数
```

---

## エラー型

`IntArray` が返す `ArrayError` の種類：

| エラー | 発生箇所 |
|---|---|
| `OutOfBounds` | `get` / `set` のインデックス超過 |
| `TooLarge` | `set` / `push` / `add` / `incr` で値が `max_value` を超える |
| `TooSmall` | `sub` / `decr` で値が 0 を下回る |
| `Empty` | `pop` を空配列に対して呼んだ |

---

## 算術演算子

要素ごとのインプレース演算。スカラー `u64` と `&IntArray` の両方をサポート。

```rust
arr += scalar: u64    // 全要素に scalar を加算
arr -= scalar: u64    // 全要素から scalar を減算
arr *= scalar: u64    // 全要素に scalar を乗算
arr += &other         // 要素ごとに加算（長さが異なれば短い方に合わせる）
arr -= &other         // 要素ごとに減算
```

オーバーフロー・アンダーフロー時は `unwrap()` によって panic する。

スカラー演算・同一 bits 長 `IntArray` 演算の fast path: ワード単位の並列演算（`add_bits` / `sub_bits` / `mul_bits`）を使用。異なる bits または長さの場合は slow path（要素ごとの get/set）。

---

## sum の高速化（bits ≤ 32 のとき）

同一ワード内の複数要素を1回の演算でまとめて加算する再帰的な並列加算：

```
初期: 1ワードに n 個の bits-bit 要素が詰まっている
step1: マスクで偶数・奇数スロットを分離して加算 → 2*bits-bit 要素が n/2 個になる
step2: 同様に繰り返す
...
最終: 64-bit の合計が1つ残る
```

この最適化は `bits ≤ 32`（= 1ワードに2要素以上入る）のときのみ有効。
`bits > 32` のときは `sum0()`（イテレータによる逐次加算）にフォールバック。

---

## 表示 (Display)

```
4[10]=0,1,2,3,4,5,6,7,8,9
^  ^
|  length
bits
```

---

## シリアライズ (serde)

- **Serialize**: 要素を `u64` のフラット配列として出力。`bits` 情報は含まない。
- **Deserialize**: フラット配列から再構築。`bits` は最大値から自動推定（`bits_for_max`）。
  - 全要素が 0 の場合: `bits = 1`
  - 最大値が `m` の場合: `bits = floor(log2(m)) + 1`

> **注意**: デシリアライズで `bits` は保存されない。
> 元の `bits` より小さい値に推定される可能性がある。

---

## 設計上の特記事項

| 項目 | 動作 |
|---|---|
| コンストラクタのエラー処理 | panic（`Result` を返さない） |
| `push` / `extend` のエラー処理 | `Result` を返す |
| オーバーフロー（演算子） | panic |
| 未使用ビット（端数ワード） | 常に 0 に保たれる |
| `bits` の公開フィールド | `pub bits: usize`（読み取り可能） |
| `length` の公開フィールド | `pub length: usize`（読み取り可能） |
