# RadixArray 設計仕様

## 概要

`RadixArray` は整数の範囲 `[A, B]` を持つ値の配列を、混合基数符号化（mixed-radix encoding）によってメモリ効率良く格納するデータ構造。

既存の `IntArray`（N ビット詰め）とは異なり、1 ワード（u64）を基数 K の数として扱い、各要素をその桁として表現する。K が 2 の冪でない場合、必ずビット詰めより多くの要素を 1 ワードに格納できる。

## 数学的根拠

### 格納効率

基数 K = B − A + 1、ワードサイズ W = 64 ビットとして：

```
epu (elements per u64) = floor(64 * ln(2) / ln(K))
```

epu は「K^epu − 1 ≤ u64::MAX < K^(epu+1) − 1」を満たす最大の整数、
すなわち「K^epu ≤ 2^64」を満たす最大の整数。
（2^64 = u64::MAX + 1 は u64 では表現できないため、検証は u128 で行う）

### ビット詰めとの比較

| K | ビット詰め epu | 基数方式 epu | 向上率 |
|---|---|---|---|
| 2 | 64 | 64 | 0% |
| 3 | 32 | 40 | +25% |
| 5 | 21 | 27 | +29% |
| 6 | 21 | 24 | +14% |
| 7 | 21 | 22 | +5% |
| 8 | 21 | 21 | 0% |
| 10 | 16 | 19 | +19% |
| 100 | 7 | 9 | +29% |

K が 2 の冪のとき、基数方式とビット詰めは等価になる。

### 符号化・復号

i 番目の要素のワードインデックスと桁位置：

```
word_idx = i / epu
digit_pos = i % epu
```

**取得（get）:**
```
stored = (data[word_idx] / K^digit_pos) % K
value  = stored + A
```

**書込（set）:**
```
stored_new  = value - A
stored_old  = (data[word_idx] / K^digit_pos) % K
data[word_idx] = data[word_idx]
               - stored_old * K^digit_pos
               + stored_new * K^digit_pos
```

## データ構造

```rust
#[derive(Clone)]  // 全フィールドが Clone を実装するため derive で十分
pub struct RadixArray {
    base: u64,          // K = B - A + 1
    offset: i64,        // A（最小値。get時に加算、set時に減算）
    epu: usize,         // elements per u64
    powers: Vec<u64>,   // powers[i] = K^i (i = 0..epu)
    length: usize,      // 要素数
    data: Vec<u64>,     // 格納ワード列
}
```

### powers テーブル

構築時に一度だけ計算し、get/set で使い回す。

```
powers[0] = 1
powers[1] = K
powers[2] = K^2
...
powers[epu-1] = K^(epu-1)
```

`powers[epu]`（= K^epu）は計算すると u64 をオーバーフローするため格納しない。ワード境界の判定は `i % epu == 0` で行う。

### ワード数

```
word_count = length.div_ceil(epu)
```

末尾ワードは部分的に使用される（未使用桁は 0）。

## エラー型

`IntArray` と同じ `IntArrayError` を共用する（依存関係として `intarray` を参照するか、`rarray` クレートに再定義する）。

```rust
pub enum RadixArrayError {
    OutOfBounds,   // index >= length
    TooLarge,      // value > B
    TooSmall,      // value < A
    Empty,         // pop() on empty array
    InvalidRange,  // K < 2 または K > u64::MAX（下記参照）
}
```

`IntArrayError` と異なり符号付き整数の範囲（A < 0 も許容）を扱うため、`TooSmall` が追加になる。

## API

### 構築

```rust
// [A, B] 範囲の要素を len 個確保し、全要素を A で初期化する
// （内部データは u64 ゼロで初期化され、get すると A が返る）
pub fn new(a: i64, b: i64, len: usize) -> Result<RadixArray, RadixArrayError>

// Vec<i64> から構築。アトミック：vals に範囲外の値が1つでもあれば Err を返し何も構築しない
pub fn new_with_vec(a: i64, b: i64, vals: Vec<i64>) -> Result<RadixArray, RadixArrayError>

// イテレータから構築。アトミック：途中で範囲外の値があれば Err を返し何も構築しない
pub fn new_with_iter<I>(a: i64, b: i64, vals: I) -> Result<RadixArray, RadixArrayError>
where I: Iterator<Item = i64>
```

### 要素アクセス

```rust
pub fn get(&self, i: usize) -> Result<i64, RadixArrayError>
pub fn set(&mut self, i: usize, v: i64) -> Result<(), RadixArrayError>
```

### スタック操作（atomic）

```rust
// 末尾に追加。成功時は新しいインデックスを返す
pub fn push(&mut self, v: i64) -> Result<usize, RadixArrayError>

// 末尾から取り出す
pub fn pop(&mut self) -> Result<i64, RadixArrayError>
```

### バルク操作（すべて atomic）

```rust
// イテレータから追加。エラー時は配列を元のサイズに戻す
pub fn extend<I>(&mut self, vals: I) -> Result<(), RadixArrayError>
where I: IntoIterator<Item = i64>

// 別の RadixArray から追加。vals の各要素を i64 値として取り出し self に push する。
// self の範囲 [A, B] に収まらない値があれば Err（アトミック）。
// fast path: vals.base == self.base かつ self.length % self.epu == 0 のとき
//            raw word をそのままコピーする（get/set を経由しない）。
// slow path: それ以外は extend(vals.iter()) に委譲。
pub fn extend_array(&mut self, vals: &RadixArray) -> Result<(), RadixArrayError>
```

### 情報取得

```rust
pub fn len(&self) -> usize
pub fn is_empty(&self) -> bool
pub fn capacity(&self) -> usize   // data.len() * epu
pub fn base(&self) -> u64         // K
pub fn range(&self) -> (i64, i64) // (A, B)

// メモリ使用量（バイト）
pub fn datasize(&self) -> usize
```

### イテレータ

```rust
pub fn iter(&self) -> RadixIter<'_>
```

`Iterator<Item = i64>` を実装。

### 統計

```rust
pub fn sum(&self) -> Option<i128>  // i64 ではオーバーフローするため i128
pub fn min(&self) -> Option<i64>
pub fn max(&self) -> Option<i64>
pub fn average(&self) -> Option<f64>
```

### リサイズ

```rust
// 内部用。末尾を切り詰めるか拡張する。
// 拡張時は新規ワードを 0 で初期化するため、追加された要素は get すると A が返る。
fn resize(&mut self, len: usize)
```

### Display

```rust
impl fmt::Display for RadixArray {
    // 例: "[A,B][5]=-1,0,1,0,-1"
}
```

## 実装上の注意点

### K のバリデーション（オーバーフロー対策）

`new` の先頭で i128 を使って K を計算し、範囲を確認する。

```rust
let k = b as i128 - a as i128 + 1;
if k < 2 || k > u64::MAX as i128 {
    return Err(RadixArrayError::InvalidRange);
}
let base = k as u64;
```

- `k < 2`: A ≥ B（A == B で K=1 の場合も含む）
- `k > u64::MAX`: 実質 A = i64::MIN かつ B = i64::MAX のケースのみ。u64 に収まらないため弾く。
- i64 同士の減算を直接行うと debug ビルドでパニックするため、必ず i128 にキャストしてから計算する。

### epu の導出

```rust
fn calc_epu(base: u64) -> usize {
    // 2^64 = u64::MAX + 1 なので f64 で近似計算し、
    // 前後を整数検証して確定させる
    let epu_approx = (64.0 * std::f64::consts::LN_2 / (base as f64).ln()) as usize;
    // epu_approx - 1, epu_approx, epu_approx + 1 の3候補を u128 で検証
    // base^epu <= u64::MAX となる最大の epu を返す
    ...
}
```

f64 の精度誤差で ±1 ずれる可能性があるため、u128 で `base^candidate <= u64::MAX` を確認してから確定する。

### powers テーブルの計算

```rust
fn calc_powers(base: u64, epu: usize) -> Vec<u64> {
    let mut p = vec![1u64; epu];
    for i in 1..epu {
        p[i] = p[i - 1] * base; // オーバーフロー不可（K^(epu-1) <= u64::MAX）
    }
    p
}
```

`powers[epu-1]` = K^(epu-1) は必ず u64 に収まる（K^epu が u64 を超えない最大の epu を選んでいるため）。

### get / set の実装詳細

```rust
pub fn get(&self, i: usize) -> Result<i64, RadixArrayError> {
    if i >= self.length { return Err(RadixArrayError::OutOfBounds); }
    let (widx, dpos) = (i / self.epu, i % self.epu);
    let stored = (self.data[widx] / self.powers[dpos]) % self.base;
    Ok(stored as i64 + self.offset)
}

pub fn set(&mut self, i: usize, v: i64) -> Result<(), RadixArrayError> {
    // IntArray と同じ順序: OutOfBounds を先に返す
    if i >= self.length { return Err(RadixArrayError::OutOfBounds); }
    if v < self.offset { return Err(RadixArrayError::TooSmall); }
    if v > self.offset + self.base as i64 - 1 { return Err(RadixArrayError::TooLarge); }
    let (widx, dpos) = (i / self.epu, i % self.epu);
    let p = self.powers[dpos];
    let old = (self.data[widx] / p) % self.base;
    let new = (v - self.offset) as u64;
    self.data[widx] = self.data[widx] - old * p + new * p;
    Ok(())
}
```

### extend_array の fast path

```rust
pub fn extend_array(&mut self, vals: &RadixArray) -> Result<(), RadixArrayError> {
    if vals.base == self.base && self.length % self.epu == 0 {
        // fast path: self がワード境界にアラインされており、base が同じ
        // → vals.data をそのまま連結できる
        self.length += vals.length;
        self.data.extend_from_slice(&vals.data);
        return Ok(());
    }
    // slow path: 値を1つずつ取り出して push（アトミック性は extend が保証）
    self.extend(vals.iter())
}
```

fast path が成立する条件：
- `vals.base == self.base`: ワードの符号化方式が同一
- `self.length % self.epu == 0`: self の末尾がワード境界にある（末尾ワードが埋まりきっている）

fast path では値の範囲チェックを行わない（同じ base なら vals の各要素は必ず self の範囲に収まる）。

### 末尾ワードの境界

`push` / `resize` で要素を増やす際、新しいワードを追加するのは `length % epu == 0` のとき。  
逆に `pop` / `resize` で減らす際も同様に末尾ワードを解放する。

### バルク操作のアトミック性

```rust
pub fn extend<I>(&mut self, vals: I) -> Result<(), RadixArrayError>
where I: IntoIterator<Item = i64>
{
    let orig = self.length;
    for v in vals {
        if let Err(e) = self.push(v) {
            self.resize(orig); // ロールバック
            return Err(e);
        }
    }
    Ok(())
}
```

`IntArray::extend` と同じパターン。

### 速度に関するトレードオフ

| 操作 | ビット詰め（IntArray） | 基数方式（RadixArray） |
|---|---|---|
| get | シフト + マスク（〜3サイクル）| 除算2回 + 剰余（〜60サイクル）|
| set | シフト + マスク + OR | 除算 + 乗算 + 加減算 |
| メモリ（K=3）| 32 要素/u64 | 40 要素/u64（+25%）|

CPU バウンドな処理では遅くなる。メモリバウンドな大規模データでは、キャッシュライン効率の向上により逆に速くなる可能性がある。

## Serde

`IntArray` と同様にフラットな整数列として直列化する。

- Serialize: `[i64, ...]`（A を加算した実際の値）
- Deserialize: 値列の min・max から A・B を自動設定する。**情報損失に注意**（下記）

```
serialize([−1, 0, 1]) → [-1, 0, 1]
deserialize([-1, 0, 1]) → RadixArray { a: -1, b: 1, ... }
```

**空配列のデシリアライズ:**  
min/max が求まらないため、`a=0, b=1`（K=2）をデフォルトとする。IntArray の空配列デシリアライズが bits=1 になるのと対応する。

```
deserialize([]) → RadixArray { a: 0, b: 1, length: 0 }
```

**Note:** シリアライズ時に A・B は保存されない。デシリアライズ後の範囲は元の範囲と異なる場合がある。

```
// 元の範囲 [-5, 5] だが、値が [0, 1] に収まっている場合
serialize(RadixArray { a: -5, b: 5, vals: [0, 1, 0] }) → [0, 1, 0]
deserialize([0, 1, 0]) → RadixArray { a: 0, b: 1 }  // a=-5, b=5 は失われる
```

また、全要素が同じ値の場合は min == max となり K=1 になるため、min を A、max+1 を B（K=2）として扱う。

```
deserialize([3, 3, 3]) → RadixArray { a: 3, b: 4 }  // K=2
```

## ファイル構成（案）

現行の `intarray` クレートとは別クレートにする。`IntArrayError` は共通化せず、`RadixArrayError` を独立して定義する。

```
rarray/
  Cargo.toml
  src/
    lib.rs       # RadixArray 本体
    iter.rs      # RadixIter
    tests.rs     # テスト
  examples/
    basic.rs
```

## テスト方針

- `epu` の境界（要素インデックス 0、epu-1、epu、epu+1）
- 末尾ワードの部分使用（length が epu の倍数でない場合）
- 範囲外の値（TooLarge / TooSmall）
- ロールバック（extend で途中エラー → 元のサイズに戻ること）
- 各 K ごとに epu の計算が正しいことを確認
- K が 2 の冪（K=2, 4, 8）でビット詰めと同等の epu になること
- 負の範囲（A < 0 < B）
- A = B は K=1（1 種類の値）→ `InvalidRange` として弾く

## 将来の拡張（スコープ外）

- `shape(a, b)` — 範囲を変えてコピー（値が収まらない場合エラー）
- `AddAssign` / `SubAssign` — SIMD 的な一括加算（基数方式では非自明）
- `fill_random` — 範囲内の乱数で埋める
- `subarray` — 部分配列（ワード境界で高速パスを取るのは困難）
