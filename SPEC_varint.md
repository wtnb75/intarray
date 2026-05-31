# VarIntArray 仕様

## 概要

`VarIntArray` は、任意精度整数（`num_bigint::BigInt`）を可変長符号（Elias gamma）で
ビット詰めし、ブロック単位で管理するパック配列である。

値ごとに必要最小限のビット数を使うため、小さい値が多いほど高いメモリ効率が得られる。
ブロック単位の管理により、要素の書き換え（`set`）を O(K) に抑える。

### 既存の型との位置づけ

| 型 | 要素型 | ビット幅 | 主な用途 |
|---|---|---|---|
| `IntArray` | `u64` | 固定 (1〜64) | 最大値が既知の符号なし整数 |
| `RadixArray` | `i64` | 固定（範囲 [A,B] で決まる） | 範囲が既知の符号あり整数 |
| `FloatArray` | `f64` | 固定 (exp+man+1) | 浮動小数点 |
| `VarIntArray` | `BigInt` | **可変** | 最大値が未知・任意精度 |

---

## 符号化方式

### Step 1: Zigzag エンコーディング（符号 → 非負）

符号あり `BigInt` を非負の `BigUint` に変換する：

```
n =  0  →  z = 0
n >  0  →  z = 2 * n
n <  0  →  z = 2 * |n| - 1
```

逆変換：

```
z が偶数  →  n = z / 2
z が奇数  →  n = -(z + 1) / 2
```

### Step 2: BigUint の Elias gamma 符号化

`BigUint z` を自己区切り型ビット列に変換する：

1. `B = z.bits() as usize`（`BigUint::bits()` は `u64` を返す。z の最小表現ビット数で、z=0 のとき B=0）
2. `B+1` を標準 Elias gamma で符号化（B=0 のとき "1" の 1 ビット）
3. z の上位 B ビットをそのまま追記（MSB ファースト）

**標準 Elias gamma（正整数 n ≥ 1）：**

```
k = floor(log2(n))
符号: k 個の 0 + "1" + n の下位 k ビット
合計: 2k+1 ビット
```

**符号化例（BigInt → ビット列）：**

| BigInt n | Zigzag z | B | gamma(B+1) | データ | 合計 |
|---|---|---|---|---|---|
| 0 | 0 | 0 | "1" | （なし） | 1 bit |
| -1 | 1 | 1 | "010" | "1" | 4 bits |
| 1 | 2 | 2 | "011" | "10" | 5 bits |
| -2 | 3 | 2 | "011" | "11" | 5 bits |
| 2 | 4 | 3 | "00100" | "100" | 8 bits |
| 127 | 254 | 8 | "0001001" | "11111110" | 15 bits |
| 2^64 | 2^65 | 66 | 13 bits | 66 bits | 79 bits |
| 2^1000 | 2^1001 | 1002 | 19 bits | 1002 bits | 1021 bits |

**Elias gamma のデコード手順（1値分）：**

```
1. 連続する 0 ビットの数を数える → k
   （"1" ビットが来るまで 1 ビットずつ読む）
2. "1" ビットを読んで捨てる（区切り）
3. k ビットを読む → lower_bits
4. B+1 = 2^k + lower_bits  → B = 2^k + lower_bits - 1
```

例: ビット列 "0001001..." を読む場合
- 0 が 3 個 → k=3
- "1" を読んで捨てる
- 次の 3 ビット "001" を読む → lower_bits=1
- B+1 = 2^3 + 1 = 9 → B = 8

**BigUint のデコード手順（1値分）：**

```
1. 上記 Elias gamma デコードで B を得る
2. B == 0 の場合: z = 0（データビットなし）
3. B > 0 の場合: 次の B ビットを MSB ファーストで読む → z
```

**BigInt のデコード手順（1値分）：**

```
1. 上記で z（BigUint）を得る
2. 逆 Zigzag:
   z が偶数 → n = z / 2
   z が奇数 → n = -(z + 1) / 2
```

**自己区切り性（self-delimiting）：**

Elias gamma 自体が自己区切り符号（先頭の連続する 0 の個数 k が長さを示す）であり、
B+1 を読み終えた時点で「次に読むべきビット数 = B」が確定する。
したがって、ブロック内のビット列を先頭から順に読むだけで、
各要素の境界を外部インデックスなしに正確に検出できる。

**複数要素のブロック内ビット列の例：**

`[0, -1, 1, -2]` を k=4 のブロックに格納した場合：

```
要素  n    z   B  gamma(B+1)  データ  要素のビット列
  0   0    0   0  "1"         ""      "1"       (1 bit)
 -1  -1    1   1  "010"       "1"     "0101"    (4 bits)
  1   1    2   2  "011"       "10"    "01110"   (5 bits)
 -2  -2    3   2  "011"       "11"    "01111"   (5 bits)

連結: "1" | "0101" | "01110" | "01111"
    = "101010111001111"  (15 bits)

MSB ファーストでバイト列に格納（端数は 0 パディング）：
  b0〜b7  : 1,0,1,0,1,0,1,1 → data[0] = 0b10101011
  b8〜b14 : 1,0,0,1,1,1,1   → data[1] = 0b10011110  (最下位 1bit はパディング)

bit_len = 15, count = 4
```

### ビット列のバイト格納（MSB ファースト）

```
ビット列: [b0 b1 b2 b3 b4 b5 b6 b7 | b8 b9 ...]
バイト:    [ data[0]                 | data[1] ...]

bit_pos p は data[p/8] のビット (7 - p%8) に対応する（MSB が先）
```

書き込み（bit_pos の位置に 1 を立てる）：
```
byte_idx = bit_pos / 8
bit_idx  = 7 - (bit_pos % 8)
data[byte_idx] |= (1u8 << bit_idx)
```

読み取り（bit_pos の位置のビット値を得る）：
```
byte_idx = bit_pos / 8
bit_idx  = 7 - (bit_pos % 8)
bit = (data[byte_idx] >> bit_idx) & 1
```

新しいビットを末尾に追記するとき（`data` の拡張が必要な場合）：
```
if bit_len % 8 == 0 {
    data.push(0u8)   // 新しいバイトを確保
}
// 上記の書き込み式で data[bit_len / 8] に書く
bit_len += 1
```

---

## データ構造

```
VarIntArray {
    k:      usize,           // ブロックサイズ（構築時に指定）
    blocks: Vec<BitBlock>,   // ブロックの配列
    length: usize,           // 総要素数
}

BitBlock {
    data:    Vec<u8>,        // Elias gamma ビット列（MSB ファースト）
    bit_len: usize,          // data 内の有効ビット数
    count:   usize,          // このブロックの要素数
}
```

**ブロックの不変条件：**
- `blocks` が空でない場合、最後以外のすべてのブロックの `count == k`
- 最後のブロックの `count` は 1 以上 k 以下
- 空配列のとき `blocks` は空

---

## 構成パラメータの制約

| 条件 | 理由 |
|---|---|
| `k >= 1` | ブロックサイズが 0 は無意味 |

違反時: `Err(ArrayError::InvalidRange)`

### k の選択ガイド

| k | get/set コスト | インデックスサイズ |
|---|---|---|
| 16 | 速い | やや多い |
| 64 | 中程度（推奨） | 少ない |
| 256 | 遅い | 非常に少ない |

---

## API

### 構築

```rust
// 空の配列を作成
VarIntArray::new(k: usize) -> Result<Self, ArrayError>

// Vec<BigInt> から構築
VarIntArray::new_with_vec(k: usize, vals: Vec<BigInt>) -> Result<Self, ArrayError>

// イテレータから構築
VarIntArray::new_with_iter(k: usize, vals: impl Iterator<Item=BigInt>) -> Result<Self, ArrayError>
```

### 要素アクセス

```rust
get(i: usize) -> Result<BigInt, ArrayError>
// ブロック blocks[i/k] の先頭から i%k 要素デコードして返す
// Err: OutOfBounds

set(i: usize, v: BigInt) -> Result<(), ArrayError>
// blocks[i/k] を全デコード → 差し替え → 再エンコード
// Err: OutOfBounds
```

`get` はブロック先頭から対象要素まで**部分デコード**（最大 `i%k + 1` 要素）。全要素を復元する必要はない。
`set` はブロックの**全要素をデコード**してから差し替え・再エンコードする。再エンコード後のブロックサイズは変わる場合がある（`bit_len` と `data` の長さを更新する）。
どちらも最悪計算量は O(K)。

### スタック操作

```rust
push(v: BigInt) -> Result<usize, ArrayError>
// 最後のブロックに追記。満杯（count == k）なら新ブロック作成。
// 戻り値: 追加された要素のインデックス
// 任意の BigInt を格納可能なため値エラーは発生しない

pop() -> Result<BigInt, ArrayError>
// 最後のブロックを全デコード → 末尾を取り出し → 再エンコード
// ブロックが空になったら blocks から削除
// Err: Empty
```

**pop の詳細手順：**

```
1. blocks が空なら Err(Empty)
2. 最後のブロック（blocks.last()）の全要素をデコード → elems: Vec<BigInt>
3. ret = elems.pop() （末尾を取り出す）
4. elems が空（count が 1 だった）なら blocks.pop() して self.length -= 1 → return Ok(ret)
5. 最後のブロックを空の BitBlock に置き換える:
       *blocks.last_mut() = BitBlock { data: vec![], bit_len: 0, count: 0 }
6. elems の各要素を push の手順（step 2〜7）で書き戻す
   （push を呼ぶと self.length が変化するため、直接 blocks.last_mut() に書き込む）
7. self.length -= 1
8. return Ok(ret)
```

**push の詳細手順：**

```
1. blocks が空、または最後のブロックの count == k の場合:
       blocks.push(BitBlock { data: vec![], bit_len: 0, count: 0 })

2. v を Zigzag エンコード → z (BigUint)
3. B = z.bits() as usize   // num-bigint の bits() は u64 を返すので usize にキャスト
4. B+1 を標準 Elias gamma で符号化 → bits_to_write: Vec<bool>
       k = floor(log2(B+1))  ← B+1 >= 1 なので常に定義される（B=0 のとき k=0）
       k 個の 0 ビット、"1" ビット、B+1 の下位 k ビット（MSB から順）
5. z の上位 B ビットを MSB ファーストで bits_to_write に追加
6. bits_to_write の各ビットを、上記「新しいビットを末尾に追記」の手順で
   blocks.last_mut() の data に書き込む（bit_len を 1 ずつ進める）
7. blocks.last_mut().count += 1
8. self.length += 1
9. return Ok(self.length - 1)
```

### 一括操作

```rust
extend(vals: impl IntoIterator<Item=BigInt>) -> Result<(), ArrayError>
// 各要素を push する。値エラーは発生しない。

extend_array(other: &VarIntArray) -> Result<(), ArrayError>
// other の全要素を self に追加する。other の k と self の k が異なっても動作する。
// fast path なし（ブロック境界が一致しないため）。iter() → push() で実装する。
```

`extend` / `extend_array` は atomic ではない（値エラーが発生しないためロールバック不要）。

### 統計・イテレーション

```rust
iter(&self) -> VarIntIter       // ExactSizeIterator<Item=BigInt>; インデックス 0 から順に返す
sum() -> Option<BigInt>         // None if empty
min() -> Option<BigInt>         // None if empty
max() -> Option<BigInt>         // None if empty
average() -> Option<f64>        // None if empty; BigInt → f64 変換で精度が落ちる場合がある
len() -> usize
is_empty() -> bool
```

`VarIntIter` はビット位置を直接保持し、`next()` のたびに 1 要素をデコードして返す：

```rust
struct VarIntIter<'a> {
    arr: &'a VarIntArray,
    block_idx: usize,    // 現在のブロック番号
    elem_in_block: usize, // ブロック内の要素番号（0-origin）
    bit_pos: usize,      // ブロック内の現在読み取りビット位置
    remaining: usize,    // 残り要素数（ExactSizeIterator 用）
}
```

`next()` の動作：
1. `remaining == 0` なら `None` を返す
2. `elem_in_block == arr.blocks[block_idx].count` なら次のブロックへ進む
   （`block_idx += 1`, `elem_in_block = 0`, `bit_pos = 0`）
3. 現在ブロックの `bit_pos` から 1 要素デコード、`bit_pos` を進める
4. `elem_in_block += 1`, `remaining -= 1` して値を返す

`sum` / `min` / `max` はすべて `iter()` を通じて逐次計算する（O(n)）。
`average` は `sum()` を `num_traits::ToPrimitive::to_f64()` で `f64` に変換してから要素数（`usize as f64`）で除算する。
`BigInt` が `f64` の精度（53 bit 仮数）を超える場合は精度損失が生じる。`to_f64()` は `num-traits = "0.2"` の `ToPrimitive` トレイトが提供する。
sum の絶対値が `f64::MAX`（約 1.8×10^308）を超える場合、`to_f64()` は `±Infinity` を返すため `average()` も `±Infinity` になる。`None` にはならない。

### メタデータ

```rust
block_size() -> usize          // = k（構築時に指定した値）
block_count() -> usize         // = blocks.len()
datasize() -> usize
// size_of::<VarIntArray>()
// + size_of::<BitBlock>() * blocks.capacity()   // Vec<BitBlock> の確保済み領域
// + 各ブロックの data.capacity()                // 各 Vec<u8> の確保済みバイト数の合計
```

---

## エラー型

`VarIntArray` が返す `ArrayError` の種類：

| エラー | 発生箇所 |
|---|---|
| `OutOfBounds` | `get` / `set` のインデックス超過 |
| `Empty` | `pop` を空配列に対して呼んだ |
| `InvalidRange` | `new` で `k == 0` |

`TooLarge` / `TooSmall` は発生しない（任意の `BigInt` を格納可能）。

---

## 操作の計算量まとめ

| 操作 | 時間計算量 | 備考 |
|---|---|---|
| `get(i)` | O(K) | ブロック内スキャン |
| `set(i, v)` | O(K) | ブロック全体の再符号化 |
| `push(v)` | O(1) 均し | ブロック満杯時のみ新規作成 |
| `pop()` | O(K) | ブロック全体の再符号化 |
| `extend(n個)` | O(n) | push の繰り返し |
| `iter()` 全走査 | O(n) | ブロックを順に展開 |

---

## `PackedArrayCore` トレイトとの関係

`BigInt` は `Copy` を実装しないため、`PackedArrayCore` トレイト（`type Item: Copy` が必要）は使用しない。
`push` / `pop` / `extend` は `VarIntArray` が独自に実装する。

---

## 表示 (Display)

```
[k=64][5]=0,-1,1,-2,1000000000000000000000
  ^    ^
  k    length
```

大きな値は十進数文字列で表示する。

---

## シリアライズ (serde)

- **Serialize**: 要素を十進文字列のフラット配列として出力（JSON の数値型では任意精度 BigInt を表現できないため）。
- **Deserialize**: 文字列配列から再構築。`k` は保存されないため、デシリアライズ時はデフォルト値（`k = 64`）を使用する。

```json
["0", "-1", "1", "-2", "1000000000000000000000"]
```

> **注意**: デシリアライズで `k` は保存されない。元の `k` と異なる場合がある。
> ただし `PartialEq` は要素値のみで比較する（`k` とブロック構造を無視）ため、
> serde round-trip の前後で `==` は成立する。

---

## Cargo.toml への追加

```toml
[dependencies]
num-bigint = { version = "0.4", features = ["serde"] }
num-traits = "0.2"
```

---

## 設計上の決定事項

| 項目 | 決定 | 理由 |
|---|---|---|
| 要素型 | `num_bigint::BigInt` | 任意精度、符号あり、Rust 標準的 |
| 符号処理 | Zigzag エンコーディング | 負の値を非負に変換してから Elias gamma を適用 |
| 長さ符号化 | Elias gamma (B+1) | 0 も自然に扱える、オーバーヘッドが O(log B) |
| ビット順序 | MSB ファースト | Elias gamma のゼロ列を素直に表現できる |
| 可変性 | Full（set 可能） | O(K) でブロック単位の再符号化 |
| ブロックサイズ k | 構築時に指定 | メモリと速度のトレードオフをユーザーが制御 |
| `PackedArrayCore` | 使用しない | `BigInt` が `Copy` でないため |
| serde 表現 | 十進文字列配列 | 任意精度を JSON で正確に表現 |
| `PartialEq` | 要素値のみ比較（k・ブロック構造を無視） | serde round-trip 後も `==` が成立するようにするため |
| デシリアライズ時の k | 64（`DEFAULT_K`） | k は外部表現に含まれないため固定値が必要 |

---

## 実装上の注意点

### 内部不変条件

`BitBlock` のビット列は常に `encode_into` → `read_elias_gamma` の組み合わせで読み書きされる。
`read_bit` / `read_elias_gamma` は不変条件が破れた場合（`data` が短すぎる等）に panic する。
外部から `BitBlock` の `data` を直接書き換えることはできないため、通常の使い方では問題ない。

### 32 ビット環境

`BigUint::bits()` は `u64` を返すが、内部で `usize` にキャストする。
32 ビット環境（`usize = u32`）では、2^32 ビット（約 500 MB の整数）を超える `BigUint` は
切り詰められて誤ったエンコードになる。実用上は 64 ビット環境のみを想定している。

### k の上限

現行実装では `k == 0` のみ弾く（`Err(InvalidRange)`）。
`k = usize::MAX` のような極端な値を設定すると全要素が 1 ブロックに収まり続け、
`set` / `pop` が O(n) に劣化する。大きな k を使う際は注意。

---

## 疑問点・未解決事項

（現時点ではなし）
