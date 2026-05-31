# RatioArray 仕様

## 概要

`RatioArray` は、任意精度有理数（`num_rational::Ratio<BigInt>`）を
可変長符号（Elias gamma）でビット詰めし、ブロック単位で管理するパック配列である。

分子と分母をそれぞれ Elias gamma で直列に符号化するため、
小さい分母（特に分母=1の整数）を多く含む場合にメモリ効率が高くなる。

`Ratio<BigInt>` は常に既約分数に正規化される（GCD で約分、分母は正）。
`RatioArray` はこの正規化済みの値を受け取って格納し、取り出す。

### 既存の型との位置づけ

| 型 | 要素型 | 主な用途 |
|---|---|---|
| `IntArray` | `u64` | 符号なし整数（固定ビット幅） |
| `RadixArray` | `i64` | 符号あり整数（固定範囲） |
| `FloatArray` | `f64` | 浮動小数点（固定精度） |
| `VarIntArray` | `BigInt` | 可変長整数 |
| `RatioArray` | `Ratio<BigInt>` | 有理数（丸め誤差なし） |

---

## 符号化方式

`Ratio<BigInt>` 値 `p/q`（`q ≥ 1`、既約）を以下の順でビット列に変換する：

```
[分子 p の符号化] [分母 q の符号化]
```

### 分子 p の符号化（VarIntArray と同一）

符号あり `BigInt p` を非負の `BigUint z` に Zigzag 変換してから拡張 Elias gamma で符号化する。

**Zigzag エンコーディング：**

```
p =  0  →  z = 0
p >  0  →  z = 2 * p
p <  0  →  z = 2 * |p| - 1
```

逆変換：

```
z が偶数  →  p = z / 2
z が奇数  →  p = -(z + 1) / 2
```

**拡張 Elias gamma（BigUint z ≥ 0）：**

1. `B = z.bits() as usize`（`BigUint::bits()` は `u64` を返す。z=0 のとき B=0）
2. `B+1` を標準 Elias gamma で符号化（k=floor(log2(B+1))、"k個の0" + "1" + "B+1の下位kビット MSBファースト"）
3. z の上位 B ビットを MSB ファーストで追記

デコードは VarIntArray §Elias gamma のデコード手順 と同一。

### 分母 q の符号化（標準 Elias gamma）

`q` は常に `q ≥ 1` であるため、標準 Elias gamma をそのまま適用できる：

**エンコード：**

```
k_bits = q.bits() as usize - 1   // = floor(log2(q))。q ≥ 1 なので常に定義される
符号: k_bits 個の 0 + "1" + q の下位 k_bits ビット（MSB ファースト）
合計: 2*k_bits + 1 ビット
```

**符号化例（分母）：**

| q | k_bits | 符号 | ビット数 |
|---|---|---|---|
| 1 | 0 | "1" | 1 |
| 2 | 1 | "010" | 3 |
| 3 | 1 | "011" | 3 |
| 4 | 2 | "00100" | 5 |
| 7 | 2 | "00111" | 5 |
| 10 | 3 | "0001010" | 7 |

**デコード：**

```
1. 連続する 0 ビットの数を数える → k_bits
2. "1" ビットを読んで捨てる
3. k_bits ビットを読む → lower_bits
4. q = 2^k_bits + lower_bits  （BigUint として構築）
```

### 符号化例（Ratio<BigInt> → ビット列）

| 値 | p | q | 分子（ビット数） | 分母（ビット数） | 合計 |
|---|---|---|---|---|---|
| 0 | 0 | 1 | "1"（1 bit） | "1"（1 bit） | 2 bits |
| 1 | 1 | 1 | "01110"（5 bits） | "1"（1 bit） | 6 bits |
| -1 | -1 | 1 | "0101"（4 bits） | "1"（1 bit） | 5 bits |
| 1/2 | 1 | 2 | "01110"（5 bits） | "010"（3 bits） | 8 bits |
| -1/3 | -1 | 3 | "0101"（4 bits） | "011"（3 bits） | 7 bits |
| 22/7 | 22 | 7 | "00111101100"（11 bits） | "00111"（5 bits） | 16 bits |

22/7 の分子詳細：z=44=101100b、B=6、gamma(7): k=2、"00111"（5 bits）、data="101100"（6 bits）

**整数値（q=1）は分母が常に "1"（1 bit）だけで済む**ため、VarIntArray と比べて 1 bit の追加のみ。

### 1要素のデコード手順（ビット列 → Ratio<BigInt>）

```
1. 分子 p のデコード:
   a. 拡張 Elias gamma のデコード:
      i.  連続する 0 ビットの数を数える → k
      ii. "1" ビットを読んで捨てる
      iii. k ビットを読む → lower_bits
      iv. B+1 = 2^k + lower_bits  → B = 2^k + lower_bits - 1
      v.  B ビットを読む → z (BigUint, MSB ファースト)。B==0 のとき z=0
   b. 逆 Zigzag: z が偶数 → p = z/2、z が奇数 → p = -(z+1)/2  (BigInt)

2. 分母 q のデコード（標準 Elias gamma）:
   a. 連続する 0 ビットの数を数える → k_bits
   b. "1" ビットを読んで捨てる
   c. k_bits ビットを読む → lower_bits
   d. q = 2^k_bits + lower_bits  (BigUint)

3. Ratio::new_raw(p, BigInt::from(q)) を返す
   （格納時に正規化済みのため GCD 再計算不要）
```

### 複数要素のブロック内ビット列の例

`[0, 1/2, -1]` を k=3 のブロックに格納した場合：

```
値    p   q  分子ビット列  分母ビット列  要素のビット列
0     0   1  "1"           "1"           "11"           (2 bits)
1/2   1   2  "01110"       "010"         "01110010"     (8 bits)
-1   -1   1  "0101"        "1"           "01011"        (5 bits)

連結: "11" | "01110010" | "01011"
    = "110111001001011"  (15 bits)

MSB ファーストでバイト列に格納（端数は 0 パディング）：
  data[0] = 0b11011100  (b0〜b7)
  data[1] = 0b10010110  (b8〜b14 + 1bit パディング)

bit_len = 15, count = 3
```

### ビット列のバイト格納（VarIntArray と同一）

```
bit_pos p は data[p/8] のビット (7 - p%8) に対応する（MSB が先）

書き込み: data[p/8] |= (1u8 << (7 - p%8))
読み取り: bit = (data[p/8] >> (7 - p%8)) & 1

末尾追記:
  if bit_len % 8 == 0 { data.push(0u8) }
  // 書き込み式で data[bit_len / 8] に書く
  bit_len += 1
```

---

## データ構造

```
RatioArray {
    k:      usize,           // ブロックサイズ（構築時に指定）
    blocks: Vec<BitBlock>,   // ブロックの配列
    length: usize,           // 総要素数
}

BitBlock {
    data:    Vec<u8>,        // エンコード済みビット列（MSB ファースト）
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

---

## API

### 構築

```rust
// 空の配列を作成
RatioArray::new(k: usize) -> Result<Self, ArrayError>

// Vec<Ratio<BigInt>> から構築
RatioArray::new_with_vec(k: usize, vals: Vec<Ratio<BigInt>>) -> Result<Self, ArrayError>

// イテレータから構築
RatioArray::new_with_iter(k: usize, vals: impl Iterator<Item=Ratio<BigInt>>) -> Result<Self, ArrayError>
```

### 要素アクセス

```rust
get(i: usize) -> Result<Ratio<BigInt>, ArrayError>
// blocks[i/k] の先頭から i%k+1 要素デコードして返す（部分デコード）
// Err: OutOfBounds

set(i: usize, v: Ratio<BigInt>) -> Result<(), ArrayError>
// blocks[i/k] を全デコード → 差し替え → 再エンコード
// Err: OutOfBounds
```

**set の詳細手順：**

```
1. i >= self.length なら Err(OutOfBounds)
2. block_idx = i / k
3. blocks[block_idx] の全要素をデコード → elems: Vec<Ratio<BigInt>>
4. elems[i % k] = v
5. blocks[block_idx] を空の BitBlock に置き換える:
       blocks[block_idx] = BitBlock { data: vec![], bit_len: 0, count: 0 }
6. elems の各要素を push の手順（step 2〜6）で blocks[block_idx] に書き戻す
   （self.length は変化しないので注意。直接 blocks[block_idx] に書き込む）
7. return Ok(())
```

値の変更でビット長が変わるため、再エンコード後は `blocks[block_idx].bit_len` と
`blocks[block_idx].data` の長さが変わる場合がある。

どちらも最悪計算量は O(K)。

### スタック操作

```rust
push(v: Ratio<BigInt>) -> Result<usize, ArrayError>
// 戻り値: 追加された要素のインデックス

pop() -> Result<Ratio<BigInt>, ArrayError>
// Err: Empty
```

**push の詳細手順：**

```
1. blocks が空、または最後のブロックの count == k の場合:
       blocks.push(BitBlock { data: vec![], bit_len: 0, count: 0 })

2. p = v.numer().clone()
   q = BigUint::try_from(v.denom().clone()).unwrap()
       // 分母は常に正なので unwrap は安全

3. 分子 p を拡張 Elias gamma で符号化:
   a. Zigzag: z = (p=0 → 0), (p>0 → 2*p), (p<0 → 2*|p|-1)  as BigUint
   b. B = z.bits() as usize
   c. k = floor(log2(B+1))
      B+1 を標準 Elias gamma で符号化 → bits_to_write に追加
      （k 個の 0 + "1" + B+1 の下位 k ビット MSBファースト）
   d. z の上位 B ビットを MSBファーストで bits_to_write に追加

4. 分母 q を標準 Elias gamma で符号化:
   a. k_bits = q.bits() as usize - 1   // floor(log2(q))
   b. k_bits 個の 0 を bits_to_write に追加
   c. "1" を bits_to_write に追加
   d. q の下位 k_bits ビットを MSBファーストで bits_to_write に追加

5. bits_to_write の各ビットを末尾追記の手順で blocks.last_mut().data に書き込む

6. blocks.last_mut().count += 1
7. self.length += 1
8. return Ok(self.length - 1)
```

**pop の詳細手順：**

```
1. blocks が空なら Err(Empty)
2. 最後のブロックの全要素をデコード → elems: Vec<Ratio<BigInt>>
3. ret = elems.pop()
4. elems が空（count が 1 だった）なら blocks.pop() して self.length -= 1 → return Ok(ret)
5. 最後のブロックを空の BitBlock に置き換える:
       *blocks.last_mut() = BitBlock { data: vec![], bit_len: 0, count: 0 }
6. elems の各要素を push の手順（step 2〜6）で書き戻す
7. self.length -= 1
8. return Ok(ret)
```

### 一括操作

```rust
extend(vals: impl IntoIterator<Item=Ratio<BigInt>>) -> Result<(), ArrayError>
extend_array(other: &RatioArray) -> Result<(), ArrayError>
// other の k と self の k が異なっても動作する。fast path なし。iter() → push() で実装。
```

`extend` / `extend_array` は atomic ではない（任意の `Ratio<BigInt>` を格納可能で値エラーが発生しないため、ロールバックは不要）。

### 統計・イテレーション

```rust
iter(&self) -> RatioIter            // ExactSizeIterator<Item=Ratio<BigInt>>
sum() -> Option<Ratio<BigInt>>      // None if empty
min() -> Option<Ratio<BigInt>>      // None if empty
max() -> Option<Ratio<BigInt>>      // None if empty
average() -> Option<Ratio<BigInt>>  // None if empty（有理数として精度損失なし）
len() -> usize
is_empty() -> bool
```

`average` は `sum() / Ratio::from_integer(BigInt::from(self.length))` で計算する。
浮動小数点と異なり精度損失なく正確な有理数値を返す。

`Ratio<BigInt>` は `Ord` を実装するため `min` / `max` の比較は完全に正確。

`sum` / `min` / `max` / `average` はすべて `iter()` を通じて逐次計算する（O(n)）。

**RatioIter（VarIntArray の VarIntIter と同一の構造）：**

```rust
struct RatioIter<'a> {
    arr: &'a RatioArray,
    block_idx: usize,       // 現在のブロック番号
    elem_in_block: usize,   // ブロック内の要素番号（0-origin）
    bit_pos: usize,         // ブロック内の現在読み取りビット位置
    remaining: usize,       // 残り要素数（ExactSizeIterator 用）
}
```

`next()` の動作：
1. `remaining == 0` なら `None`
2. `elem_in_block == arr.blocks[block_idx].count` なら次ブロックへ
   （`block_idx += 1, elem_in_block = 0, bit_pos = 0`）
3. `bit_pos` から分子・分母を順にデコードし `bit_pos` を進める
4. `elem_in_block += 1, remaining -= 1` して `Ratio::new_raw(p, BigInt::from(q))` を返す

> `Ratio::new_raw` を使うのは、格納時に既約正規化済みであるため再計算不要だから。

### メタデータ

```rust
block_size() -> usize     // = k
block_count() -> usize    // = blocks.len()
datasize() -> usize
// size_of::<RatioArray>()
// + size_of::<BitBlock>() * blocks.capacity()
// + 各ブロックの data.capacity() の合計
```

---

## エラー型

| エラー | 発生箇所 |
|---|---|
| `OutOfBounds` | `get` / `set` のインデックス超過 |
| `Empty` | `pop` を空配列に対して呼んだ |
| `InvalidRange` | `new` で `k == 0` |

---

## 操作の計算量まとめ

| 操作 | 時間計算量 | 備考 |
|---|---|---|
| `get(i)` | O(K) | ブロック内部分デコード |
| `set(i, v)` | O(K) | ブロック全体の再符号化 |
| `push(v)` | O(1) 均し | ブロック満杯時のみ新規作成 |
| `pop()` | O(K) | ブロック全体の再符号化 |
| `extend(n個)` | O(n) | push の繰り返し |
| `iter()` 全走査 | O(n) | |
| `sum` / `min` / `max` / `average` | O(n) | iter() 経由 |

---

## `PackedArrayCore` トレイトとの関係

`Ratio<BigInt>` は `Copy` を実装しないため、`PackedArrayCore` は使用しない。
`push` / `pop` / `extend` は `RatioArray` が独自に実装する。

---

## 表示 (Display)

```
[k=64][4]=0,1/2,-1/3,22/7
  ^    ^
  k    length
```

整数値（q=1）は `Ratio` の `Display` に準拠して "p" と表示する（"p/1" ではない）。

---

## シリアライズ (serde)

- **Serialize**: 各要素を `Ratio` の `Display` でフォーマットした文字列配列として出力。
  `q=1`（整数）のとき `"p"`、それ以外は `"p/q"` となる（num-rational の `Display` 実装による、実測確認済み）。
- **Deserialize**: 文字列配列から `s.parse::<BigRational>()` で再構築する。
  `num-rational` の `FromStr` は `"p/q"` 形式と整数 `"p"` 形式の両方を受け付ける。
  `k` はデフォルト値（`k = 64`）を使用する。

```json
["0", "1/2", "-1/3", "22/7"]
```

> **注意**: デシリアライズで `k` は保存されない。元の `k` と異なる場合がある。
> ただし `PartialEq` / `Eq` は要素値のみで比較する（`k` とブロック構造を無視）ため、
> serde round-trip の前後で `==` は成立する。

---

## Cargo.toml への追加

```toml
[dependencies]
num-rational = { version = "0.4", features = ["serde"] }
# num-bigint は既存の依存として前提（num-rational 0.4 は num-bigint 0.4 を使用）
```

---

## 設計上の決定事項

| 項目 | 決定 | 理由 |
|---|---|---|
| 要素型 | `Ratio<BigInt>` | 任意精度、正規化保証、`num` エコシステムと統一 |
| 分子符号化 | Zigzag + 拡張 Elias gamma | VarIntArray と同一、符号ありゼロを自然に扱える |
| 分母符号化 | 標準 Elias gamma | `q ≥ 1` なので直接適用可。q=1（整数）が 1 bit で最小 |
| average の戻り値 | `Ratio<BigInt>` | 有理数除算は精度損失なし（f64 より正確） |
| デコード時の構築 | `Ratio::new_raw` | 格納時に正規化済みのため GCD 再計算不要 |
| ブロック構造 | VarIntArray と同一 | 実装の再利用・一貫性 |
| serde 表現 | "p/q" 文字列配列 | 任意精度を正確に表現 |
| `PartialEq` / `Eq` | 要素値のみ比較（k・ブロック構造を無視） | serde round-trip 後も `==` が成立するようにするため |
| デシリアライズ時の k | 64（`DEFAULT_K`） | k は外部表現に含まれないため固定値が必要 |
| 共有コード | `src/bits.rs` に抽出 | `VarIntArray` と共通のビット I/O・Zigzag・Elias gamma ヘルパーを重複なく共有 |

---

## 実装上の注意点

### 入力の正規化前提（`Ratio::new_raw` の危険性）

`push` / `set` が受け取る `Ratio<BigInt>` は正規化済み（GCD=1、分母>0）を前提とする。
`Ratio::new` で構築した場合は自動正規化されるが、`Ratio::new_raw` でバイパスすると以下の問題が起きる：

| 不正な入力 | 挙動 |
|---|---|
| `Ratio::new_raw(p, BigInt::from(0))` | `write_denom` 内の `assert!` で panic（release/debug 両方） |
| `Ratio::new_raw(p, BigInt::from(-1))` | `BigUint::try_from` が失敗し `unwrap()` で panic |

`Ratio::new_raw` は "invariants を呼び出し側が保証する" 低レベル API である。
`RatioArray` の公開 API に渡す場合は必ず `Ratio::new` を使うこと。

### 内部不変条件

`BitBlock` のビット列は常に `encode_into` → `decode_one` の組み合わせで読み書きされる。
`read_bit` / `read_elias_gamma` / `read_denom` は不変条件が破れた場合（`data` が短すぎる等）に panic する。
外部から `BitBlock` の `data` を直接書き換えることはできないため、通常の使い方では問題ない。

### 32 ビット環境

`BigUint::bits()` は `u64` を返すが、内部で `usize` にキャストする。
32 ビット環境（`usize = u32`）では、2^32 ビット（約 500 MB の値）を超える分子・分母は
切り詰められて誤ったエンコードになる。実用上は 64 ビット環境のみを想定している。

### k の上限

現行実装では `k == 0` のみ弾く（`Err(InvalidRange)`）。
`k = usize::MAX` のような極端な値を設定すると全要素が 1 ブロックに収まり続け、
`set` / `pop` が O(n) に劣化する。大きな k を使う際は注意。

### 要素サイズの制限なし

任意精度の `BigInt` を格納できるため、数十億ビットの値を push すると
その場で数百 MB を消費する。デシリアライズ経由で外部 JSON を取り込む場合は
特に注意（文字列長に上限を設けることを推奨）。

---

## 疑問点・未解決事項

（現時点ではなし）
