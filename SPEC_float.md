# FloatArray 仕様

## 概要

`FloatArray` は、仮数部ビット数 `man_bits` と指数部ビット数 `exp_bits` を実行時に指定できる、
カスタム精度浮動小数点数のパック配列である。

各要素は `1 + exp_bits + man_bits` ビットを消費し、`Vec<u64>` に詰め込んで格納する。
ユーザーへの公開型は `f64`（get/set のインターフェイス）。

既存の `IntArray`（N-bit 符号なし整数配列）および `RadixArray`（任意範囲符号あり整数配列）と
同一の `PackedArrayCore` トレイトを実装し、push/pop/extend の共通ロジックを共有する。

---

## ビットフォーマット

要素は IEEE 754 ライクな固定フォーマットで格納する：

```
bit[total-1]                          : sign (1 bit)
bit[man_bits .. man_bits+exp_bits-1]  : exponent (exp_bits bits, biased) ※両端を含む
bit[0 .. man_bits-1]                  : mantissa (man_bits bits) ※両端を含む
```

数値例（FLOAT16: exp_bits=5, man_bits=10, total=16）：
```
bit15      : sign
bit14..10  : exponent (5 bits)
bit9..0    : mantissa (10 bits)
```

- `total = 1 + exp_bits + man_bits`
- `bias = 2^(exp_bits-1) - 1`
- `exp_max = 2^exp_bits - 1`（全ビット1 = Inf/NaN を示す予約値）

これは IEEE 754 のエンコーディング規則に従い、標準フォーマット（float16, bfloat16, float32, float64）と
ビットレベルで互換性がある。

---

## 標準フォーマット定数

```rust
pub const FLOAT16:  (usize, usize) = (5,  10);  // IEEE 754 半精度
pub const BFLOAT16: (usize, usize) = (8,   7);  // Brain Float 16
pub const FLOAT32:  (usize, usize) = (8,  23);  // IEEE 754 単精度
pub const FLOAT64:  (usize, usize) = (11, 52);  // IEEE 754 倍精度
```

---

## パッキング方式

`IntArray` と同じビット詰め方式（詳細は `SPEC_intarray.md` §パッキング方式を参照）。
`bits` を `total = 1 + exp_bits + man_bits` に置き換えたものが FloatArray のパッキング。

```
bpu = 64 / total          // elements per u64 word（小数切り捨て）
word_count = ceil(len / bpu)

word_index = i / bpu
bit_offset = (i % bpu) * total
mask = (1u64 << total) - 1  // total == 64 のとき u64::MAX

get:  (data[word_index] >> bit_offset) & mask
set:  data[word_index] = (data[word_index] & !(mask << bit_offset))
                        | (encoded << bit_offset)
```

| フォーマット | total | bpu |
|---|---|---|
| FLOAT16 / BFLOAT16 | 16 | 4 |
| FLOAT32 | 32 | 2 |
| FLOAT64 | 64 | 1 |
| カスタム 8bit (2e+5m) | 8 | 8 | ※ exp_bits=2 だと表現範囲が [1.0, 4.0) のみ。実験的用途向け |

---

## 構成パラメータの制約

`new(exp_bits, man_bits, len)` は以下を検証し、違反時に `Err(ArrayError::InvalidRange)` を返す：

| 条件 | 理由 |
|---|---|
| `exp_bits >= 2` | exp=0 をゼロ、exp=all1 を Inf/NaN に予約するため最低 2 bit 必要 |
| `exp_bits <= 11` | インターフェイスが f64 のため、f64 の指数部幅 (11 bit) を超えても範囲は広がらない |
| `man_bits >= 1` | 仮数部が 0 bit だと精度が完全に失われる |
| `man_bits <= 52` | インターフェイスが f64 のため、f64 の仮数部幅 (52 bit) を超えても精度は向上しない |
| `1 + exp_bits + man_bits <= 64` | u64 に収まること（前 2 条件から自動的に満たされるが、意図を明示するため検証する） |

> **精度の上限**: インターフェイスが `f64` である以上、実効精度の上限は FLOAT64（11e + 52m）。
> `exp_bits > 11` または `man_bits > 52` は意味のないビット消費になるためエラーとする。

---

## エンコーディング規則（f64 → カスタムフォーマット）

1. `v.to_bits()` から `sign`, `f64_exp`, `f64_man` を取り出す。
2. **NaN / Inf** (`f64_exp == 0x7FF`):
   - `custom_exp = exp_max`（全ビット1）
   - Inf: `custom_man = 0`
   - NaN: `custom_man = 1`（ペイロードは正規化せず最小 NaN）
3. **ゼロ / 非正規化数** (`f64_exp == 0`):
   - Flush-to-zero: 符号ビットのみ保持、指数・仮数は 0
4. **通常値**:
   - `custom_exp = f64_exp - 1023 + bias`
   - `custom_exp >= exp_max` → オーバーフロー → ±Inf として格納（エラーにしない）
   - `custom_exp <= 0` → アンダーフロー → ±0 として格納（エラーにしない）
   - 仮数の切り詰め（truncate）: `f64_man >> (52 - man_bits)`（man_bits ≤ 52 の場合）

## デコーディング規則（カスタムフォーマット → f64）

1. `raw` から `sign`, `custom_exp`, `custom_man` を取り出す。
2. `custom_exp == exp_max` → Inf（`custom_man == 0`）または NaN（`custom_man != 0`）。
3. `custom_exp == 0` → ±0.0（flush-to-zero で非正規化数は存在しない）。
4. 通常値:
   - `f64_exp = custom_exp - bias + 1023`（u64 にキャスト）
   - 仮数の拡張（ゼロ埋め）: `f64_man = custom_man << (52 - man_bits)`
   - `f64::from_bits((sign << 63) | (f64_exp << 52) | f64_man)`

---

## API

### 構築

```rust
// 全要素を +0.0 で初期化（全ビット0 → 符号=0、指数=0、仮数=0）
FloatArray::new(exp_bits, man_bits, len) -> Result<Self, ArrayError>

// Vec<f64> から構築
FloatArray::new_with_vec(exp_bits, man_bits, vals: Vec<f64>) -> Result<Self, ArrayError>

// イテレータから構築
FloatArray::new_with_iter(exp_bits, man_bits, vals: impl Iterator<Item=f64>) -> Result<Self, ArrayError>
```

### 要素アクセス

```rust
get(i: usize) -> Result<f64, ArrayError>   // Err(OutOfBounds) のみ
set(i: usize, v: f64) -> Result<(), ArrayError>  // Err(OutOfBounds) のみ
```

値の範囲エラーは発生しない（オーバーフロー→Inf、アンダーフロー→0 として格納）。

### スタック操作

```rust
push(v: f64) -> Result<usize, ArrayError>   // 戻り値は新要素のインデックス
pop() -> Result<f64, ArrayError>            // Err(Empty)
```

push は値のエラーで失敗しない（任意の f64 を格納可能）。

### 一括操作

```rust
extend(vals: impl IntoIterator<Item=f64>) -> Result<(), ArrayError>
extend_array(other: &FloatArray) -> Result<(), ArrayError>
```

`extend` は値エラーで失敗しないため、常に `Ok(())` を返す（任意の f64 を格納可能なため）。
ただし実装は IntArray/RadixArray と同じ `PackedArrayCore::core_extend` を使用し、
仮に将来エラーが発生した場合のロールバック機構も共有している。

`extend_array` はフォーマットが一致し、`self` が word-aligned（`length % bpu == 0`）の場合に
raw word コピーの fast path を使用する。それ以外は要素ごとに decode(f64) → encode を経由する slow path。

### 統計

```rust
iter() -> FloatIter           // ExactSizeIterator<Item=f64>; インデックス 0 から順に返す
sum() -> Option<f64>          // None if empty
min() -> Option<f64>          // None if empty; NaN があれば他の値を優先
max() -> Option<f64>          // 同上
average() -> Option<f64>      // None if empty
```

min/max は `f64::min` / `f64::max` を使い、NaN があれば非 NaN 側を返す（NaN は「値なし」扱い）。
ただし全要素が NaN の場合は NaN を返す。

`sum()` / `average()` は NaN を含む場合 NaN を返す（f64 加算の伝播による）。

### メタデータ

```rust
len() -> usize
is_empty() -> bool
exp_bits() -> usize
man_bits() -> usize
capacity() -> usize     // data.len() * bpu
datasize() -> usize     // 構造体 + Vec データのバイト数
```

---

## エラー型

`FloatArray` が返す `ArrayError` の種類：

| エラー | 発生箇所 |
|---|---|
| `OutOfBounds` | `get` / `set` のインデックス超過 |
| `Empty` | `pop` を空配列に対して呼んだ |
| `InvalidRange` | `new` でパラメータが制約違反 |

`TooLarge` / `TooSmall` は `FloatArray` では発生しない（サイレント変換）。

---

## 表示 (Display)

```
[e5m10][4]=1,2,3,4
  ^  ^  ^
  |  |  length
  |  man_bits
  exp_bits
```

---

## シリアライズ (serde)

- **Serialize**: 要素を f64 のフラット配列として出力。フォーマット情報は含まない。
- **Deserialize**: フラット配列から FLOAT64（11e+52m）として再構築。

> **注意**: シリアライズ・デシリアライズで `exp_bits`/`man_bits` は保存されない。
> 精度を変えて格納していた場合、デシリアライズ後は FLOAT64 精度になる。

---

## 設計上の決定事項

| 項目 | 決定 | 理由 |
|---|---|---|
| 非正規化数 | Flush-to-zero | 実装シンプル、ML/センサーデータ用途では十分 |
| NaN | NaN として保存・復元するが、ペイロードは正規化（`custom_man=1` に統一） | ペイロード保存は複雑でユースケースがほぼない |
| Inf | ±Inf として保存・復元 | 数値計算の正確性のため |
| 仮数の丸め | 切り捨て (truncate) | 高速、実装シンプル |
| オーバーフロー | ±Inf に変換（エラーなし） | float らしい振る舞い |
| アンダーフロー | ±0 に変換（エラーなし） | flush-to-zero と一致 |
| ユーザー型 | `f64` | Rust の標準浮動小数点型 |

---

## 疑問点・未解決事項

（現時点ではなし）
