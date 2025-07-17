# Egaroucid Engine コードベース概要

Egaroucidエンジンの各ファイルと関数の概要、呼び出し関係をまとめたドキュメントです。

## 目次
1. [プロジェクト概要](#プロジェクト概要)
2. [ファイル構成](#ファイル構成)
3. [主要な検索アルゴリズム](#主要な検索アルゴリズム)
4. [関数依存関係](#関数依存関係)
5. [パフォーマンス最適化](#パフォーマンス最適化)

## プロジェクト概要

Egaroucidは高性能なオセロ（リバーシ）AIエンジンです。主な特徴：
- Negascout（Principal Variation Search）アルゴリズム
- Null Window Search（NWS）による高速化
- 並列探索（Young Brothers Wait Concept）
- 置換表による枝刈り
- Multi-ProbCut（MPC）による枝刈り
- Enhanced Transposition Cutoff（ETC）
- 評価関数の最適化

## ファイル構成

### 1. 検索関連（Search）

#### midsearch.hpp
**概要**: 中盤探索のメインファイル。Negascoutアルゴリズムの実装。

**主要関数**:
- `int nega_scout(Search *search, int alpha, int beta, int depth, bool skipped, uint64_t legal, bool is_end_search, bool *searching)`
  - **目的**: Negascoutアルゴリズムによる探索
  - **呼び出し元**: `first_nega_scout_legal`, `aspiration_search`, 自己再帰
  - **呼び出し先**: `transposition_cutoff`, `move_list_evaluate`, `ybwc_search_young_brothers`, `nega_alpha_ordering_nws`
  
- `int nega_scout_policy(Search *search, int alpha, int beta, int depth, bool skipped, uint64_t legal, bool is_end_search, bool *searching)`
  - **目的**: 方針探索（最善手の決定）
  - **呼び出し元**: AIのメイン探索ルーチン
  - **呼び出し先**: `move_list_evaluate`, `ybwc_search_young_brothers`

- `std::pair<int, int> first_nega_scout_legal(Search *search, int alpha, int beta, int depth, bool is_end_search, std::vector<Clog_result> clogs, uint64_t legal, uint64_t strt, bool *searching)`
  - **目的**: ルートノードでの探索
  - **呼び出し元**: AIのメインインターフェース
  - **呼び出し先**: `nega_scout`, `nega_alpha_ordering_nws`

- `int aspiration_search(Search *search, int alpha, int beta, int predicted_value, int depth, bool skipped, uint64_t legal, bool is_end_search, bool *searching)`
  - **目的**: Aspiration Search（予測値に基づく探索窓の最適化）
  - **呼び出し元**: `nega_scout`
  - **呼び出し先**: `nega_scout`

#### midsearch_nws.hpp
**概要**: Null Window Searchの実装。

**主要関数**:
- `int nega_alpha_ordering_nws(Search *search, int alpha, int depth, bool skipped, uint64_t legal, bool is_end_search, std::vector<bool*> &searchings)`
  - **目的**: 並列対応のNWS探索
  - **呼び出し元**: `nega_scout`, `first_nega_scout_legal`, 自己再帰
  - **呼び出し先**: `nega_alpha_ordering_nws_simple`, `move_list_evaluate_nws`, `ybwc_search_young_brothers_nws`

- `int nega_alpha_ordering_nws_simple(Search *search, int alpha, int depth, bool skipped, uint64_t legal, bool *searching)`
  - **目的**: シンプルなNWS探索（並列処理なし）
  - **呼び出し元**: `nega_alpha_ordering_nws`
  - **呼び出し先**: `move_list_evaluate_nws`, 自己再帰

- `int nega_alpha_eval1_nws(Search *search, int alpha, bool skipped)`
  - **目的**: 深度1でのNWS探索
  - **呼び出し元**: `nega_alpha_ordering_nws_simple`
  - **呼び出し先**: `mid_evaluate_diff`

### 2. 移動評価・順序付け（Move Ordering）

#### move_ordering.hpp
**概要**: 手の評価と順序付けの最適化。

**主要関数**:
- `bool move_list_evaluate(Search *search, std::vector<Flip_value> &move_list, uint_fast8_t moves[], int depth, int alpha, int beta, bool *searching)`
  - **目的**: 中盤での手の評価
  - **呼び出し元**: `nega_scout`, `first_nega_scout_legal`
  - **呼び出し先**: `move_evaluate`

- `bool move_list_evaluate(Search *search, Flip_value move_list[], int canput, uint_fast8_t moves[], int depth, int alpha, int beta, bool *searching)` [配列版オーバーロード]
  - **目的**: 配列版の手の評価
  - **呼び出し元**: `nega_scout`, `first_nega_scout_legal`
  - **呼び出し先**: `move_evaluate`

- `bool move_list_evaluate_nws(Search *search, std::vector<Flip_value> &move_list, uint_fast8_t moves[], int depth, int alpha, bool *searching)`
  - **目的**: NWSでの手の評価
  - **呼び出し元**: `nega_alpha_ordering_nws`
  - **呼び出し先**: `move_evaluate_nws`

- `bool move_list_evaluate_nws(Search *search, Flip_value move_list[], int canput, uint_fast8_t moves[], int depth, int alpha, bool *searching)` [配列版オーバーロード]
  - **目的**: 配列版のNWS手の評価
  - **呼び出し元**: `nega_alpha_ordering_nws_simple`
  - **呼び出し先**: `move_evaluate_nws`

- `void swap_next_best_move(std::vector<Flip_value> &move_list, int strt, int siz)`
- `void swap_next_best_move(Flip_value move_list[], int strt, int siz)` [配列版オーバーロード]
  - **目的**: 最善手を先頭に移動
  - **呼び出し元**: すべての探索関数
  - **呼び出し先**: なし

- `void move_list_sort(std::vector<Flip_value> &move_list)`
- `void move_list_sort(Flip_value move_list[], int canput)` [配列版オーバーロード]
  - **目的**: 手リストの並び替え
  - **呼び出し元**: 並列探索前の準備
  - **呼び出し先**: `std::sort`

### 3. 並列探索（Parallel Search）

#### ybwc.hpp
**概要**: Young Brothers Wait Conceptによる並列探索。

**主要関数**:
- `void ybwc_search_young_brothers(Search *search, int *alpha, int *beta, int *v, int *best_move, int n_available_moves, uint32_t hash_code, int depth, bool is_end_search, std::vector<Flip_value> &move_list, int canput, bool need_best_move, bool *searching)`
  - **目的**: 並列でのNegascout探索
  - **呼び出し元**: `nega_scout`, `first_nega_scout_legal`
  - **呼び出し先**: `nega_scout`, `nega_alpha_ordering_nws`

- `void ybwc_search_young_brothers_nws(Search *search, int alpha, int *v, int *best_move, int n_available_moves, uint32_t hash_code, int depth, bool is_end_search, std::vector<Flip_value> &move_list, int canput, std::vector<bool*> &searchings)`
  - **目的**: 並列でのNWS探索
  - **呼び出し元**: `nega_alpha_ordering_nws`
  - **呼び出し先**: `nega_alpha_ordering_nws`

### 4. 置換表・枝刈り（Transposition & Pruning）

#### transposition_cutoff.hpp
**概要**: Enhanced Transposition Cutoffによる枝刈り。

**主要関数**:
- `bool etc_nws(Search *search, std::vector<Flip_value> &move_list, int depth, int alpha, int *v, int *n_etc_done)`
  - **目的**: ETCによる枝刈り
  - **呼び出し元**: `nega_alpha_ordering_nws_simple`, `nega_alpha_ordering_nws`
  - **呼び出し先**: `transposition_table`関連

#### transposition_table.hpp
**概要**: 置換表の実装。

#### multi_probcut.hpp
**概要**: Multi-ProbCutによる枝刈り。

### 5. 評価関数（Evaluation）

#### evaluate.hpp, evaluate_*.hpp
**概要**: 位置評価関数の実装。

#### stability.hpp
**概要**: 石の安定性評価。

### 6. 基本構造（Basic Structures）

#### search.hpp
**概要**: 探索の基本構造とSearchクラスの定義。

**主要クラス**:
- `Search`
  - **目的**: 探索状態の管理
  - **メンバー**: `board`, `eval`, `n_nodes`, `mpc_level`, `use_multi_thread`等
  - **メソッド**: `move()`, `undo()`, `pass()`等

#### board.hpp
**概要**: オセロ盤面の実装。

#### common.hpp
**概要**: 共通定数・ユーティリティ関数。

## 主要な検索アルゴリズム

### 1. Negascout (Principal Variation Search)
```
nega_scout() -> move_list_evaluate() -> nega_alpha_ordering_nws() -> nega_scout()
```

### 2. Null Window Search
```
nega_alpha_ordering_nws() -> move_list_evaluate_nws() -> move_evaluate_nws()
```

### 3. 並列探索フロー
```
nega_scout() -> ybwc_search_young_brothers() -> [並列実行] -> nega_scout()
```

## 関数依存関係

### トップレベル（AIインターフェース）
- `first_nega_scout_legal()` - ルートノードでの探索

### 中盤探索
- `nega_scout()` - メインのNegascout実装
- `nega_alpha_ordering_nws()` - NWS実装
- `aspiration_search()` - Aspiration Search

### 移動評価
- `move_list_evaluate()` - 手の評価
- `move_evaluate()` - 個別手の評価
- `swap_next_best_move()` - 手の並び替え

### 並列処理
- `ybwc_search_young_brothers()` - 並列探索制御

### 枝刈り・最適化
- `transposition_cutoff()` - 置換表カットオフ
- `etc_nws()` - Enhanced Transposition Cutoff
- `mpc()` - Multi-ProbCut

## パフォーマンス最適化

### 1. データ構造の最適化
- **従来**: `std::vector<Flip_value>` - 動的メモリ割り当て
- **最適化**: `Flip_value[35]` - 固定配列（スタック上）
- **効果**: メモリ割り当てオーバーヘッドの削減

### 2. 関数オーバーロード戦略
多くの関数に配列版オーバーロードを提供：
- `move_list_evaluate(vector版)` / `move_list_evaluate(配列版)`
- `move_list_evaluate_nws(vector版)` / `move_list_evaluate_nws(配列版)`
- `swap_next_best_move(vector版)` / `swap_next_best_move(配列版)`
- `move_list_sort(vector版)` / `move_list_sort(配列版)`

### 3. ハイブリッドアプローチ
- **基本**: 配列を使用
- **互換性**: 配列のみをサポートしない関数にはベクター変換を使用
- **例**: `etc_nws()`, `ybwc_search_young_brothers_nws()`は現在ベクターのみ対応

### 4. 最適化の段階的導入
1. **第1段階**: `nega_alpha_ordering_nws_simple()` - 完全配列版
2. **第2段階**: `nega_alpha_ordering_nws()` - ハイブリッド版
3. **第3段階**: より多くの関数に配列版オーバーロードを追加予定

## 注意事項

### 現在の制限
- 一部の関数（`etc_nws`, `ybwc_search_young_brothers_nws`）はベクター版のみ対応
- これらの関数には配列↔ベクター変換を使用（一時的なパフォーマンス低下）

### 今後の改善点
1. 全ての主要関数に配列版オーバーロードを追加
2. 配列↔ベクター変換の削除
3. より深い最適化（SIMD、キャッシュ最適化等）

---

*最終更新: 2025年7月17日*
*作成者: GitHub Copilot*
