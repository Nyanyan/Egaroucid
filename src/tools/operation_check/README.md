# 動作自動確認スクリプト（Issue #652）

リリース前や機能追加後に、GUI版・コンソール版の動作を確認するためのスクリプトです。
コンソール版とGUIの基礎確認にはPython 3.9以降と標準ライブラリを使います。
実GUI操作の記録にはCodexのcomputer-use環境（`@oai/sky`）、画像ファイルの検証にはPillowを使います。

`console.py`はコンソールの入出力を検証します。`gui.py`は内部処理と画面描画の基礎確認です。
マウス、実キー入力、IME、クリップボード、ネットワーク、Book操作などのリリース確認は、
`desktop.py`と実GUI操作の結果で別に判定します。**GUIの基礎確認15件だけでは、リリース確認を合格にしません。**

先にアプリケーションをビルドし、確認したい実行ファイルを指定してください。
スクリプトは指定された実行ファイルを検証します。ソースの自動ビルドやダウンロードは行いません。
任意の作業フォルダから実行できますが、以下の例はリポジトリのルートで実行することを想定しています。

## 実行方法

```powershell
$env:PYTHONUTF8 = '1'
# コンソール版：ビルド済みの実行ファイルとresources/を用意して実行
python src/tools/operation_check/console.py --exe bin/Egaroucid_for_Console.exe

# GUI版：公式のVisual Studio用スクリプトでPortable版をビルドして実行
python src/tools/release_script/build_gui.py -c SIMD_Portable
python src/tools/operation_check/gui.py --exe bin/Egaroucid.exe

# アプリケーションを起動せずに、確認ケースの一覧を表示
python src/tools/operation_check/console.py --list
python src/tools/operation_check/gui.py --list

# 変更した機能に関係するケースを実行（--caseは複数指定可能）
python src/tools/operation_check/console.py --case threads --case ponder
python src/tools/operation_check/gui.py --case settings --case annotations

# 配布用の翻訳ファイルとリリーステンプレートの一致も確認
python src/tools/operation_check/gui.py --case resources --templates src/tools/release_script/format_files/0_common_files/languages
```

必要に応じて、次のオプションを指定できます。

| オプション | 用途 |
| --- | --- |
| `--resources PATH` | 配布リソースが別の場所にある場合に、そのフォルダを指定 |
| `--output-dir PATH` | 結果やログの保存先を指定 |
| `--timeout SECONDS` | 子プロセスごとの制限時間を秒で指定（既定値は30秒） |
| `--fail-fast` | 最初の失敗で実行を終了 |

GUI版の全ケースの実行には、開発用PCで約1分かかります。その間、確認用のアプリケーションが複数回起動・終了します。
時間のかかる確認は、進捗を見られるコンソールで実行してください。
実行中のケース、全体の件数と現在の実行位置、経過時間、推定残り時間が表示されます。
Ctrl+Cで中断すると、実行中の子アプリケーションも停止します。

GUI版の確認には、操作可能なデスクトップと正常に動作するグラフィックスドライバーが必要です。
`gui.py`自身はマウス・キーボード操作を行いません。実操作の実行方法は後述します。

### 終了コードと保存結果

| 終了コード | 意味 |
| --- | --- |
| **0** | 選択した全ケースに合格 |
| **1** | 1件以上のケースが不合格 |
| **2** | 引数が不正、または実行ファイル・リソースフォルダが存在しない |

ケース名の誤字もエラーになります。また、`python -O`で実行した場合も失敗の判定は有効です。

実行するたびに、`build/operation_check/{console,gui}/`の下へ日時付きのフォルダを作成します。
その中の`results.json`には、実行ファイルとリソースのパス、各ケースの結果・所要時間・保存先、選択した全ケースの実行が完了したかを記録します。
`--fail-fast`を指定していなければ、失敗後も残りのケースを実行します。

各ケースのフォルダには、入力、コマンド引数、標準出力、標準エラー出力、例外の詳細を保存します。
アプリケーションの終了コードが0でも、タイムアウトや必要な出力の欠落があれば不合格になります。
中断した場合も、それまでに完了したケースの結果は残ります。`complete: false`は、未実行のケースが残っていることを表します。

## コンソール版の確認範囲

| ケース | 確認内容 |
| --- | --- |
| `metadata` | バージョン表示、起動オプションのヘルプ、対話コマンドのヘルプ |
| `argument-error` | 必須の起動引数を省略した場合にエラーになること |
| `perft` | パスの数え方が異なる両モードで、深さ5の末端局面数が1,396になること |
| `history` | 大文字・小文字を混ぜた着手、棋譜出力、戻す・進む、分岐、初期化 |
| `illegal-moves` | 不正な座標、石のあるマスや非合法なマスへの着手、途中に不正な手を含む棋譜の取り消し |
| `search` | 2局面での静音モードの探索結果、初期局面の4手すべてのヒント |
| `endgame` | 1マス空き局面の最善手と完全読みの評価値（+64）、setboard/new、明示的なパス |
| `threads`, `ponder` | スレッド数の変更、不正な指定の拒否、先読みの有無による探索と終了 |
| `gtp` | 応答番号と区切り、エンジン情報、known_command、合法手、手番、着手、戻す、初期化、genmove、不正着手の拒否 |
| `solve-file` | 問題ファイルからの終盤局面の読込と完全読み |
| `selfplay` | 自己対戦の全棋譜を、独立した配列ベースのルール実装で再生できること |
| `book` | 配布ブックから局面を読み込み、初手の選択にブックを使うこと |

探索結果は合法手かどうかを確認し、終盤では完全読みの評価値も比較します。
中盤の評価値や、同じ評価の手からどれを選ぶかには依存しません。
既定では1スレッド、ハッシュレベル18を使います。

配布物にhash18の初期化用データがない場合、エンジンは代わりにデータを生成します。
この処理に伴う特定の2種類のメッセージは許容し、元のログにはそのまま残します。
それ以外のエンジンエラーは、通常のケースでは不合格になります。
一方、不正入力を試すケースでは、想定したエラーが出たかを個別に確認します。

## GUI版の基礎確認（gui.py）

GUI版には、明示的に指定した場合だけ動く`--operation-check CASE_DIRECTORY`モードを用意しています。
このモードは`gui.py`から呼び出してください。専用の識別文字列と毎回異なるトークンで、確認用の実行を識別します。
このオプションを指定しなければ、アプリケーションは通常どおり起動・動作します。

GUI版の確認には、SIMD_Portable、Generic_Portable、AVX512_Portableのいずれかを使ってください。
インストール版は、**個人設定を読み込む前に**確認モードの起動を拒否します。
また、古い実行ファイルは確認用オプションを無視して個人設定を読み込む可能性があります。
そのため、Python側でも実行ファイル内の識別文字列を調べ、対応していない実行ファイルを起動前に拒否します。
リリース用ビルドスクリプトで生成した、実行ファイル圧縮などを施していないビルドを使ってください。

確認処理は、通常の読込とメイン画面の作成が終わるのを待ってから、製品と共通の保存・取込・プロファイル・キャッシュ・探索の各関数を呼び出します。
終了時も通常のCloseシーンを通ります。
GUI側の結果には、今回の要求と同じトークンとケース名が必要です。さらに、正常終了時のショートカット設定ファイルが作成されたことも確認します。
GUI側の結果とログは、各ケースのフォルダ内の`native-result.json`、`native-XX.json`、`native.log`に保存します。

| ケース | 確認内容 |
| --- | --- |
| `resources` | 必須の配布ファイル、4言語のJSONと画像編集画面のラベル、指定時はテンプレートとの一致 |
| `startup` | リソースとフォント、初期盤面と合法手、画面が描画されること、正常終了 |
| `settings-defaults` | 盤面画像の設定項目がない旧設定ファイルの読込 |
| `settings-invalid` | 盤面画像の設定値の型や範囲が不正な場合の処理 |
| `settings` | 製品と共通の保存処理による画像編集・表示設定の保存と再読込、その保存結果からの再起動 |
| `transcript` | 棋譜形式の自動認識、不正棋譜の拒否、盤面文字列の入出力、独立したルール実装との比較 |
| `profiles` | AI・表示プロファイルの保存と読込、多言語の名前 |
| `game-files` | 対局JSON・CSVの保存と読込、棋譜、多言語の対局情報、複数行のメモ |
| `annotations` | 局面ごとの注釈の保持と削除、進行の初期化、矢印と重複の拒否、全消去、マス単位の座標判定 |
| `forced-openings` | 登録した強制定石から合法な継続手が得られること |
| `search` | GUIで初期化したエンジンの探索結果が合法手であること |
| `scenes-english`, `scenes-japanese`, `scenes-simplified-chinese`, `scenes-traditional-chinese-taiwan` | 各言語で9画面の遷移と描画。メイン、文字列取込、盤面編集、対局情報、出力、盤面画像、定石設定、ショートカット設定、棋譜管理が対象 |

画面描画のケースでは、600×400ピクセル以上の画像が得られ、単色またはほぼ単色になっていないことを確認します。
各画面は`frame-SCENE.png`として保存されるので、文字の欠けや重なり、デザインはその画像を見て確認してください。
自動判定の対象は描画の基本動作であり、画像のピクセル単位の一致までは比較しません。

## GUI版の実操作によるリリース確認（desktop.py）

ボタンのクリック、実キー入力、Microsoft IMEでの変換・確定、フォーカス復帰、ドラッグ＆ドロップ、
クリップボード、パス入力、ネットワークサービス、Book操作、Egaroucid Windows配布インストーラーを確認対象にします。
全30項目の一覧と必要な判定条件は`desktop.py`の`CASES`にあります。
未実行の項目、途中で止まった項目、失敗した項目が一つでもあれば、リリース確認の終了コードは1です。

現在のEgaroucidは、ファイルパスの入力欄とドラッグ＆ドロップを使います。
OS標準のファイル選択ダイアログを呼び出す実装はないため、読込・保存先の入力、取消、不正ファイルの拒否を確認します。

### 隔離したGUIの起動

```powershell
python src/tools/operation_check/desktop.py launch --exe bin/Egaroucid.exe
```

表示された`desktop`フォルダには、`run.json`、`request.json`、棋譜サンプル、正常・不正なBookを用意します。
GUIはそのフォルダのデータだけを使います。通常のシーンやボタンを操作するのはOS入力ドライバーです。
GUI側のdesktopモードは、現在のシーン・局面・IME変換中文字列・処理の実行状態などを`desktop-state.json`へ記録するだけです。
このモードからシーンを切り替えたり、着手やテストの期待値を設定したりはしません。

別アプリへのフォーカス移動と、OSのクリップボードの内容を確認するため、次の補助ウィンドウも使えます。
`RUN_DIRECTORY`を、起動時に表示された`desktop`フォルダの絶対パスに置き換えてください。

```powershell
python src/tools/operation_check/desktop_fixture.py RUN_DIRECTORY
```

### OS入力と結果の記録

`desktop_driver.mjs`は、computer-use環境で選択した実ウィンドウに一操作ずつ入力を送り、直後の画面を取得します。
座標操作の前に表示を確認してください。ウィンドウの識別子やスクリーンショットの識別子は使い回しません。
次の例の`RUN_DIRECTORY`とモジュールのパスも、実際の絶対パスに置き換えます。

```javascript
const {sky} = await import('@oai/sky');
const {DesktopDriver} = await import('file:///D:/github/othello/Egaroucid/src/tools/operation_check/desktop_driver.mjs');
const {beginScenario, nextScenarioStep} = await import('file:///D:/github/othello/Egaroucid/src/tools/operation_check/desktop_scenarios.mjs');
const driver = await DesktopDriver.open(sky, 'RUN_DIRECTORY');
const candidates = (await sky.list_windows()).filter(w => w.app.includes('今回のapplicationフォルダのパス'));
if (candidates.length !== 1) throw new Error('検証用GUIを一意に選択できません');
await driver.select(candidates[0]);
beginScenario(driver, 'mouse-gameplay');
```

画面を確認した後、次の呼出しで一操作を実行します。取得された画面を確認してから、次の操作へ進みます。

```javascript
await nextScenarioStep(driver);
```

用意済みのシナリオは、盤面クリック、Book統合、Book学習の開始・停止・再開、Book修復、Edax準拠修復、Book削減、リーフ再計算、ライン数再計算、現在の棋譜のBook登録です。
準備時には、ドラッグ確認用の定石2件、棋譜2件、移動先フォルダも作成します。
定石は`document/forced_openings/`、棋譜は`document/games/release_test/`にあります。
並べ替え後はCSVの行順を、フォルダ間移動後は移動先のCSV・棋譜JSONと移動元からの除去を確認してください。
日本語UI、表示倍率1.0、803×530または803×531ピクセルのウィンドウが前提です。
Book処理が完了画面を表示する場合は、処理の完了を状態ファイルでも確認してから「戻る」を押します。
完了していなければ入力を止めるため、処理中の「強制停止」を「戻る」と取り違えません。
シナリオがまだない項目も`driver.act(...)`で実操作と結果を記録できます。
学習の停止テストには、未学習の小規模Book、深さ15、レベル19を使います。
レベル1や拡張済みのBookでは停止前に完了するため、実行中から停止できたかを確認できません。

```javascript
driver.begin('keyboard-shortcuts');
await driver.act({type: 'key', key: 'Left'});
await driver.expect('before-caps', {discs: 4, history_size: 2});
```

この例では、着手済みの5石・履歴2局面から開始します。必要な全条件を確認した後に`driver.complete()`を呼びます。
実操作を伴わない内部テストの結果や、必要な判定条件が欠けた記録は、実GUI操作の合格として受け付けません。
各ケースの操作・判定・時刻は`CASE.result.json`に保存します。
Windowsでは、実操作確認モードに限り、マウスの受信メッセージと各フレームの入力状態を`desktop-mouse.tsv`にも保存します。
`message`行はフレーム番号・時刻・シーン・Windowsメッセージ番号・クライアント座標・ボタン状態、
`frame`行はフレーム番号・時刻・シーン・Siv3Dのシーン座標・左ボタンのdown/pressed/upです。
クライアント座標とシーン座標は、DPIや描画倍率によって異なります。
この記録は入力を変更せず、通常起動では有効になりません。

現段階では、全30項目をPythonだけで無人実行できる状態にはなっていません。
OS入力の記録と結果集計、上記の再利用可能な操作シナリオを実装しています。
未実装・未実施の操作を、基礎確認の合格で代用しないでください。

### 保存ファイルと集計

```powershell
# 保存したBookのヘッダー、局面数、レコード長、盤面の重複などを検証
python src/tools/operation_check/book_probe.py RUN_DIRECTORY/document/book_copy.egbk3 --count 90 --report RUN_DIRECTORY/book-saved.json

# 統合前のBookから期待する局面集合を計算し、欠落・余分な局面がないか比較
python src/tools/operation_check/book_probe.py MERGED_EGBK3 --expected-union INPUT_DAT EXTRA_EGBK3 --report RUN_DIRECTORY/book-merged.json

# 学習前の局面をすべて保持し、新しい局面が追加されたか比較
python src/tools/operation_check/book_probe.py LEARNED_EGBK3 --contains ORIGINAL_EGBK3 --min-added 1 --report RUN_DIRECTORY/book-learned.json

# 指定棋譜の各局面が保存され、学習前より局面が増えたことを検証
python src/tools/operation_check/book_probe.py LEARNED_EGBK3 --contains ORIGINAL_EGBK3 --transcript c4c3c2b2 --min-added 1 --report RUN_DIRECTORY/book-transcript.json

# 保存したPNGの大きさ、指定ピクセル、透過、クリップボードとの一致を検証
python src/tools/operation_check/image_probe.py SAVED_PNG --size 420 420 --alpha 0,0,0 --clipboard-json RUN_DIRECTORY/fixture-image.json --report RUN_DIRECTORY/image-saved.json

# 未実行も含めて全項目を集計（未確認の項目があれば終了コード1）
python src/tools/operation_check/desktop.py report RUN_DIRECTORY
```

保存ファイルの検証結果も`driver.expectJsonFile(...)`で操作記録に結び付けます。
Book処理ではメイン画面に戻れることに加え、保存後の局面数やリーフ、ライン数を確認してください。
画像では保存ボタンを押したことに加え、マークと矢印の画素、消去後の画素、透過、OSのクリップボードから取得したPNGのハッシュを比較します。

Egaroucid Windows配布インストーラーの新規導入・更新・削除は、検証用Windows環境で実行します。
Portable版のテスト結果では、この3項目を合格にしません。

GUIで設定を変更した後の再起動には、元のGUIが終了していることを確認して、次を使います。
実行ファイルのハッシュが準備時と変わっていれば、再起動を拒否します。

```powershell
python src/tools/operation_check/desktop.py resume RUN_DIRECTORY
```

## 個人設定との分離とケースの追加

実行時には、指定した実行ファイル、同じフォルダのDLL、存在する場合はLICENSE、配布リソースを確認用フォルダへコピーします。
元の`appdata/`や`document/`はコピーしません。
GUIの各ケースには専用の設定、プロファイル、対局フォルダ、空のブックを用意し、その実行中は自動更新確認を無効にします。

WindowsのGUI版は作業フォルダを基準にリソースを読むため、各ケースのフォルダから確認用リソースのコピーへハードリンクを作成します。
ハードリンクを使えない場合は、ファイルをコピーします。生成したファイルはすべて出力フォルダ内に保存します。
コンソール版のブック確認でも、確認用にコピーした配布ブックを使います。
元の設定や個人のブックには書き込みません。確認結果は実行後も残るので、不要になった実行フォルダは削除してください。

ケースを追加するには、`Context`を受け取る関数を書き、`console.py`または`gui.py`の一覧へ`Case(name, description, function)`を登録します。
ケース名を維持すると、機能変更後も同じ名前で対象を絞って確認できます。
条件の判定には`require(condition, message)`を使い、子プロセスの起動にはログと制限時間を扱う`ctx.run(...)`を使ってください。

GUI本体側の確認処理を追加する場合は、`Gui_probe::check_case()`に分岐を追加し、Python側にもケースを登録してからGUIを再ビルドします。
確認コードでは、製品と共通の処理を呼び出し、その結果や保存ファイルを検証してください。

### 確認スクリプト自身のテスト

実行処理と、独立したルール実装には短時間で終わるテストを用意しています。GUI用SDKは不要です。

```powershell
python -m unittest discover -s src/tools/operation_check -p "test_*.py" -v
node --test src/tools/operation_check/test_desktop_scenarios.mjs
```

失敗後の継続、最初の失敗での停止と未完了の記録、不正なケース名や制限時間、タイムアウト時のログ保存、基本局面でのルール判定を確認します。
また、古い・失敗した・互換性のないGUI応答や、確認モードに対応していない実行ファイルを拒否できることも確認します。

## 検証記録

2026年10月29日にWindows 11で確認しました。**リリース判定は未合格です。**
コンソール版13件、最新GUIの基礎確認15件、Pythonの自己テスト22件、操作シナリオの自己テスト5件は合格しました。
実GUI操作の定義済み30項目のうち、オセロクエストの検証1項目はユーザー指定により今回スキップします。
今回の対象29項目中22項目を、修正段階の異なるビルドを通じて確認しました。
スキップを合格には数えません。同じビルドで対象29項目すべてに合格した結果ではありません。

実操作で見つかった入力不具合、確認できた保存結果、ドラッグなどの未合格項目は、
[2026年10月29日の検証記録](validation-20261029.md)に記載しています。
ログ、操作記録、ファイル検証結果は`build/issue652/`に保存しています。このフォルダはGitの管理対象外です。
