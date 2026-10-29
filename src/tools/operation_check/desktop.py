"""Egaroucidの実GUI操作テストの準備と、リリース判定用の結果集計。"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import uuid

from common import Context, ROOT, require
from gui import fixture
from book_fixture import create_book


# Internal-function smoke tests cannot satisfy these cases. Each desktop case
# needs actual OS input events and named, machine-checked postconditions.
CASES = {
    "mouse-gameplay": ("盤面クリック・非合法手・戻す・進む", ["legal-move", "illegal-move", "back", "forward"]),
    "keyboard-shortcuts": ("キー操作・Caps Lock・フォーカス復帰", ["before-caps", "after-caps", "after-focus"]),
    "ime": ("IMEで日本語の変換・確定・画面切替後の再入力", ["composition", "committed", "reentered"]),
    "image-editor": ("画像編集の選択・矢印・消去・保存ボタン", ["selected", "saved-image", "cleared"]),
    "clipboard-text": ("GUIのコピー操作と貼り付け取込", ["copied", "pasted"]),
    "clipboard-image": ("画像の保存とクリップボードの画像内容", ["png-saved", "clipboard-matches"]),
    "drop-file": ("OSからの棋譜ファイルのドラッグ＆ドロップ", ["dropped", "imported"]),
    "drag-opening": ("定石の並べ替えとフォルダ間移動", ["reordered", "moved"]),
    "drag-game": ("棋譜の並べ替えとフォルダ間移動", ["reordered", "moved"]),
    "file-path-open": ("GUIのファイルパス入力・取消・不正ファイル", ["opened", "cancelled", "invalid-rejected"]),
    "file-path-save": ("GUIの保存先パス入力・保存・取消", ["saved", "cancelled"]),
    "profiles-ui": ("GUIからAI・表示プロファイルを保存・復元", ["ai-saved", "ai-restored", "display-saved", "display-restored"]),
    "settings-restart-ui": ("GUIで変更した設定の再起動後の復元", ["changed", "restored"]),
    "network-update": ("GUIからの実サーバーへの更新確認", ["requested", "response"]),
    "network-quest": ("オセロクエスト棋譜の一覧・詳細・取込・エラー", ["list", "detail", "imported", "error"]),
    "book-import": ("GUIからブック読込・不正ファイルの拒否", ["imported", "invalid-rejected"]),
    "book-export": ("GUIからegbk3・Edaxブックの出力と再読込", ["egbk3", "edax", "reloaded"]),
    "book-merge": ("GUIからブック結合", ["completed", "saved"]),
    "book-reference": ("GUIからブック参照", ["completed"]),
    "book-learn": ("GUIからブック拡張の開始・停止・再開・保存", ["started", "stopped", "resumed", "saved"]),
    "book-learn-transcript": ("GUIから棋譜に沿ったブック拡張", ["completed", "saved"]),
    "book-store": ("GUIからブックへの評価値追加", ["completed", "saved"]),
    "book-fix": ("GUIからブック修復", ["completed", "saved"]),
    "book-fix-edax": ("GUIからEdax準拠のブック修復", ["completed", "saved"]),
    "book-reduce": ("GUIからブック縮小", ["completed", "saved"]),
    "book-leaves": ("GUIから葉の再計算", ["completed", "saved"]),
    "book-lines": ("GUIから定石数の再計算", ["completed", "saved"]),
    "installer-new": ("Egaroucid Windows配布インストーラーで新規導入", ["installed", "launched", "payload"]),
    "installer-upgrade": ("Egaroucid Windows配布インストーラーで更新", ["upgraded", "settings-preserved", "launched"]),
    "installer-uninstall": ("Egaroucid Windows配布版のアンインストール", ["uninstalled", "user-data-preserved"]),
}


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def prepare(args):
    directory = args.output_dir.absolute() / datetime.now().strftime("%Y%m%d-%H%M%S-%f") / "desktop"
    directory.mkdir(parents=True)
    args.exe = args.exe.absolute()
    args.resources = (args.resources or args.exe.parent / "resources").absolute()
    binary = args.exe.read_bytes()
    marker = "egaroucid-desktop-check-v1"
    require(any(marker.encode(encoding) in binary for encoding in ("ascii", "utf-16-le", "utf-32-le")),
            "読み取り専用の実操作テスト記録に対応したGUI Portableをビルドしてください")
    context = Context(args, directory)
    fixture(context, language=args.language, settings={"book_learn_depth": 5, "use_book_learn_depth": True})
    token = str(uuid.uuid4())
    write_json(directory / "request.json", {"marker": "egaroucid-operation-check-v1", "case": "desktop", "token": token})
    (directory / "sample-transcript.txt").write_text("f5d6c3d3\n", encoding="utf-8")
    (directory / "invalid-book.egbk3").write_bytes(b"not a valid book")
    create_book(directory / "sample-book.egbk3")
    write_json(directory / "run.json", {
        "protocol": "egaroucid-desktop-check-v1", "token": token,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "executable": str(context.application()), "executable_sha256": hashlib.sha256(binary).hexdigest(),
        "directory": str(directory), "cases": {
            name: {"description": description, "required": assertions, "status": "not_run"}
            for name, (description, assertions) in CASES.items()
        },
    })
    print(directory)
    print("実行ファイル:", context.application())
    print("起動引数: --operation-check", directory)
    return directory


def launch(args):
    directory = prepare(args)
    run_file = directory / "run.json"
    run = json.loads(run_file.read_text(encoding="utf-8"))
    import os
    environment = dict(os.environ)
    environment["EGAROUCID_SHORTCUT_DIAG_LOG"] = str(directory / "shortcut-input.log")
    process = subprocess.Popen([run["executable"], "--operation-check", str(directory)], cwd=directory, env=environment)
    run["process_id"] = process.pid
    write_json(run_file, run)
    print("Process ID:", process.pid)


def validate_case(case, evidence):
    require(evidence.get("status") == "passed", "実操作テストが未完了または不合格です")
    actions = evidence.get("actions", [])
    require(actions and all(action.get("driver") == "os-input" and action.get("completed") is True for action in actions), "完了したOS入力の記録がありません")
    assertions = {item["name"]: item for item in evidence.get("assertions", [])}
    missing = set(case["required"]) - set(assertions)
    require(not missing, "未確認の項目: " + ", ".join(sorted(missing)))
    require(all(item.get("passed") is True for item in assertions.values()), "失敗した確認項目があります")
    require(all(item.get("expected") for item in assertions.values()), "判定条件のない確認項目があります")


def report(directory):
    run = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    results = []
    for name, case in run["cases"].items():
        result = {"name": name, "description": case["description"], "status": "not_run"}
        evidence_file = directory / (name + ".result.json")
        if evidence_file.is_file():
            try:
                evidence = json.loads(evidence_file.read_text(encoding="utf-8"))
                require(evidence.get("token") == run["token"] and evidence.get("case") == name, "古い実行結果または別ケースの結果です")
                validate_case(case, evidence)
                result["status"] = "passed"
            except Exception as error:
                result.update(status="failed", error=str(error))
        results.append(result)
        print(f"{result['status']:8} {name}: {case['description']}")
    passed = all(item["status"] == "passed" for item in results)
    write_json(directory / "release-results.json", {"kind": "desktop-release", "passed": passed, "cases": results})
    print("リリース実操作テスト:", "全件合格" if passed else "未合格（未実行も合格には含めません）")
    return 0 if passed else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "launch"):
        prep = sub.add_parser(command)
        prep.add_argument("--exe", type=Path, default=ROOT / "bin/Egaroucid.exe")
        prep.add_argument("--resources", type=Path)
        prep.add_argument("--output-dir", type=Path, default=ROOT / "build/operation_check/desktop")
        prep.add_argument("--language", default="japanese")
    check = sub.add_parser("report")
    check.add_argument("directory", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "launch":
        launch(args)
    else:
        raise SystemExit(report(args.directory))
