"""
複数の構成で Egaroucid GUI版 をビルドし、リネーム・移動するスクリプト
"""

import os
import sys
import shutil
from pathlib import Path
from build_gui import build_project, find_msbuild, PROJECT_ROOT

# バージョン情報
VERSION = "7_8_0"

# ビルド構成
CONFIGURATIONS = [
    ["SIMD", "2_GUI_Installer_exes"],
    ["SIMD_AMD", "2_GUI_Installer_exes"],
    ["Generic", "2_GUI_Installer_exes"], 
    ["AVX512", "2_GUI_Installer_exes"],
    ["AVX512_AMD", "2_GUI_Installer_exes"],
    ["SIMD_Portable", "3_GUI_Portable_exes"],
    ["SIMD_Portable_AMD", "3_GUI_Portable_exes"],
    ["Generic_Portable", "3_GUI_Portable_exes"],
    ["AVX512_Portable", "3_GUI_Portable_exes"],
    ["AVX512_Portable_AMD", "3_GUI_Portable_exes"],
]

# 出力先ディレクトリ
SCRIPT_DIR = Path(__file__).parent


def ensure_output_dir(output_dir):
    """出力ディレクトリを作成"""
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"出力ディレクトリ: {output_dir}")


def get_build_output_path(configuration, platform="x64"):
    """ビルド出力ファイルのパスを取得"""
    # すべての構成で bin\ ディレクトリに出力される
    output_dir = PROJECT_ROOT / "bin"
    return output_dir / "Egaroucid.exe"


def rename_and_move(configuration, output_dir, platform="x64"):
    """ビルド成果物をリネームして移動"""
    source_path = get_build_output_path(configuration, platform)
    
    if not source_path.exists():
        print(f"✗ エラー: ビルド成果物が見つかりません: {source_path}")
        return False
    
    # 新しいファイル名を生成
    new_filename = f"Egaroucid_{VERSION}_{configuration}.exe"
    dest_path = output_dir / new_filename
    
    # ファイルをコピー
    try:
        shutil.copy2(source_path, dest_path)
        print(f"✓ 移動完了: {new_filename}")
        print(f"  サイズ: {dest_path.stat().st_size:,} bytes")
        return True
    except Exception as e:
        print(f"✗ エラー: ファイルの移動に失敗: {e}")
        return False


def build_all():
    """すべての構成でビルド"""
    # MSBuild の存在確認
    msbuild_path = find_msbuild()
    if not msbuild_path:
        print("エラー: MSBuild.exe が見つかりません。")
        print("Visual Studio がインストールされているか確認してください。")
        return False
    
    print(f"MSBuild: {msbuild_path}")
    print(f"バージョン: {VERSION}")
    print(f"ビルド構成: {', '.join([c[0] for c in CONFIGURATIONS])}")
    print("=" * 60)
    
    # 各構成でビルド
    success_count = 0
    failed_configs = []
    
    for i, config in enumerate(CONFIGURATIONS, 1):
        configuration_name = config[0]
        output_dir_name = config[1]
        output_dir = SCRIPT_DIR / "format_files" / output_dir_name
        
        print(f"\n[{i}/{len(CONFIGURATIONS)}] {configuration_name} のビルド開始")
        print(f"出力先: {output_dir}")
        print("-" * 60)
        
        # 出力ディレクトリを作成
        ensure_output_dir(output_dir)
        
        # ビルド実行
        success = build_project(configuration=configuration_name, platform="x64", clean=False, verbose=False)
        
        if success:
            # リネームして移動
            if rename_and_move(configuration_name, output_dir, platform="x64"):
                success_count += 1
            else:
                failed_configs.append(f"{configuration_name} (移動失敗)")
        else:
            failed_configs.append(f"{configuration_name} (ビルド失敗)")
        
        print("-" * 60)
    
    # 結果サマリー
    print("\n" + "=" * 60)
    print("ビルド完了")
    print("=" * 60)
    print(f"成功: {success_count}/{len(CONFIGURATIONS)}")
    
    if failed_configs:
        print(f"失敗: {len(failed_configs)}")
        for config in failed_configs:
            print(f"  - {config}")
        return False
    else:
        print("\nすべてのビルドが成功しました！")
        print("\n生成されたファイル:")
        for config in CONFIGURATIONS:
            configuration_name = config[0]
            output_dir_name = config[1]
            output_dir = SCRIPT_DIR / "format_files" / output_dir_name
            filename = f"Egaroucid_{VERSION}_{configuration_name}.exe"
            filepath = output_dir / filename
            if filepath.exists():
                print(f"  - {filename} ({filepath.stat().st_size:,} bytes) -> {output_dir}")
        return True


def main():
    print("Egaroucid GUI版 一括ビルドスクリプト")
    print("=" * 60)
    
    success = build_all()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
