"""
Visual Studio を使用して Egaroucid_for_Console.cpp をビルドするスクリプト
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

# プロジェクトルートディレクトリ
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent

# プロジェクトファイル
SOLUTION_FILE = PROJECT_ROOT / "Egaroucid_for_Console.sln"
VCXPROJ_FILE = PROJECT_ROOT / "Egaroucid_for_Console.vcxproj"

# Visual Studio のデフォルトインストールパス
VS_PATHS = [
    r"C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe",
    r"C:\Program Files\Microsoft Visual Studio\2022\Professional\MSBuild\Current\Bin\MSBuild.exe",
    r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\MSBuild\Current\Bin\MSBuild.exe",
    r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\MSBuild\Current\Bin\MSBuild.exe",
    r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\MSBuild\Current\Bin\MSBuild.exe",
    r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise\MSBuild\Current\Bin\MSBuild.exe",
]


def find_msbuild():
    """MSBuild.exe のパスを検索"""
    # 環境変数から検索
    msbuild_path = os.environ.get("MSBUILD_PATH")
    if msbuild_path and os.path.exists(msbuild_path):
        return msbuild_path
    
    # デフォルトパスから検索
    for path in VS_PATHS:
        if os.path.exists(path):
            return path
    
    # vswhere を使用して検索
    try:
        vswhere_path = r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
        if os.path.exists(vswhere_path):
            result = subprocess.run(
                [vswhere_path, "-latest", "-requires", "Microsoft.Component.MSBuild",
                 "-find", r"MSBuild\**\Bin\MSBuild.exe"],
                capture_output=True,
                text=True,
                check=True
            )
            paths = result.stdout.strip().split('\n')
            if paths and paths[0]:
                return paths[0]
    except Exception as e:
        print(f"vswhere での検索に失敗: {e}")
    
    return None


def build_project(configuration="Release", platform="x64", clean=False, verbose=False):
    """プロジェクトをビルド"""
    # MSBuild を検索
    msbuild_path = find_msbuild()
    if not msbuild_path:
        print("エラー: MSBuild.exe が見つかりません。")
        print("Visual Studio がインストールされているか確認してください。")
        print("または、環境変数 MSBUILD_PATH に MSBuild.exe のパスを設定してください。")
        return False
    
    print(f"MSBuild: {msbuild_path}")
    print(f"プロジェクト: {SOLUTION_FILE}")
    print(f"構成: {configuration}|{platform}")
    
    # ビルドコマンドを構築
    cmd = [
        msbuild_path,
        str(SOLUTION_FILE),
        f"/p:Configuration={configuration}",
        f"/p:Platform={platform}",
        "/m",  # 並列ビルド
    ]
    
    if clean:
        cmd.append("/t:Clean,Build")
    else:
        cmd.append("/t:Build")
    
    if verbose:
        cmd.append("/v:detailed")
    else:
        cmd.append("/v:minimal")
    
    # ビルド実行
    print("\nビルド開始...")
    print(f"コマンド: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=False)
        
        if result.returncode == 0:
            print("\n✓ ビルド成功")
            
            # 出力ファイルの確認
            if platform == "x64":
                output_dir = PROJECT_ROOT / "x64" / configuration
            else:
                output_dir = PROJECT_ROOT / configuration
            
            output_file = output_dir / "Egaroucid_for_Console.exe"
            if output_file.exists():
                print(f"出力: {output_file}")
                print(f"サイズ: {output_file.stat().st_size:,} bytes")
            
            return True
        else:
            print(f"\n✗ ビルド失敗 (終了コード: {result.returncode})")
            return False
    
    except Exception as e:
        print(f"\nエラー: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Visual Studio を使用して Egaroucid_for_Console をビルド"
    )
    
    parser.add_argument(
        "-c", "--configuration",
        choices=["Debug", "Release", "SIMD", "SIMD_GGS", "AVX512", "Generic"],
        default="Release",
        help="ビルド構成 (デフォルト: Release)"
    )
    
    parser.add_argument(
        "-p", "--platform",
        choices=["Win32", "x64"],
        default="x64",
        help="プラットフォーム (デフォルト: x64)"
    )
    
    parser.add_argument(
        "--clean",
        action="store_true",
        help="クリーンビルドを実行"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="詳細なビルドログを出力"
    )
    
    args = parser.parse_args()
    
    # プロジェクトファイルの存在確認
    if not SOLUTION_FILE.exists():
        print(f"エラー: ソリューションファイルが見つかりません: {SOLUTION_FILE}")
        return 1
    
    # ビルド実行
    success = build_project(
        configuration=args.configuration,
        platform=args.platform,
        clean=args.clean,
        verbose=args.verbose
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
