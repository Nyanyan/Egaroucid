"""Run the real empty-folder check with Siv3D and a read-denied Windows CSV.

Requires MSVC and SIV3D_0_6_16IME (the GUI project's configured SDK).
--ref checks the older helper against the same regression without changing files.
Fixtures and build outputs remain below ignored/pr643-fixes/folders.
"""
import argparse
import os
import pathlib
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", help="Git revision instead of the working tree")
    args = parser.parse_args()
    directory = pathlib.Path(__file__).absolute().parent
    root = directory.parents[2]
    output = root / "ignored" / "pr643-fixes" / "folders"
    output.mkdir(parents=True, exist_ok=True)
    path = "src/gui/function/util.hpp"
    source = (subprocess.check_output(["git", "show", args.ref + ":" + path], cwd=root).decode("utf-8-sig")
              if args.ref else (root / path).read_text(encoding="utf-8-sig"))
    start = source.index("inline bool is_list_folder_empty(")
    end = source.index("\n}", start) + 2
    (output / "folder_empty_function.hpp").write_text(source[start:end], encoding="utf-8")
    sdk = pathlib.Path(os.environ["SIV3D_0_6_16IME"])
    vswhere = pathlib.Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)")) / "Microsoft Visual Studio/Installer/vswhere.exe"
    vs = pathlib.Path(subprocess.check_output([str(vswhere), "-latest", "-products", "*", "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath"], text=True).strip())
    vcvars = vs / "VC/Auxiliary/Build/vcvars64.bat"
    exe = output / "folder_empty_regression.exe"
    command = ["cl", "/nologo", "/std:c++latest", "/Zc:__cplusplus", "/EHsc", "/O1", "/MT", "/utf-8",
               "/D_ENABLE_EXTENDED_ALIGNED_STORAGE", "/D_SILENCE_CXX20_CISO646_REMOVED_WARNING",
               "/I" + str(output), "/I" + str(sdk / "include"), "/I" + str(sdk / "include/ThirdParty"),
               str(directory / "folder_empty_regression.cpp"), "/Fo" + str(output / "folder_empty_regression.obj"),
               "/Fe" + str(exe), "/link", "/LIBPATH:" + str(sdk / "lib/Windows"), "/SUBSYSTEM:CONSOLE",
               "advapi32.lib", "crypt32.lib", "dwmapi.lib", "gdi32.lib", "imm32.lib", "ole32.lib",
               "oleaut32.lib", "opengl32.lib", "shell32.lib", "shlwapi.lib", "user32.lib", "winmm.lib",
               "ws2_32.lib", "msimg32.lib"]
    script = output / "build.cmd"
    script.write_text('@echo off\ncall "' + str(vcvars) + '" >nul\n' + subprocess.list2cmdline(command) + '\nexit /b %ERRORLEVEL%\n')
    with (output / "build.log").open("w") as log:
        built = subprocess.run(["cmd.exe", "/d", "/c", str(script)], cwd=output, stdout=log, stderr=subprocess.STDOUT)
    if built.returncode:
        print((output / "build.log").read_text(errors="replace"))
        raise SystemExit(built.returncode)
    fixture = tempfile.mkdtemp(prefix="fixtures-", dir=output)
    raise SystemExit(subprocess.run([str(exe), fixture], timeout=15).returncode)


if __name__ == "__main__":
    main()
