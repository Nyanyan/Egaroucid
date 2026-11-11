"""実GUIテストで使うフォーカス移動・クリップボード確認用ウィンドウ。"""
import argparse
import ctypes
from ctypes import wintypes
import hashlib
import json
from pathlib import Path
import tkinter as tk


def clipboard_image():
    user32, kernel32, shell32 = ctypes.windll.user32, ctypes.windll.kernel32, ctypes.windll.shell32
    user32.RegisterClipboardFormatW.argtypes = [wintypes.LPCWSTR]
    user32.RegisterClipboardFormatW.restype = wintypes.UINT
    user32.GetClipboardData.argtypes = [wintypes.UINT]
    user32.GetClipboardData.restype = wintypes.HANDLE
    shell32.DragQueryFileW.argtypes = [wintypes.HANDLE, wintypes.UINT, wintypes.LPWSTR, wintypes.UINT]
    shell32.DragQueryFileW.restype = wintypes.UINT
    kernel32.GlobalLock.argtypes = [wintypes.HGLOBAL]
    kernel32.GlobalLock.restype = ctypes.c_void_p
    kernel32.GlobalUnlock.argtypes = [wintypes.HGLOBAL]
    if not user32.OpenClipboard(None):
        raise RuntimeError("OpenClipboard failed")
    result = {"files": []}
    try:
        handle = user32.GetClipboardData(15)  # CF_HDROP
        if handle:
            for index in range(shell32.DragQueryFileW(handle, 0xffffffff, None, 0)):
                length = shell32.DragQueryFileW(handle, index, None, 0)
                name = ctypes.create_unicode_buffer(length + 1)
                shell32.DragQueryFileW(handle, index, name, length + 1)
                path = Path(name.value)
                result["files"].append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        effect = user32.GetClipboardData(user32.RegisterClipboardFormatW("Preferred DropEffect"))
        pointer = kernel32.GlobalLock(effect) if effect else None
        if pointer:
            try:
                result["copy_effect"] = ctypes.cast(pointer, ctypes.POINTER(wintypes.DWORD)).contents.value == 1
            finally:
                kernel32.GlobalUnlock(effect)
        result["has_image_data"] = any(user32.IsClipboardFormatAvailable(format_id) for format_id in
            (2, 8, 17, user32.RegisterClipboardFormatW("PNG"), user32.RegisterClipboardFormatW("image/svg+xml")))
    finally:
        user32.CloseClipboard()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    window = tk.Tk()
    window.title("Egaroucid release test fixture")
    window.geometry("560x440+950+120")
    tk.Label(window, text="Egaroucid release test fixture", font=("Arial", 14)).pack(pady=10)
    text = tk.Text(window, width=52, height=5)
    text.insert("1.0", "f5d6c3d3")
    text.pack(padx=10, pady=5)
    status = tk.StringVar(value="Ready")
    def copy_text():
        window.clipboard_clear()
        window.clipboard_append(text.get("1.0", "end-1c"))
        status.set("Copied text")
    def record_text():
        value = text.get("1.0", "end-1c")
        (args.directory / "fixture-text.json").write_text(json.dumps({"text": value}), encoding="utf-8")
        status.set("Recorded text")
    def record_image():
        try:
            image = clipboard_image()
            (args.directory / "fixture-image.json").write_text(json.dumps(image), encoding="utf-8")
            status.set("Image file recorded" if image["files"] else "No file on clipboard")
        except Exception as error:
            status.set(str(error))
    tk.Button(window, text="Copy text", command=copy_text).pack(pady=3)
    tk.Button(window, text="Record text", command=record_text).pack(pady=3)
    tk.Button(window, text="Read clipboard file", command=record_image).pack(pady=3)
    presets = tk.Frame(window)
    presets.pack(pady=8)
    def copy_preset(value):
        text.delete("1.0", "end")
        text.insert("1.0", value)
        copy_text()
    for index, (label, value) in enumerate((
        ("Copy valid book path", "./sample-book.egbk3"),
        ("Copy invalid book path", "./invalid-book.egbk3"),
        ("Copy merge book path", "./extra-book.egbk3"),
        ("Copy Edax export path", "./document/book_copy.dat"),
        ("Copy transcript", "f5d6c3d3"),
    )):
        tk.Button(presets, text=label, command=lambda value=value: copy_preset(value)).grid(
            row=index // 2, column=index % 2, padx=4, pady=3)
    tk.Label(window, textvariable=status).pack(pady=3)
    window.mainloop()
