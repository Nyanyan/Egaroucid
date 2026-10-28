// Issue #641: test the production observer/matcher with deterministic input.
// Compile through test_shortcut_key_state.py, which supplies the generated header.
#define NOMINMAX
#include <windows.h>
#include <algorithm>
#include <functional>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using String = std::u32string;
struct Shortcut_key_elem { String name; std::vector<String> keys; };
String windows_shortcut_key_token(const String& key) { return key; }

bool shortcut_key_name_to_windows_vk(const String& key, int* vk) {
    if (key.size() == 1 && U'A' <= key[0] && key[0] <= U'Z') {
        *vk = static_cast<int>(key[0]);
        return true;
    }
    const std::unordered_map<String, int> names = {
        {U"Ctrl", VK_CONTROL}, {U"Shift", VK_SHIFT}, {U"Left", VK_LEFT},
        {U"CapsLock", VK_CAPITAL}, {U"Enter", VK_RETURN},
    };
    const auto found = names.find(key);
    if (found == names.end()) return false;
    *vk = found->second;
    return true;
}

std::unordered_set<int> async_pressed;
SHORT test_async_key_state(int vk) {
    return async_pressed.count(vk) ? static_cast<SHORT>(0x8000) : 0;
}
UINT test_map_virtual_key(UINT scan, UINT) {
    switch (scan) {
    case 0x2A: return VK_LSHIFT;
    case 0x36: return VK_RSHIFT;
    case 0x1D: return VK_LCONTROL;
    case 0xE01D: return VK_RCONTROL;
    case 0x1E: return 'A';
    case 0x3A: return VK_CAPITAL;
    default: return 0;
    }
}
#define GetAsyncKeyState test_async_key_state
#define MapVirtualKeyW test_map_virtual_key
#include "shortcut_key_state_under_test.hpp"
#undef GetAsyncKeyState
#undef MapVirtualKeyW

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void observe(UINT message, int vk = 0, int scan = 0, bool extended = false, bool repeated = false) {
    LPARAM data = 1 | (static_cast<LPARAM>(scan) << 16);
    if (extended) data |= static_cast<LPARAM>(1ULL << 24);
    if (repeated) data |= static_cast<LPARAM>(1ULL << 30);
    if (message == WM_KEYUP || message == WM_SYSKEYUP) data |= static_cast<LPARAM>(3ULL << 30);
    windows_shortcut_observe_message(nullptr, message, vk, data);
}

Windows_shortcut_match_details match(std::initializer_list<String> keys) {
    const auto& message = windows_shortcut_message_state();
    Windows_shortcut_keyboard_state snapshot;
    snapshot.focused = message.focused;
    snapshot.pressed_vks = message.pressed_vks;
    snapshot.down_vks = message.pending_down_vks;
    Windows_shortcut_match_details result;
    const Shortcut_key_elem shortcut{U"test", keys};
    require(get_windows_shortcut_match_details(shortcut, snapshot, &result), "unsupported test key");
    return result;
}

void expect_normal_shortcuts() {
    observe(WM_KEYDOWN, 'A', 0x1E);
    require(match({U"A"}).down_found, "A blocked by a stale key");
    observe(WM_KEYUP, 'A', 0x1E);
    observe(WM_KEYDOWN, VK_CONTROL, 0x1D);
    observe(WM_KEYDOWN, 'V', 0x2F);
    require(match({U"Ctrl", U"V"}).down_found, "Ctrl+V blocked by a stale key");
    observe(WM_KEYUP, 'V', 0x2F);
    observe(WM_KEYUP, VK_CONTROL, 0x1D);
    observe(WM_KEYDOWN, VK_LEFT, 0x4B, true);
    require(match({U"Left"}).down_found, "Left arrow blocked by a stale key");
    observe(WM_KEYUP, VK_LEFT, 0x4B, true);
}

int main() {
    int failures = 0;
    int cases = 0;
    auto test = [&](const char* name, const std::function<void()>& body) {
        ++cases;
        async_pressed.clear();
        observe(WM_SETFOCUS);
        try { body(); std::cout << "PASS: " << name << '\n'; }
        catch (const std::exception& error) { ++failures; std::cerr << "FAIL: " << name << ": " << error.what() << '\n'; }
    };

    test("Caps down / IME alphanumeric up", [] {
        observe(WM_KEYDOWN, VK_CAPITAL, 0x3A);
        observe(WM_KEYUP, 0xF0, 0x3A);
        expect_normal_shortcuts();
    });
    test("IME alphanumeric down / Caps up", [] {
        observe(WM_KEYDOWN, 0xF0, 0x3A);
        observe(WM_KEYUP, VK_CAPITAL, 0x3A);
        expect_normal_shortcuts();
    });
    test("IME key identity changes during repeat", [] {
        observe(WM_KEYDOWN, VK_CAPITAL, 0x3A);
        windows_shortcut_message_state().pending_down_vks.clear();
        observe(WM_KEYDOWN, 0xF0, 0x3A, false, true);
        require(match({U"CapsLock"}).pressed_found, "repeat changed the original Caps identity");
        require(!match({U"CapsLock"}).down_found, "repeat generated another down edge");
        observe(WM_KEYUP, 0xF0, 0x3A);
        expect_normal_shortcuts();
    });
    test("Shift and Caps with IME/system messages", [] {
        observe(WM_SYSKEYDOWN, VK_SHIFT, 0x2A);
        observe(WM_SYSKEYDOWN, VK_CAPITAL, 0x3A);
        require(match({U"Shift", U"CapsLock"}).down_found, "Shift+Caps assignment failed");
        observe(WM_SYSKEYUP, 0xF0, 0x3A);
        observe(WM_SYSKEYUP, VK_SHIFT, 0x2A);
        expect_normal_shortcuts();
    });
    test("Caps Lock remains assignable", [] {
        observe(WM_KEYDOWN, VK_CAPITAL, 0x3A);
        require(match({U"CapsLock"}).down_found, "Caps Lock assignment was ignored");
        observe(WM_KEYUP, VK_CAPITAL, 0x3A);
        expect_normal_shortcuts();
    });
    test("extended/non-extended keys sharing a VK", [] {
        observe(WM_KEYDOWN, VK_RETURN, 0x1C);
        observe(WM_KEYDOWN, VK_RETURN, 0x1C, true);
        observe(WM_KEYUP, VK_RETURN, 0x1C);
        require(match({U"Enter"}).pressed_found, "main Enter released held numpad Enter");
        observe(WM_KEYUP, VK_RETURN, 0x1C, true);
        require(!match({U"Enter"}).pressed_found, "numpad Enter remained held");
    });
    test("left/right modifier aliases remain independent", [] {
        observe(WM_KEYDOWN, VK_CONTROL, 0x1D);
        observe(WM_KEYDOWN, VK_CONTROL, 0x1D, true);
        observe(WM_KEYUP, VK_CONTROL, 0x1D);
        observe(WM_KEYDOWN, 'V', 0x2F);
        require(match({U"Ctrl", U"V"}).down_found, "left Ctrl release cleared right Ctrl");
        observe(WM_KEYUP, VK_CONTROL, 0x1D, true);
        require(match({U"Ctrl", U"V"}).modifier_mismatch, "Ctrl alias survived both releases");
    });
    test("scan-less injected keys still work", [] {
        observe(WM_KEYDOWN, 'A');
        require(match({U"A"}).down_found, "scan-less A failed");
        observe(WM_KEYUP, 'A');
        require(!match({U"A"}).pressed_found, "scan-less release failed");
    });
    test("VK_PROCESSKEY resolves to the physical letter", [] {
        observe(WM_KEYDOWN, VK_PROCESSKEY, 0x1E);
        require(match({U"A"}).down_found, "IME letter did not resolve");
        observe(WM_KEYUP, 'A', 0x1E);
        require(!match({U"A"}).pressed_found, "IME letter did not release");
    });
    test("quick tap retains its pending down edge", [] {
        observe(WM_KEYDOWN, 'A', 0x1E);
        observe(WM_KEYUP, 'A', 0x1E);
        require(windows_shortcut_message_state().pending_down_vks.count('A') != 0, "quick tap down edge lost");
    });
    test("focus changes clear both key and scan state", [] {
        for (UINT message : {WM_KILLFOCUS, WM_ACTIVATE, WM_ACTIVATEAPP, WM_NCDESTROY}) {
            observe(WM_KEYDOWN, VK_CAPITAL, 0x3A);
            observe(message);
            require(windows_shortcut_message_state().pressed_vks.empty(), "focus reset left held keys");
            require(windows_shortcut_message_state().pending_down_vks.empty(), "focus reset left pending edges");
            observe(WM_SETFOCUS);
            observe(WM_KEYDOWN, 0xF0, 0x3A);
            require(windows_shortcut_message_state().pressed_vks.count(0xF0) != 0, "focus reset left old scan identity");
            observe(WM_KEYUP, 0xF0, 0x3A);
        }
        expect_normal_shortcuts();
    });
    test("ordinary extra keys still block exact matching", [] {
        observe(WM_KEYDOWN, 'B', 0x30);
        observe(WM_KEYDOWN, 'A', 0x1E);
        require(!match({U"A"}).down_found, "extra B was ignored");
    });
    test("missing Caps key-up recovers without a focus change", [] {
        observe(WM_KEYDOWN, VK_CAPITAL, 0x3A);
        auto& state = windows_shortcut_message_state();
        state.pending_down_vks.clear(); // The first frame consumed the down.
        windows_shortcut_reconcile_message_keys_unlocked(state);
        expect_normal_shortcuts();
        observe(WM_KEYDOWN, 0xF0, 0x3A);
        require(state.pressed_vks.count(0xF0) != 0, "stale scan survived reconciliation");
        observe(WM_KEYUP, VK_CAPITAL, 0x3A);
    });
    test("held Caps remains pressed during reconciliation", [] {
        observe(WM_KEYDOWN, VK_CAPITAL, 0x3A);
        auto& state = windows_shortcut_message_state();
        state.pending_down_vks.clear();
        async_pressed.insert(VK_CAPITAL);
        windows_shortcut_reconcile_message_keys_unlocked(state);
        require(match({U"CapsLock"}).pressed_found, "held Caps was discarded");
    });
    test("reconciliation preserves a new short down for its first frame", [] {
        observe(WM_KEYDOWN, 'A', 0x1E);
        auto& state = windows_shortcut_message_state();
        windows_shortcut_reconcile_message_keys_unlocked(state);
        require(match({U"A"}).down_found, "new down was discarded before its first frame");
        state.pending_down_vks.clear();
        windows_shortcut_reconcile_message_keys_unlocked(state);
        require(!match({U"A"}).pressed_found, "released key was not removed on the next frame");
    });
    test("completed short tap keeps its pending edge during reconciliation", [] {
        observe(WM_KEYDOWN, 'A', 0x1E);
        observe(WM_KEYUP, 'A', 0x1E);
        auto& state = windows_shortcut_message_state();
        windows_shortcut_reconcile_message_keys_unlocked(state);
        require(state.pending_down_vks.count('A') != 0, "completed tap edge was discarded");
    });
    test("reconciliation retains held right Ctrl and its generic alias", [] {
        observe(WM_KEYDOWN, VK_CONTROL, 0x1D);
        observe(WM_KEYDOWN, VK_CONTROL, 0x1D, true);
        auto& state = windows_shortcut_message_state();
        state.pending_down_vks.clear();
        async_pressed = {VK_CONTROL, VK_RCONTROL};
        windows_shortcut_reconcile_message_keys_unlocked(state);
        require(state.pressed_vks.count(VK_LCONTROL) == 0, "released left Ctrl survived");
        observe(WM_KEYDOWN, 'V', 0x2F);
        require(match({U"Ctrl", U"V"}).down_found, "held right Ctrl alias was cleared");
    });
    test("unmapped IME fallback clears stale scan identities", [] {
        observe(WM_KEYDOWN, VK_CAPITAL, 0x3A);
        windows_shortcut_message_state().pending_down_vks.clear();
        observe(WM_KEYDOWN, VK_PROCESSKEY);
        observe(WM_KEYDOWN, 0xF0, 0x3A);
        require(windows_shortcut_message_state().pressed_vks.count(0xF0) != 0, "IME fallback left a stale scan");
        observe(WM_KEYUP, VK_CAPITAL, 0x3A);
        expect_normal_shortcuts();
    });

    std::cout << cases - failures << '/' << cases << " passed\n";
    return failures ? 1 : 0;
}
