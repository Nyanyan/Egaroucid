# include <Siv3D.hpp>

# if SIV3D_PLATFORM(WINDOWS)
# include <Windows.h>
# include <imm.h>
# endif

namespace
{
	[[nodiscard]]
	bool HasIMEEditingText()
	{
		return (not TextInput::GetEditingText().isEmpty());
	}

	void ClearTextAreaState(TextAreaEditState& state)
	{
		state.clear();
		state.active = true;
	}

	[[nodiscard]]
	String FlushTextInputBuffer()
	{
		String discarded;
		TextInput::UpdateText(discarded, TextInputMode::DenyControl);
		return discarded;
	}

# if SIV3D_PLATFORM(WINDOWS)
	[[nodiscard]]
	int32 DrainMessageRange(const UINT first, const UINT last)
	{
		int32 count = 0;
		MSG msg{};
		while (PeekMessageW(&msg, nullptr, first, last, PM_REMOVE))
		{
			++count;
		}
		return count;
	}

	[[nodiscard]]
	int32 DrainPendingInputMessages()
	{
		int32 total = 0;
		total += DrainMessageRange(WM_KEYFIRST, WM_KEYLAST);
		total += DrainMessageRange(WM_CHAR, WM_DEADCHAR);
		total += DrainMessageRange(WM_SYSKEYDOWN, WM_SYSDEADCHAR);
		total += DrainMessageRange(WM_IME_STARTCOMPOSITION, WM_IME_COMPOSITION);
		total += DrainMessageRange(WM_IME_SETCONTEXT, WM_IME_KEYUP);
		return total;
	}

	void ResetIMEContextAssociation()
	{
		if (const auto hwnd = static_cast<HWND>(Platform::Windows::Window::GetHWND()))
		{
			if (HIMC oldContext = ImmAssociateContext(hwnd, nullptr))
			{
				ImmAssociateContext(hwnd, oldContext);
			}

			ImmAssociateContextEx(hwnd, nullptr, IACE_DEFAULT);
		}
	}

	void CancelIMEComposition()
	{
		if (const auto hwnd = static_cast<HWND>(Platform::Windows::Window::GetHWND()))
		{
			if (const auto himc = ImmGetContext(hwnd))
			{
				const BOOL wasOpen = ImmGetOpenStatus(himc);

				// Clear current composition and candidate UI.
				ImmNotifyIME(himc, NI_COMPOSITIONSTR, CPS_COMPLETE, 0);
				ImmNotifyIME(himc, NI_COMPOSITIONSTR, CPS_CANCEL, 0);
				ImmNotifyIME(himc, NI_CLOSECANDIDATE, 0, 0);

				// Some IME implementations keep the comp string after CPS_CANCEL.
				// Force the composition string to empty.
				wchar_t empty[] = L"";
				ImmSetCompositionStringW(himc, SCS_SETSTR, empty, sizeof(wchar_t), empty, sizeof(wchar_t));

				// Re-open state is restored immediately to keep IME usability.
				ImmSetOpenStatus(himc, FALSE);
				ImmSetOpenStatus(himc, wasOpen);

				ImmReleaseContext(hwnd, himc);
			}
		}

		ResetIMEContextAssociation();
	}
# endif
}

enum class SceneName
{
	Main,
	TextInput,
};

struct SharedData
{
	bool sanitizeTextInputSceneOnEnter = false;
};

using App = SceneManager<SceneName, SharedData>;

class MainScene : public App::Scene
{
private:
	Font m_titleFont{ 34, Typeface::Heavy };
	Font m_bodyFont{ 18 };
	int32 m_aCount = 0;
	int32 m_dCount = 0;
	int32 m_eCount = 0;
	int32 m_lastDrainCount = 0;

public:
	using App::Scene::Scene;

	void update() override
	{
		// Intentionally avoid TextInput::UpdateText() in this scene.
		// IME committed text can remain pending and appear when a TextArea becomes active later.
		if (KeyA.down())
		{
			++m_aCount;
		}
		if (KeyD.down())
		{
			++m_dCount;
		}
		if (KeyE.down())
		{
			++m_eCount;
		}

		if (SimpleGUI::Button(U"Open Sub Scene (TextArea)", Vec2{ 320, 500 }, 320))
		{
# if SIV3D_PLATFORM(WINDOWS)
			CancelIMEComposition();
			m_lastDrainCount = DrainPendingInputMessages();
# endif
			FlushTextInputBuffer();
			getData().sanitizeTextInputSceneOnEnter = true;
			changeScene(SceneName::TextInput, 0s);
		}

# if SIV3D_PLATFORM(WINDOWS)
		if (SimpleGUI::Button(U"Cancel IME Composition", Vec2{ 320, 540 }, 320))
		{
			CancelIMEComposition();
		}
# endif
	}

	void draw() const override
	{
		Scene::SetBackground(ColorF{ 0.17, 0.24, 0.33 });

		m_titleFont(U"IME Pending Text Repro").draw(Arg::topCenter(500, 30), ColorF{ 0.95 });
		m_bodyFont(U"1) Turn IME ON in Japanese kana mode").draw(40, 120, Palette::White);
		m_bodyFont(U"2) Press A twice, D twice, E once on this scene").draw(40, 150, Palette::White);
		m_bodyFont(U"3) Click the button below (scene transitions immediately)").draw(40, 180, Palette::White);
		m_bodyFont(U"Expected repro: TextArea may already contain committed IME text.").draw(40, 210, Palette::Orange);

		m_bodyFont(U"Key counts in this scene").draw(40, 280, Palette::White);
		m_bodyFont(U"A: {}"_fmt(m_aCount)).draw(80, 320, Palette::Skyblue);
		m_bodyFont(U"D: {}"_fmt(m_dCount)).draw(80, 350, Palette::Skyblue);
		m_bodyFont(U"E: {}"_fmt(m_eCount)).draw(80, 380, Palette::Skyblue);

		const String editingText = TextInput::GetEditingText();
		m_bodyFont(U"IME editing text: [{}]"_fmt(editingText)).draw(40, 430, Palette::White);
		m_bodyFont(U"(Open Sub Scene now transitions even while IME composition is active)").draw(40, 460, Palette::Orange);
		m_bodyFont(U"Drained pending Win32 input messages: {}"_fmt(m_lastDrainCount)).draw(40, 490, Palette::White);
	}
};

class TextInputScene : public App::Scene
{
private:
	Font m_titleFont{ 32, Typeface::Bold };
	Font m_bodyFont{ 18 };
	TextAreaEditState m_textArea;
	TextEditState m_probeTextBox;
	bool m_enterSanitizing = true;
	int32 m_enterSanitizeFrames = 0;
	int32 m_enterStableFrames = 0;
	bool m_enterSanitizeForced = false;
	int32 m_lastEnterDrainCount = 0;
	String m_lastProbeText;
	bool m_pendingClear = false;
	int32 m_pendingClearFrames = 0;
	int32 m_pendingClearStableFrames = 0;
	int32 m_lastClearDrainCount = 0;

public:
	TextInputScene(const InitData& init)
		: IScene{ init }
	{
		ClearTextAreaState(m_textArea);
		m_textArea.active = false;
	}

	void update() override
	{
		if (getData().sanitizeTextInputSceneOnEnter)
		{
			getData().sanitizeTextInputSceneOnEnter = false;
			m_enterSanitizing = true;
			m_enterSanitizeFrames = 0;
			m_enterStableFrames = 0;
			m_enterSanitizeForced = false;
			m_lastProbeText.clear();
			m_probeTextBox.text.clear();
			m_probeTextBox.cursorPos = 0;
			m_probeTextBox.active = true;
			m_pendingClear = false;
			m_textArea.active = false;
		}

		if (m_enterSanitizing)
		{
			++m_enterSanitizeFrames;

			// Keep a hidden text box active during the first few frames only.
			// Keeping it active forever can itself keep IME composition alive.
			if (m_enterSanitizeFrames <= 3)
			{
				m_probeTextBox.active = true;
			}
			else
			{
				m_probeTextBox.active = false;
			}
			SimpleGUI::TextBox(m_probeTextBox, Vec2{ -10000, -10000 }, 80, 64);

# if SIV3D_PLATFORM(WINDOWS)
			CancelIMEComposition();
			m_lastEnterDrainCount = DrainPendingInputMessages();
# endif
			const String discarded = FlushTextInputBuffer();
			m_lastProbeText = m_probeTextBox.text;

			if (HasIMEEditingText() || (not discarded.isEmpty()) || (not m_probeTextBox.text.isEmpty()))
			{
				m_enterStableFrames = 0;
				ClearTextAreaState(m_textArea);
				m_textArea.active = false;
				m_probeTextBox.text.clear();
				m_probeTextBox.cursorPos = 0;
			}
			else
			{
				++m_enterStableFrames;
			}

			if ((2 <= m_enterStableFrames) || (20 <= m_enterSanitizeFrames))
			{
				m_enterSanitizeForced = (20 <= m_enterSanitizeFrames);
				m_enterSanitizing = false;
				m_probeTextBox.active = false;
				m_probeTextBox.text.clear();
				m_probeTextBox.cursorPos = 0;
				ClearTextAreaState(m_textArea);
			}
		}

		if (not m_enterSanitizing)
		{
			SimpleGUI::TextArea(m_textArea, Vec2{ 100, 240 }, SizeF{ 800, 46 }, 128);
		}

		if (SimpleGUI::Button(U"Back To Main", Vec2{ 100, 320 }, 180))
		{
# if SIV3D_PLATFORM(WINDOWS)
			CancelIMEComposition();
# endif
			FlushTextInputBuffer();
			changeScene(SceneName::Main, 0s);
		}

		if (SimpleGUI::Button(U"Clear TextArea", Vec2{ 300, 320 }, 180))
		{
			m_pendingClear = true;
			m_pendingClearFrames = 0;
			m_pendingClearStableFrames = 0;
			m_textArea.active = false;
		}

		if (m_pendingClear)
		{
			++m_pendingClearFrames;
# if SIV3D_PLATFORM(WINDOWS)
			CancelIMEComposition();
			m_lastClearDrainCount = DrainPendingInputMessages();
# endif
			const String discarded = FlushTextInputBuffer();

			if (HasIMEEditingText() || (not discarded.isEmpty()))
			{
				m_pendingClearStableFrames = 0;
			}
			else
			{
				++m_pendingClearStableFrames;
			}

			if ((2 <= m_pendingClearStableFrames) || (120 <= m_pendingClearFrames))
			{
				m_pendingClear = false;
				ClearTextAreaState(m_textArea);
			}
		}

# if SIV3D_PLATFORM(WINDOWS)
		if (SimpleGUI::Button(U"Cancel IME Composition", Vec2{ 500, 320 }, 220))
		{
			CancelIMEComposition();
			ClearTextAreaState(m_textArea);
		}
# endif
	}

	void draw() const override
	{
		Scene::SetBackground(ColorF{ 0.12, 0.31, 0.24 });
		m_titleFont(U"Sub Scene (TextArea)").draw(Arg::topCenter(500, 30), ColorF{ 0.95 });
		m_bodyFont(U"If IME text leaked from MainScene, it appears in the TextArea on entry.").draw(100, 140, Palette::White);
		m_bodyFont(U"Try repeating the flow with Microsoft IME / Google Japanese Input.").draw(100, 170, Palette::White);
		m_bodyFont(U"Current IME editing text: [{}]"_fmt(TextInput::GetEditingText())).draw(100, 200, Palette::White);
		m_bodyFont(U"Probe TextBox text: [{}]"_fmt(m_lastProbeText)).draw(100, 230, Palette::White);
		m_bodyFont(U"Sanitize forced exit: {}"_fmt(m_enterSanitizeForced)).draw(100, 260, Palette::White);
		m_bodyFont(U"Enter sanitize drained messages: {}"_fmt(m_lastEnterDrainCount)).draw(100, 360, Palette::White);
		m_bodyFont(U"Clear-action drained messages: {}"_fmt(m_lastClearDrainCount)).draw(100, 390, Palette::White);
		if (m_enterSanitizing)
		{
			RectF{ 100, 240, 800, 46 }.draw(ColorF{ 0.15 });
			m_bodyFont(U"Sanitizing IME composition before TextArea activation...").draw(120, 252, Palette::Orange);
		}
	}
};

void Main()
{
	Window::Resize(1000, 650);
	Window::SetTitle(U"tmp_542_IME_bug");
	Scene::SetBackground(ColorF{ 0.2 });
	System::SetTerminationTriggers(UserAction::CloseButtonClicked);

	App manager;
	manager.add<MainScene>(SceneName::Main);
	manager.add<TextInputScene>(SceneName::TextInput);
	manager.init(SceneName::Main);

	while (System::Update())
	{
		if (!manager.update())
		{
			break;
		}
	}
}

