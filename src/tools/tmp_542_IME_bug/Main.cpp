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
	void CancelIMEComposition()
	{
		if (const auto hwnd = static_cast<HWND>(Platform::Windows::Window::GetHWND()))
		{
			if (const auto himc = ImmGetContext(hwnd))
			{
				const BOOL wasOpen = ImmGetOpenStatus(himc);

				// Clear current composition and candidate UI.
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
	}
};

class TextInputScene : public App::Scene
{
private:
	Font m_titleFont{ 32, Typeface::Bold };
	Font m_bodyFont{ 18 };
	TextAreaEditState m_textArea;
	bool m_enterSanitizing = true;
	int32 m_enterSanitizeFrames = 0;
	int32 m_enterStableFrames = 0;
	bool m_pendingClear = false;
	int32 m_pendingClearFrames = 0;
	int32 m_pendingClearStableFrames = 0;

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
			m_pendingClear = false;
			m_textArea.active = false;
		}

		if (m_enterSanitizing)
		{
			++m_enterSanitizeFrames;
# if SIV3D_PLATFORM(WINDOWS)
			CancelIMEComposition();
# endif
			const String discarded = FlushTextInputBuffer();

			if (HasIMEEditingText() || (not discarded.isEmpty()))
			{
				m_enterStableFrames = 0;
				ClearTextAreaState(m_textArea);
				m_textArea.active = false;
			}
			else
			{
				++m_enterStableFrames;
			}

			if ((2 <= m_enterStableFrames) || (120 <= m_enterSanitizeFrames))
			{
				m_enterSanitizing = false;
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

