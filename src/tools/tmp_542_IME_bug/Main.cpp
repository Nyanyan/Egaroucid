# include <Siv3D.hpp>

# if SIV3D_PLATFORM(WINDOWS)
# include <Windows.h>
# include <imm.h>
# include <msctf.h>
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
	template <class T>
	void SafeRelease(T*& p) noexcept
	{
		if (p)
		{
			p->Release();
			p = nullptr;
		}
	}

	void PostEscapeKeyToWindow(const HWND hwnd)
	{
		if (not hwnd)
		{
			return;
		}

		PostMessageW(hwnd, WM_KEYDOWN, VK_ESCAPE, 0x00010001);
		PostMessageW(hwnd, WM_KEYUP, VK_ESCAPE, 0xC0010001);
	}

	[[nodiscard]]
	bool CancelTSFComposition()
	{
		using TFGetThreadMgrFn = HRESULT(WINAPI*)(ITfThreadMgr**);

		HMODULE hMsctf = LoadLibraryW(L"msctf.dll");
		if (not hMsctf)
		{
			return false;
		}

		const auto tfGetThreadMgr = reinterpret_cast<TFGetThreadMgrFn>(GetProcAddress(hMsctf, "TF_GetThreadMgr"));
		if (not tfGetThreadMgr)
		{
			FreeLibrary(hMsctf);
			return false;
		}

		ITfThreadMgr* threadMgr = nullptr;
		const HRESULT hrGetMgr = tfGetThreadMgr(&threadMgr);
		FreeLibrary(hMsctf);
		if (FAILED(hrGetMgr) || (not threadMgr))
		{
			return false;
		}

		ITfDocumentMgr* docMgr = nullptr;
		ITfContext* context = nullptr;
		ITfContextOwnerCompositionServices* compositionServices = nullptr;
		bool cancelled = false;

		if (SUCCEEDED(threadMgr->GetFocus(&docMgr)) && docMgr)
		{
			if (SUCCEEDED(docMgr->GetTop(&context)) && context)
			{
				if (SUCCEEDED(context->QueryInterface(IID_ITfContextOwnerCompositionServices,
					reinterpret_cast<void**>(&compositionServices))) && compositionServices)
				{
					// null = terminate all active compositions in this context
					cancelled = SUCCEEDED(compositionServices->TerminateComposition(nullptr));
				}
			}
		}

		SafeRelease(compositionServices);
		SafeRelease(context);
		SafeRelease(docMgr);
		SafeRelease(threadMgr);
		return cancelled;
	}

	void CancelIMEComposition()
	{
		const auto hwnd = static_cast<HWND>(Platform::Windows::Window::GetHWND());
		if (not hwnd)
		{
			return;
		}

		if (const auto himc = ImmGetContext(hwnd))
		{
			const BOOL wasOpen = ImmGetOpenStatus(himc);

			ImmNotifyIME(himc, NI_COMPOSITIONSTR, CPS_COMPLETE, 0);
			ImmNotifyIME(himc, NI_COMPOSITIONSTR, CPS_CANCEL, 0);
			ImmNotifyIME(himc, NI_CLOSECANDIDATE, 0, 0);

			wchar_t empty[] = L"";
			ImmSetCompositionStringW(himc, SCS_SETSTR, empty, sizeof(wchar_t), empty, sizeof(wchar_t));

			ImmSetOpenStatus(himc, FALSE);
			ImmSetOpenStatus(himc, wasOpen);

			ImmReleaseContext(hwnd, himc);
		}

		(void)CancelTSFComposition();
		SendMessageW(hwnd, WM_IME_ENDCOMPOSITION, 0, 0);
		PostMessageW(hwnd, WM_IME_ENDCOMPOSITION, 0, 0);
		PostEscapeKeyToWindow(hwnd);
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
			(void)FlushTextInputBuffer();
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
		m_bodyFont(U"(Transition requests IME cancel, then moves to sub scene)").draw(40, 460, Palette::Orange);
	}
};

class TextInputScene : public App::Scene
{
private:
	Font m_titleFont{ 32, Typeface::Bold };
	Font m_bodyFont{ 18 };
	TextAreaEditState m_textArea;
	bool m_waitingForCleanIME = true;
	int32 m_sanitizeFrames = 0;
	int32 m_cleanStableFrames = 0;
	bool m_sanitizeTimedOut = false;
	String m_lastDiscarded;

	void restartSanitize()
	{
		m_waitingForCleanIME = true;
		m_sanitizeFrames = 0;
		m_cleanStableFrames = 0;
		m_sanitizeTimedOut = false;
		m_lastDiscarded.clear();
		ClearTextAreaState(m_textArea);
		m_textArea.active = false;
	}

	void runSanitizeStep()
	{
		++m_sanitizeFrames;

# if SIV3D_PLATFORM(WINDOWS)
		CancelIMEComposition();
# endif
		m_lastDiscarded = FlushTextInputBuffer();

		if (HasIMEEditingText() || (not m_lastDiscarded.isEmpty()))
		{
			m_cleanStableFrames = 0;
		}
		else
		{
			++m_cleanStableFrames;
		}

		if (2 <= m_cleanStableFrames)
		{
			m_waitingForCleanIME = false;
			m_sanitizeTimedOut = false;
			ClearTextAreaState(m_textArea);
			m_textArea.active = true;
			return;
		}

		// Avoid "freeze-looking" behavior.
		// If IME still remains, keep TextArea inactive and ask for manual retry.
		if (60 <= m_sanitizeFrames)
		{
			m_waitingForCleanIME = false;
			m_sanitizeTimedOut = true;
			m_textArea.active = false;
		}
	}

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
			restartSanitize();
		}

		if (m_waitingForCleanIME)
		{
			runSanitizeStep();
		}

		if (m_textArea.active)
		{
			SimpleGUI::TextArea(m_textArea, Vec2{ 100, 240 }, SizeF{ 800, 46 }, 128);
		}
		else
		{
			RectF{ 100, 240, 800, 46 }.draw(ColorF{ 0.16 });
			if (m_waitingForCleanIME)
			{
				m_bodyFont(U"Sanitizing IME composition before TextArea activation...").draw(120, 252, Palette::Orange);
			}
			else
			{
				m_bodyFont(U"TextArea is inactive (IME composition is still pending).").draw(120, 252, Palette::Orange);
			}
		}

		if (SimpleGUI::Button(U"Back To Main", Vec2{ 100, 320 }, 180))
		{
# if SIV3D_PLATFORM(WINDOWS)
			CancelIMEComposition();
# endif
			(void)FlushTextInputBuffer();
			changeScene(SceneName::Main, 0s);
		}

		if (SimpleGUI::Button(U"Clear TextArea", Vec2{ 300, 320 }, 180))
		{
# if SIV3D_PLATFORM(WINDOWS)
			CancelIMEComposition();
# endif
			(void)FlushTextInputBuffer();
			ClearTextAreaState(m_textArea);
			m_textArea.active = (not HasIMEEditingText());
		}

# if SIV3D_PLATFORM(WINDOWS)
		if (SimpleGUI::Button(U"Cancel IME Composition", Vec2{ 500, 320 }, 220))
		{
			restartSanitize();
		}
# endif

		if ((not m_textArea.active) && (not m_waitingForCleanIME))
		{
			if (SimpleGUI::Button(U"Retry Activate TextArea", Vec2{ 740, 320 }, 220))
			{
				restartSanitize();
			}
		}
	}

	void draw() const override
	{
		Scene::SetBackground(ColorF{ 0.12, 0.31, 0.24 });
		m_titleFont(U"Sub Scene (TextArea)").draw(Arg::topCenter(500, 30), ColorF{ 0.95 });
		m_bodyFont(U"If IME text leaked from MainScene, it appears in the TextArea on entry.").draw(100, 140, Palette::White);
		m_bodyFont(U"Try repeating the flow with Microsoft IME / Google Japanese Input.").draw(100, 170, Palette::White);
		m_bodyFont(U"Current IME editing text: [{}]"_fmt(TextInput::GetEditingText())).draw(100, 200, Palette::White);
		m_bodyFont(U"Last discarded by UpdateText: [{}]"_fmt(m_lastDiscarded)).draw(100, 230, Palette::White);
		m_bodyFont(U"Sanitize frames: {} / stable: {}"_fmt(m_sanitizeFrames, m_cleanStableFrames)).draw(100, 360, Palette::White);
		m_bodyFont(U"Sanitize timed out: {}"_fmt(m_sanitizeTimedOut)).draw(100, 390, Palette::White);
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
		if (not manager.update())
		{
			break;
		}
	}
}
