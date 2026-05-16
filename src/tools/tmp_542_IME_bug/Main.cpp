# include <Siv3D.hpp>

# if SIV3D_PLATFORM(WINDOWS)
# include <Windows.h>
# include <imm.h>
# include <msctf.h>
# endif

namespace
{
	struct DeferredImeCandidateWindowState
	{
		bool requested{ false };
		Vec2 pos{ 0.0, 0.0 };
	};

	DeferredImeCandidateWindowState g_deferredImeCandidateWindow;

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

	[[nodiscard]]
	Vec2 CalculateTextAreaEditingTextPos(
		const TextAreaEditState& text,
		const Vec2& pos,
		const SizeF& size
	)
	{
		// Match SimpleGUI::TextArea internal text render region (Siv3D 0.6.x).
		const RectF region = SimpleGUI::TextAreaRegion(pos, size);
		const RectF textRenderRegion = region.stretched(-2.0, -(6.0 + 3.0), -2.0, -8.0);
		const Vec2 defaultPos = textRenderRegion.pos.movedBy(0.0, text.scrollY);

		if ((text.cursorPos == 0) || text.glyphs.isEmpty())
		{
			return defaultPos;
		}

		const size_t targetIndex = (text.cursorPos - 1);
		if (targetIndex >= text.glyphs.size())
		{
			return defaultPos;
		}

		for (const auto& clipInfo : text.clipInfos)
		{
			if (clipInfo.index != targetIndex)
			{
				continue;
			}

			const Glyph& glyph = text.glyphs[targetIndex];
			const bool isLineFeed = (glyph.codePoint == U'\n');
			const double caretX = (clipInfo.pos.x + (isLineFeed ? 0.0 : clipInfo.clipRect.w));
			const double caretY = (clipInfo.pos.y - glyph.getOffset().y);
			return Vec2{ caretX, caretY };
		}

		return defaultPos;
	}

	[[nodiscard]]
	Vec2 CalculateTextAreaIMECandidatePos(
		const TextAreaEditState& text,
		const Vec2& pos,
		const SizeF& size
	)
	{
		const Vec2 editingPos = CalculateTextAreaEditingTextPos(text, pos, size);
		return Vec2{ editingPos.x, (editingPos.y + SimpleGUI::GetFont().height() + 2.0) };
	}

	void DrawIMECandidateWindowLimited(const Vec2& pos)
	{
# if SIV3D_PLATFORM(WINDOWS)
		const auto& candidateState = Platform::Windows::TextInput::GetCandidateState();
		if (candidateState.candidates.isEmpty())
		{
			return;
		}

		const Font& font = SimpleGUI::GetFont();
		constexpr ColorF CANDIDATE_WINDOW_COLOR{ 0.98 };
		constexpr ColorF CANDIDATE_WINDOW_FRAME_COLOR{ 0.75 };
		constexpr ColorF CANDIDATE_SELECTED_BACKGROUND_COLOR{ 0.55, 0.85, 1.0 };
		constexpr ColorF CANDIDATE_TEXT_COLOR{ 0.11 };
		constexpr ColorF CANDIDATE_MINIMAP_COLOR{ 0.67 };
		constexpr double CANDIDATE_MARGIN = 4.0;
		constexpr double CANDIDATE_PADDING = 12.0;
		constexpr double CANDIDATE_MINIMAP_WIDTH = 20.0;

		const double candidateItemHeight = (font.height() + CANDIDATE_MARGIN);
		const double availableHeight = (Scene::Size().y - pos.y - 2.0);
		int32 visibleCount = 0;
		if (0.0 < candidateItemHeight)
		{
			visibleCount = static_cast<int32>(availableHeight / candidateItemHeight);
		}
		visibleCount = Clamp(visibleCount, 0, static_cast<int32>(candidateState.candidates.size()));
		if (visibleCount <= 0)
		{
			return;
		}

		double boxWidth = 0.0;
		for (const auto& candidate : candidateState.candidates)
		{
			boxWidth = Max<double>(boxWidth, font(candidate).region().w);
		}
		boxWidth += (CANDIDATE_PADDING * 2 + CANDIDATE_MINIMAP_WIDTH);

		const RectF boxRect{ pos, boxWidth, (candidateItemHeight * visibleCount) };
		boxRect
			.drawShadow(Vec2{ 0, 2 }, 8)
			.draw(CANDIDATE_WINDOW_COLOR)
			.drawFrame(1, 0, CANDIDATE_WINDOW_FRAME_COLOR);

		int32 currentIndex = candidateState.pageStartIndex;
		for (int32 i = 0; i < visibleCount; ++i)
		{
			const bool selected = (candidateState.selectedIndex && (currentIndex == *candidateState.selectedIndex));
			const Vec2 itemPos{ pos.x, (pos.y + i * candidateItemHeight) };
			if (selected)
			{
				RectF{ itemPos, (boxWidth - CANDIDATE_MINIMAP_WIDTH), candidateItemHeight }
					.stretched(-1, 0)
					.draw(CANDIDATE_SELECTED_BACKGROUND_COLOR);
			}
			if (candidateState.candidates[i])
			{
				font(candidateState.candidates[i]).draw(
					itemPos.movedBy(CANDIDATE_PADDING, (CANDIDATE_MARGIN * 0.5 - 1.0)),
					CANDIDATE_TEXT_COLOR
				);
			}
			++currentIndex;
		}

		const bool hasPrev = (candidateState.pageStartIndex != 0);
		const bool hasNext = ((candidateState.pageStartIndex + visibleCount) < candidateState.count);
		if (hasPrev)
		{
			const Vec2 scrollPos{
				(pos.x + boxWidth - CANDIDATE_MINIMAP_WIDTH * 0.5 - 1),
				(pos.y + 11)
			};
			scrollPos.asCircle(3.5).draw(CANDIDATE_MINIMAP_COLOR);
			scrollPos.movedBy(0, 8).asCircle(2.8).draw(CANDIDATE_MINIMAP_COLOR);
			scrollPos.movedBy(0, 15).asCircle(2.25).draw(CANDIDATE_MINIMAP_COLOR);
		}
		if (hasNext)
		{
			const Vec2 scrollPos{
				(pos.x + boxWidth - CANDIDATE_MINIMAP_WIDTH * 0.5 - 1),
				(pos.y + visibleCount * candidateItemHeight - 9)
			};
			scrollPos.asCircle(3.5).draw(CANDIDATE_MINIMAP_COLOR);
			scrollPos.movedBy(0, -8).asCircle(2.8).draw(CANDIDATE_MINIMAP_COLOR);
			scrollPos.movedBy(0, -15).asCircle(2.25).draw(CANDIDATE_MINIMAP_COLOR);
		}
# else
		SimpleGUI::IMECandidateWindow(pos);
# endif
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
	String transitionEditingText;
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
			getData().transitionEditingText = TextInput::GetEditingText();
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
	String m_staleEditingText;
	bool m_staleGuardEnabled = false;
	int32 m_staleGuardStableFrames = 0;
	int32 m_staleReappearCount = 0;

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

	[[nodiscard]]
	bool IsStaleCompositionReappeared() const
	{
		if (not m_staleGuardEnabled)
		{
			return false;
		}

		const String editing = TextInput::GetEditingText();
		return (not editing.isEmpty()) && (editing == m_staleEditingText);
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
			m_staleEditingText = getData().transitionEditingText;
			m_staleGuardEnabled = (not m_staleEditingText.isEmpty());
			m_staleGuardStableFrames = 0;
			m_staleReappearCount = 0;
			restartSanitize();
		}

		if (m_waitingForCleanIME)
		{
			runSanitizeStep();
		}
		else if (IsStaleCompositionReappeared())
		{
			++m_staleReappearCount;
			restartSanitize();
		}
		else if (m_staleGuardEnabled)
		{
			const String editing = TextInput::GetEditingText();
			if (editing.isEmpty())
			{
				++m_staleGuardStableFrames;
				if (2 <= m_staleGuardStableFrames)
				{
					m_staleGuardEnabled = false;
				}
			}
			else if (editing != m_staleEditingText)
			{
				// New user composition has started; stop stale-string guard.
				m_staleGuardEnabled = false;
			}
			else
			{
				m_staleGuardStableFrames = 0;
			}
		}

		const Vec2 textAreaPos{ 100, 240 };
		const SizeF textAreaSize{ 800, 46 };
		bool requestImeCandidateWindow = false;
		Vec2 imeCandidatePos{ 0.0, 0.0 };

		if (m_textArea.active)
		{
			SimpleGUI::TextArea(m_textArea, textAreaPos, textAreaSize, 128);
			if (m_textArea.active)
			{
				requestImeCandidateWindow = true;
				imeCandidatePos = CalculateTextAreaIMECandidatePos(m_textArea, textAreaPos, textAreaSize);
			}
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

		if (requestImeCandidateWindow)
		{
			g_deferredImeCandidateWindow.requested = true;
			g_deferredImeCandidateWindow.pos = imeCandidatePos;
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
		m_bodyFont(U"Stale guard text: [{}]"_fmt(m_staleEditingText)).draw(100, 260, Palette::White);
		m_bodyFont(U"Stale guard enabled: {}"_fmt(m_staleGuardEnabled)).draw(100, 290, Palette::White);
		m_bodyFont(U"Stale reappear count: {}"_fmt(m_staleReappearCount)).draw(100, 320, Palette::White);
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

		if (g_deferredImeCandidateWindow.requested)
		{
			DrawIMECandidateWindowLimited(g_deferredImeCandidateWindow.pos);
			g_deferredImeCandidateWindow.requested = false;
		}
	}
}
