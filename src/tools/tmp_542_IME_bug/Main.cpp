# include <Siv3D.hpp>

# if SIV3D_PLATFORM(WINDOWS)
namespace s3d::Platform::Windows::TextInput
{
	void EnableIME();
}
# endif

enum class SceneName
{
	Main,
	TextInput,
};

struct SharedData
{
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
	MainScene(const InitData& init)
		: IScene{ init }
	{
# if SIV3D_PLATFORM(WINDOWS)
		Platform::Windows::TextInput::DisableIME();
# endif
	}

	void update() override
	{
# if SIV3D_PLATFORM(WINDOWS)
		Platform::Windows::TextInput::DisableIME();
# endif

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
			Platform::Windows::TextInput::EnableIME();
# endif
			changeScene(SceneName::TextInput, 0s);
		}
	}

	void draw() const override
	{
		Scene::SetBackground(ColorF{ 0.17, 0.24, 0.33 });

		m_titleFont(U"IME Pending Text Repro").draw(Arg::topCenter(500, 30), ColorF{ 0.95 });
		m_bodyFont(U"1) Turn IME ON in Japanese kana mode").draw(40, 120, Palette::White);
		m_bodyFont(U"2) Press A twice, D twice, E once on this scene").draw(40, 150, Palette::White);
		m_bodyFont(U"3) Click the button below to open the TextArea scene").draw(40, 180, Palette::White);
		m_bodyFont(U"Expected repro: TextArea may already contain committed IME text.").draw(40, 210, Palette::Orange);

		m_bodyFont(U"Key counts in this scene").draw(40, 280, Palette::White);
		m_bodyFont(U"A: {}"_fmt(m_aCount)).draw(80, 320, Palette::Skyblue);
		m_bodyFont(U"D: {}"_fmt(m_dCount)).draw(80, 350, Palette::Skyblue);
		m_bodyFont(U"E: {}"_fmt(m_eCount)).draw(80, 380, Palette::Skyblue);
	}
};

class TextInputScene : public App::Scene
{
private:
	Font m_titleFont{ 32, Typeface::Bold };
	Font m_bodyFont{ 18 };
	TextAreaEditState m_textArea;

public:
	TextInputScene(const InitData& init)
		: IScene{ init }
	{
# if SIV3D_PLATFORM(WINDOWS)
		Platform::Windows::TextInput::EnableIME();
# endif
		m_textArea.text.clear();
		m_textArea.cursorPos = 0;
		m_textArea.rebuildGlyphs();
		m_textArea.active = true;
	}

	void update() override
	{
# if SIV3D_PLATFORM(WINDOWS)
		Platform::Windows::TextInput::EnableIME();
# endif

		SimpleGUI::TextArea(m_textArea, Vec2{ 100, 240 }, SizeF{ 800, 46 }, 128);

		if (SimpleGUI::Button(U"Back To Main", Vec2{ 100, 320 }, 180))
		{
# if SIV3D_PLATFORM(WINDOWS)
			Platform::Windows::TextInput::DisableIME();
# endif
			changeScene(SceneName::Main, 0s);
		}

		if (SimpleGUI::Button(U"Clear TextArea", Vec2{ 300, 320 }, 180))
		{
			m_textArea.text.clear();
			m_textArea.cursorPos = 0;
			m_textArea.rebuildGlyphs();
		}
	}

	void draw() const override
	{
		Scene::SetBackground(ColorF{ 0.12, 0.31, 0.24 });
		m_titleFont(U"Sub Scene (TextArea)").draw(Arg::topCenter(500, 30), ColorF{ 0.95 });
		m_bodyFont(U"If IME text leaked from MainScene, it appears in the TextArea on entry.").draw(100, 140, Palette::White);
		m_bodyFont(U"Try repeating the flow with Microsoft IME / Google Japanese Input.").draw(100, 170, Palette::White);
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
