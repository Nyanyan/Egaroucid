#include <iostream>
#include <future>
#include "ai.hpp"
#include "gui/gui.hpp"


using namespace std;

void Main() {
	Size window_size = Size(WINDOW_SIZE_X, WINDOW_SIZE_Y);
	Window::Resize(window_size);
	Window::SetStyle(WindowStyle::Sizable);
	Scene::SetResizeMode(ResizeMode::Keep);
	Window::SetTitle(U"Egaroucid {}"_fmt(EGAROUCID_VERSION));
	System::SetTerminationTriggers(UserAction::NoAction);
	//Console.open();
	stringstream logger_stream;
	cerr.rdbuf(logger_stream.rdbuf());
	string logger;
	String logger_String;
	
	App scene_manager;
	scene_manager.add <Silent_load> (U"Silent_load");
	scene_manager.add <Load>(U"Load");
	scene_manager.add <Main_scene>(U"Main_scene");
	scene_manager.add <Import_book>(U"Import_book");
	scene_manager.add <Refer_book>(U"Refer_book");
	scene_manager.add <Learn_book>(U"Learn_book");
	scene_manager.add <Import_transcript>(U"Import_transcript");
	scene_manager.add <Import_board>(U"Import_board");
	scene_manager.add <Edit_board>(U"Edit_board");
	scene_manager.add <Import_game>(U"Import_game");
	scene_manager.add <Export_game>(U"Export_game");
	scene_manager.add <Close>(U"Close");
	scene_manager.setFadeColor(Palette::Black);
	scene_manager.init(U"Silent_load");

	while (System::Update()) {
		scene_manager.update();
		while (getline(logger_stream, logger))
			logger_String = Unicode::Widen(logger);
		logger_stream.clear();
		if (scene_manager.get()->menu_elements.show_log) {
			scene_manager.get()->fonts.font12(logger_String).draw(Arg::bottomLeft(8, WINDOW_SIZE_Y - 5), scene_manager.get()->colors.white);
		}
	}
}
