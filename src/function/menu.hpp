#pragma once
#include <iostream>
#include <string>
#include <vector>
#include "./../gui/menu.hpp"
#include "preference.hpp"

struct Menu_contents_mode {
	bool simple;
	bool professional;
	bool game;
};

struct Menu_contents_game {
	bool start;
	bool analyze;
};

struct Menu_contents_setting_ai {
	bool use_book;
	int ai_level;
	int hint_level;
	int graph_level;
	int error_level;
	int n_thread;
};

struct Menu_contents_setting_player {
	bool human_first;
	bool ai_first;
	bool both_ai;
	bool both_human;
};

struct Menu_contents_setting {
	Menu_contents_setting_ai ai;
	Menu_contents_setting_player player;
};

struct Menu_contents_display_hint {
	bool disc_difference;
	int n_disc_difference;
	bool human_value;
	bool umigame_value;
};

struct Menu_contents_display {
	bool hint;
	Menu_contents_display_hint hint_elem;
	bool graph;
	bool joseki_on_cell;
	bool popup_in_end;
	bool log;
};

struct Menu_contents_joseki {
	bool input;
	bool reference;
};

struct Menu_contents_inout_in {
	bool record;
	bool board;
	bool edit_board;
	bool game;
};

struct Menu_contents_inout_out {
	bool record;
	bool game;
};

struct Menu_contents_inout {
	Menu_contents_inout_in in;
	Menu_contents_inout_out out;
};

struct Menu_contents_manipulate_transform {
	bool rotate_180;
	bool black_line;
	bool white_line;
};

struct Menu_contents_manipulate {
	bool stop;
	bool resume;
	bool go;
	bool back;
	Menu_contents_manipulate_transform transform_elem;
};

struct Menu_contents_help {
	bool usage;
	bool report;
	bool update_check;
	bool license;
};

struct Menu_contents_language {
	bool* acts;
	vector<string> name;
};

struct Menu_contents {
	bool dummy;
	Menu_contents_mode mode;
	Menu_contents_game game;
	Menu_contents_setting setting;
	Menu_contents_display display;
	Menu_contents_joseki joseki;
	Menu_contents_inout inout;
	Menu_contents_manipulate manipulate;
	Menu_contents_help help;
	Menu_contents_language language;
};


void create_menu_simple(Font menu_font, Menu_contents* contents, Menu *menu) {
	menu_title title;
	menu_elem menu_e, side_menu, side_side_menu;

	// settings
	title.init(language.get("settings", "settings"));

	contents->setting.ai.ai_level = min(contents->setting.ai.ai_level, 20);
	contents->setting.ai.hint_level = min(contents->setting.ai.hint_level, 20);
	contents->setting.ai.graph_level = min(contents->setting.ai.graph_level, 20);
	contents->setting.ai.n_thread = min(contents->setting.ai.n_thread, 32);

	menu_e.init_button(language.get("ai_settings", "ai_settings"), &contents->dummy);
	side_menu.init_bar(language.get("ai_settings", "ai_level"), &contents->setting.ai.ai_level, contents->setting.ai.ai_level, 0, 20);
	menu_e.push(side_menu);
	side_menu.init_bar(language.get("ai_settings", "hint_level"), &contents->setting.ai.hint_level, contents->setting.ai.hint_level, 0, 20);
	menu_e.push(side_menu);
	side_menu.init_bar(language.get("ai_settings", "graph_level"), &contents->setting.ai.graph_level, contents->setting.ai.graph_level, 0, 20);
	menu_e.push(side_menu);
	side_menu.init_bar(language.get("settings", "thread", "thread"), &contents->setting.ai.n_thread, contents->setting.ai.n_thread, 1, 32);
	menu_e.push(side_menu);
	title.push(menu_e);

	menu_e.init_button(language.get("settings", "play", "play"), &contents->dummy);
	side_menu.init_radio(language.get("settings", "play", "human_first"), &contents->setting.player.human_first, contents->setting.player.human_first);
	menu_e.push(side_menu);
	side_menu.init_radio(language.get("settings", "play", "human_second"), &contents->setting.player.ai_first, contents->setting.player.ai_first);
	menu_e.push(side_menu);
	side_menu.init_radio(language.get("settings", "play", "both_ai"), &contents->setting.player.both_ai, contents->setting.player.both_ai);
	menu_e.push(side_menu);
	side_menu.init_radio(language.get("settings", "play", "both_human"), &contents->setting.player.both_human, contents->setting.player.both_human);
	menu_e.push(side_menu);
	title.push(menu_e);

	menu->push(title);

	// display
	title.init(language.get("display", "display"));

	menu_e.init_check(language.get("display", "hint", "hint"), &contents->display.hint, contents->display.hint);
	title.push(menu_e);

	menu_e.init_check(language.get("display", "graph"), &contents->display.graph, contents->display.graph);
	title.push(menu_e);

	menu_e.init_check(language.get("display", "joseki_on_cell"), &contents->display.joseki_on_cell, contents->display.joseki_on_cell);
	title.push(menu_e);


	menu_e.init_check(language.get("display", "end_popup"), &contents->display.popup_in_end, contents->display.popup_in_end);
	title.push(menu_e);
	menu_e.init_check(language.get("display", "log"), &contents->display.log, contents->display.log);
	title.push(menu_e);

	menu->push(title);

	// manipulate
	title.init(language.get("operation", "operation"));

	menu_e.init_button(language.get("operation", "stop_read"), &contents->manipulate.stop);
	title.push(menu_e);
	menu_e.init_button(language.get("operation", "resume_read"), &contents->manipulate.resume);
	title.push(menu_e);
	menu_e.init_button(language.get("operation", "forward"), &contents->manipulate.go);
	title.push(menu_e);
	menu_e.init_button(language.get("operation", "backward"), &contents->manipulate.back);
	title.push(menu_e);

	menu->push(title);
}

void create_menu_professional(Font menu_font, Menu_contents* contents, Menu* menu) {
	menu_title title;
	menu_elem menu_e, side_menu, side_side_menu;

	// settings
	title.init(language.get("settings", "settings"));

	contents->setting.ai.n_thread = min(contents->setting.ai.n_thread, 32);

	menu_e.init_button(language.get("ai_settings", "ai_settings"), &contents->dummy);
	side_menu.init_bar(language.get("ai_settings", "ai_level"), &contents->setting.ai.ai_level, contents->setting.ai.ai_level, 0, 60);
	menu_e.push(side_menu);
	side_menu.init_bar(language.get("ai_settings", "hint_level"), &contents->setting.ai.hint_level, contents->setting.ai.hint_level, 0, 60);
	menu_e.push(side_menu);
	side_menu.init_bar(language.get("ai_settings", "graph_level"), &contents->setting.ai.graph_level, contents->setting.ai.graph_level, 0, 60);
	menu_e.push(side_menu);
	side_menu.init_bar(language.get("settings", "thread", "thread"), &contents->setting.ai.n_thread, contents->setting.ai.n_thread, 1, 32);
	menu_e.push(side_menu);
	title.push(menu_e);

	menu_e.init_button(language.get("settings", "play", "play"), &contents->dummy);
	side_menu.init_radio(language.get("settings", "play", "human_first"), &contents->setting.player.human_first, contents->setting.player.human_first);
	menu_e.push(side_menu);
	side_menu.init_radio(language.get("settings", "play", "human_second"), &contents->setting.player.ai_first, contents->setting.player.ai_first);
	menu_e.push(side_menu);
	side_menu.init_radio(language.get("settings", "play", "both_ai"), &contents->setting.player.both_ai, contents->setting.player.both_ai);
	menu_e.push(side_menu);
	side_menu.init_radio(language.get("settings", "play", "both_human"), &contents->setting.player.both_human, contents->setting.player.both_human);
	menu_e.push(side_menu);
	title.push(menu_e);

	menu->push(title);

	// display
	title.init(language.get("display", "display"));

	menu_e.init_check(language.get("display", "hint", "hint"), &contents->display.hint, contents->display.hint);
	side_menu.init_check(language.get("display", "hint", "stone_value"), &contents->display.hint_elem.disc_difference, contents->display.hint_elem.disc_difference);
	side_side_menu.init_bar(language.get("display", "hint", "show_number"), &contents->display.hint_elem.n_disc_difference, contents->display.hint_elem.n_disc_difference, 1, 64);
	side_menu.push(side_side_menu);
	menu_e.push(side_menu);

	side_menu.init_check(language.get("display", "hint", "human_value"), &contents->display.hint_elem.human_value, contents->display.hint_elem.human_value);
	menu_e.push(side_menu);
	side_menu.init_check(language.get("display", "hint", "umigame_value"), &contents->display.hint_elem.umigame_value, contents->display.hint_elem.umigame_value);
	menu_e.push(side_menu);
	title.push(menu_e);

	menu_e.init_check(language.get("display", "graph"), &contents->display.graph, contents->display.graph);
	title.push(menu_e);

	menu_e.init_check(language.get("display", "joseki_on_cell"), &contents->display.joseki_on_cell, contents->display.joseki_on_cell);
	title.push(menu_e);


	menu_e.init_check(language.get("display", "end_popup"), &contents->display.popup_in_end, contents->display.popup_in_end);
	title.push(menu_e);
	menu_e.init_check(language.get("display", "log"), &contents->display.log, contents->display.log);
	title.push(menu_e);

	menu->push(title);

	// joseki
	title.init(language.get("book", "book"));

	menu_e.init_button(language.get("book", "settings"), &contents->dummy);
	menu_e.init_button(language.get("book", "import"), &contents->joseki.input);
	title.push(menu_e);
	menu_e.init_button(language.get("book", "book_reference"), &contents->joseki.reference);
	title.push(menu_e);

	menu->push(title);

	// in_out
	title.init(language.get("in_out", "in_out"));

	menu_e.init_button(language.get("in_out", "in"), &contents->dummy);
	side_menu.init_button(language.get("in_out", "input_record"), &contents->inout.in.record);
	menu_e.push(side_menu);
	side_menu.init_button(language.get("in_out", "input_board"), &contents->inout.in.board);
	menu_e.push(side_menu);
	side_menu.init_button(language.get("in_out", "edit_board"), &contents->inout.in.edit_board);
	menu_e.push(side_menu);
	side_menu.init_button(language.get("in_out", "input_game"), &contents->inout.in.game);
	menu_e.push(side_menu);
	title.push(menu_e);

	menu_e.init_button(language.get("in_out", "out"), &contents->dummy);
	side_menu.init_button(language.get("in_out", "output_record"), &contents->inout.out.record);
	menu_e.push(side_menu);
	side_menu.init_button(language.get("in_out", "output_game"), &contents->inout.out.game);
	menu_e.push(side_menu);
	title.push(menu_e);

	menu->push(title);

	// manipulate
	title.init(language.get("operation", "operation"));

	menu_e.init_button(language.get("operation", "stop_read"), &contents->manipulate.stop);
	title.push(menu_e);
	menu_e.init_button(language.get("operation", "resume_read"), &contents->manipulate.resume);
	title.push(menu_e);
	menu_e.init_button(language.get("operation", "forward"), &contents->manipulate.go);
	title.push(menu_e);
	menu_e.init_button(language.get("operation", "backward"), &contents->manipulate.back);
	title.push(menu_e);
	menu_e.init_button(language.get("operation", "convert", "convert"), &contents->dummy);
	side_menu.init_button(language.get("operation", "convert", "vertical"), &contents->manipulate.transform_elem.rotate_180);
	menu_e.push(side_menu);
	side_menu.init_button(language.get("operation", "convert", "black_line"), &contents->manipulate.transform_elem.black_line);
	menu_e.push(side_menu);
	side_menu.init_button(language.get("operation", "convert", "white_line"), &contents->manipulate.transform_elem.white_line);
	menu_e.push(side_menu);
	title.push(menu_e);

	menu->push(title);
}

void create_menu_game(Font menu_font, Menu_contents* contents, Menu* menu) {
	menu_title title;
	menu_elem menu_e, side_menu, side_side_menu;

	// settings
	title.init(language.get("settings", "settings"));

	contents->setting.ai.ai_level = min(contents->setting.ai.ai_level, 20);
	contents->setting.ai.n_thread = min(contents->setting.ai.n_thread, 32);

	menu_e.init_button(language.get("ai_settings", "ai_settings"), &contents->dummy);
	side_menu.init_bar(language.get("ai_settings", "ai_level"), &contents->setting.ai.ai_level, contents->setting.ai.ai_level, 0, 20);
	menu_e.push(side_menu);
	side_menu.init_bar(language.get("ai_settings", "error_level"), &contents->setting.ai.error_level, contents->setting.ai.error_level, 0, 25);
	menu_e.push(side_menu);
	side_menu.init_bar(language.get("settings", "thread", "thread"), &contents->setting.ai.n_thread, contents->setting.ai.n_thread, 1, 32);
	menu_e.push(side_menu);
	title.push(menu_e);

	menu_e.init_button(language.get("settings", "play", "play"), &contents->dummy);
	side_menu.init_radio(language.get("settings", "play", "human_first"), &contents->setting.player.human_first, contents->setting.player.human_first);
	menu_e.push(side_menu);
	side_menu.init_radio(language.get("settings", "play", "human_second"), &contents->setting.player.ai_first, contents->setting.player.ai_first);
	menu_e.push(side_menu);
	side_menu.init_radio(language.get("settings", "play", "both_ai"), &contents->setting.player.both_ai, contents->setting.player.both_ai);
	menu_e.push(side_menu);
	side_menu.init_radio(language.get("settings", "play", "both_human"), &contents->setting.player.both_human, contents->setting.player.both_human);
	menu_e.push(side_menu);
	title.push(menu_e);

	menu->push(title);

	// display
	title.init(language.get("display", "display"));

	menu_e.init_check(language.get("display", "graph"), &contents->display.graph, contents->display.graph);
	title.push(menu_e);

	menu_e.init_check(language.get("display", "end_popup"), &contents->display.popup_in_end, contents->display.popup_in_end);
	title.push(menu_e);
	menu_e.init_check(language.get("display", "log"), &contents->display.log, contents->display.log);
	title.push(menu_e);

	menu->push(title);
}

Menu create_menu(Font menu_font, Texture checkbox, Menu_contents *contents) {
	Menu menu;
	menu_title title;
	menu_elem menu_e, side_menu, side_side_menu;

	// mode
	title.init(language.get("mode", "mode"));

	menu_e.init_radio(language.get("mode", "entry_mode"), &contents->mode.simple, contents->mode.simple);
	title.push(menu_e);
	menu_e.init_radio(language.get("mode", "professional_mode"), &contents->mode.professional, contents->mode.professional);
	title.push(menu_e);
	menu_e.init_radio(language.get("mode", "serious_game"), &contents->mode.game, contents->mode.game);
	title.push(menu_e);

	menu.push(title);

	// game
	title.init(language.get("play", "game"));

	menu_e.init_button(language.get("play", "new_game"), &contents->game.start);
	title.push(menu_e);
	menu_e.init_button(language.get("play", "analyze"), &contents->game.analyze);
	title.push(menu_e);

	menu.push(title);

	if (contents->mode.simple) {
		create_menu_simple(menu_font, contents, &menu);
	}
	else if (contents->mode.professional) {
		create_menu_professional(menu_font, contents, &menu);
	}
	else {
		create_menu_game(menu_font, contents, &menu);
	}

	// help
	title.init(language.get("help", "help"));

	menu_e.init_button(language.get("help", "how_to_use"), &contents->help.usage);
	title.push(menu_e);
	menu_e.init_button(language.get("help", "bug_report"), &contents->help.report);
	title.push(menu_e);
	menu_e.init_check(language.get("help", "auto_update_check"), &contents->help.update_check, contents->help.update_check);
	title.push(menu_e);
	menu_e.init_button(language.get("help", "license"), &contents->help.license);
	title.push(menu_e);

	menu.push(title);

	// language
	title.init(U"Language");
	for (int i = 0; i < (int)contents->language.name.size(); ++i) {
		menu_e.init_radio(language_name.get(contents->language.name[i]), &contents->language.acts[i], contents->language.acts[i]);
		title.push(menu_e);
	}
	menu.push(title);

	menu.init(0, 0, menu_font, checkbox);

	return menu;
}

void menu_contents_init_preference(Menu_contents* contents, vector<string> languages, bool* language_acts, Preference* preference) {
	contents->dummy = false;

	contents->mode.simple = false;
	contents->mode.professional = false;
	contents->mode.game = false;
	if (preference->int_mode == 0) {
		contents->mode.simple = true;
	}
	else if (preference->int_mode == 1) {
		contents->mode.professional = true;
	}
	else {
		contents->mode.game = true;
	}


	contents->game.start = false;
	contents->game.analyze = false;

	contents->setting.ai.use_book = preference->use_book;
	contents->setting.ai.ai_level = preference->ai_level;
	contents->setting.ai.hint_level = preference->hint_level;
	contents->setting.ai.graph_level = preference->graph_level;
	contents->setting.ai.error_level = preference->error_level;
	contents->setting.ai.n_thread = preference->n_thread_idx;
	contents->setting.player.human_first = false;
	contents->setting.player.ai_first = false;
	contents->setting.player.both_ai = false;
	contents->setting.player.both_human = false;
	if (preference->use_ai_mode == 0) {
		contents->setting.player.human_first = true;
	}
	else if (preference->use_ai_mode == 1) {
		contents->setting.player.ai_first = true;
	}
	else if (preference->use_ai_mode == 2) {
		contents->setting.player.both_ai = true;
	}
	else {
		contents->setting.player.both_human = true;
	}

	contents->display.hint = preference->use_hint_flag;
	contents->display.hint_elem.disc_difference = preference->normal_hint;
	contents->display.hint_elem.n_disc_difference = preference->hint_num;
	contents->display.hint_elem.human_value = preference->human_hint;
	contents->display.hint_elem.umigame_value = preference->umigame_hint;
	contents->display.graph = preference->use_graph_flag;
	contents->display.joseki_on_cell = preference->show_over_joseki;
	contents->display.popup_in_end = preference->show_end_popup;
	contents->display.log = preference->show_log;

	contents->joseki.input = false;
	contents->joseki.reference = false;

	contents->inout.in.record = false;
	contents->inout.in.board = false;
	contents->inout.in.edit_board = false;
	contents->inout.in.game = false;
	contents->inout.out.record = false;
	contents->inout.out.game = false;

	contents->manipulate.stop = false;
	contents->manipulate.resume = false;
	contents->manipulate.go = false;
	contents->manipulate.back = false;
	contents->manipulate.transform_elem.rotate_180 = false;
	contents->manipulate.transform_elem.black_line = false;
	contents->manipulate.transform_elem.white_line = false;

	contents->help.usage = false;
	contents->help.report = false;
	contents->help.update_check = preference->auto_update_check;
	contents->help.license = false;

	contents->language.name = languages;
	bool true_inserted = false;
	for (int i = 0; i < (int)languages.size(); ++i) {
		if (languages[i] == preference->lang_name) {
			true_inserted = true;
			language_acts[i] = true;
		}
		else {
			language_acts[i] = false;
		}
	}
	if (!true_inserted) {
		language_acts[0] = true;
	}
	contents->language.acts = language_acts;
}

void update_preference(Preference* preference, Menu_contents menu_contents) {
	if (menu_contents.mode.simple) {
		preference->int_mode = 0;
	}
	else if (menu_contents.mode.professional) {
		preference->int_mode = 1;
	}
	else if (menu_contents.mode.game) {
		preference->int_mode = 2;
	}
	else {
		preference->int_mode = 0;
	}
	preference->use_book = menu_contents.setting.ai.use_book;
	preference->ai_level = menu_contents.setting.ai.ai_level;
	preference->hint_level = menu_contents.setting.ai.hint_level;
	preference->graph_level = menu_contents.setting.ai.graph_level;
	preference->error_level = menu_contents.setting.ai.error_level;
	if (menu_contents.setting.player.human_first) {
		preference->use_ai_mode = 0;
	}
	else if (menu_contents.setting.player.ai_first) {
		preference->use_ai_mode = 1;
	}
	else if (menu_contents.setting.player.both_ai) {
		preference->use_ai_mode = 2;
	}
	else if (menu_contents.setting.player.both_human) {
		preference->use_ai_mode = 3;
	}
	else {
		preference->use_ai_mode = 0;
	}
	preference->use_hint_flag = menu_contents.display.hint;
	preference->normal_hint = menu_contents.display.hint_elem.disc_difference;
	preference->human_hint = menu_contents.display.hint_elem.human_value;
	preference->umigame_hint = menu_contents.display.hint_elem.umigame_value;
	preference->show_end_popup = menu_contents.display.popup_in_end;
	preference->n_thread_idx = menu_contents.setting.ai.n_thread;
	preference->hint_num = menu_contents.display.hint_elem.n_disc_difference;
	preference->show_log = menu_contents.display.log;
	preference->use_graph_flag = menu_contents.display.graph;
	preference->auto_update_check = menu_contents.help.update_check;
	preference->show_over_joseki = menu_contents.display.joseki_on_cell;
	for (int i = 0; i < (int)menu_contents.language.name.size(); ++i) {
		if (menu_contents.language.acts[i]) {
			preference->lang_name = menu_contents.language.name[i];
		}
	}
}
