#pragma once
#include <iostream>

struct Preference {
	int int_mode;
	bool use_book;
	int ai_level;
	int hint_level;
	int graph_level;
	int error_level;
	int use_ai_mode;
	bool use_hint_flag;
	bool normal_hint;
	bool human_hint;
	bool umigame_hint;
	bool show_end_popup;
	int n_thread_idx;
	int hint_num;
	bool show_log;
	bool use_graph_flag;
	bool auto_update_check;
	bool show_over_joseki;
	string lang_name;
	string book_file;
};

int import_int(TextReader* reader) {
	String line;
	if (reader->readLine(line)) {
		try {
			return Parse<int32>(line);
		}
		catch (const ParseError& e) {
			return -INF;
		}
	}
	else {
		return -INF;
	}
}

string import_str(TextReader* reader) {
	String line;
	if (reader->readLine(line)) {
		return line.narrow();
	}
	else {
		return "undefined";
	}
}

bool import_preference_elem_with_check(TextReader *reader, int* elem) {
	*elem = import_int(reader);
	return *elem != -INF;
}

bool import_preference_elem_with_check(TextReader* reader, bool* elem) {
	int int_elem = import_int(reader);
	if (int_elem != -INF) {
		*elem = (bool)int_elem;
	}
	return int_elem != -INF;
}

bool import_preference_elem_with_check(TextReader* reader, string* elem) {
	*elem = import_str(reader);
	return *elem != "undefined";
}

bool import_preference(Preference* preference) {
	String appdata_dir = FileSystem::GetFolderPath(SpecialFolder::LocalAppData);
	TextReader reader(U"{}Egaroucid/setting.txt"_fmt(appdata_dir));
	if (!reader) {
		return false;
	}
	bool imported = true;
	imported &= import_preference_elem_with_check(&reader, &preference->int_mode);
	imported &= import_preference_elem_with_check(&reader, &preference->use_book);
	imported &= import_preference_elem_with_check(&reader, &preference->ai_level);
	imported &= import_preference_elem_with_check(&reader, &preference->hint_level);
	imported &= import_preference_elem_with_check(&reader, &preference->graph_level);
	imported &= import_preference_elem_with_check(&reader, &preference->error_level);
	imported &= import_preference_elem_with_check(&reader, &preference->use_ai_mode);
	imported &= import_preference_elem_with_check(&reader, &preference->use_hint_flag);
	imported &= import_preference_elem_with_check(&reader, &preference->normal_hint);
	imported &= import_preference_elem_with_check(&reader, &preference->human_hint);
	imported &= import_preference_elem_with_check(&reader, &preference->umigame_hint);
	imported &= import_preference_elem_with_check(&reader, &preference->show_end_popup);
	imported &= import_preference_elem_with_check(&reader, &preference->n_thread_idx);
	imported &= import_preference_elem_with_check(&reader, &preference->hint_num);
	imported &= import_preference_elem_with_check(&reader, &preference->show_log);
	imported &= import_preference_elem_with_check(&reader, &preference->use_graph_flag);
	imported &= import_preference_elem_with_check(&reader, &preference->auto_update_check);
	imported &= import_preference_elem_with_check(&reader, &preference->show_over_joseki);
	imported &= import_preference_elem_with_check(&reader, &preference->lang_name);
	imported &= import_preference_elem_with_check(&reader, &preference->book_file);
	String line;
	if (reader.readLine(line)) {
		return false;
	}
	return imported;
}

void export_preference(Preference* preference) {
	String appdata_dir = FileSystem::GetFolderPath(SpecialFolder::LocalAppData);
	TextWriter writer(U"{}Egaroucid/setting.txt"_fmt(appdata_dir));
	if (writer) {
		writer.writeln(preference->int_mode);
		writer.writeln((int)preference->use_book);
		writer.writeln(preference->ai_level);
		writer.writeln(preference->hint_level);
		writer.writeln(preference->graph_level);
		writer.writeln(preference->error_level);
		writer.writeln(preference->use_ai_mode);
		writer.writeln((int)preference->use_hint_flag);
		writer.writeln((int)preference->normal_hint);
		writer.writeln((int)preference->human_hint);
		writer.writeln((int)preference->umigame_hint);
		writer.writeln((int)preference->show_end_popup);
		writer.writeln(preference->n_thread_idx);
		writer.writeln(preference->hint_num);
		writer.writeln((int)preference->show_log);
		writer.writeln((int)preference->use_graph_flag);
		writer.writeln((int)preference->auto_update_check);
		writer.writeln((int)preference->show_over_joseki);
		writer.writeln(Unicode::Widen(preference->lang_name));
		writer.writeln(Unicode::Widen(preference->book_file));
	}
}

void set_default_preference(Preference* preference, string document_dir) {
	preference->int_mode = 0;
	preference->use_book = true;
	preference->ai_level = 15;
	preference->hint_level = 15;
	preference->graph_level = 15;
	preference->error_level = 0;
	preference->use_ai_mode = 0;
	preference->use_hint_flag = true;
	preference->normal_hint = true;
	preference->human_hint = false;
	preference->umigame_hint = false;
	preference->show_end_popup = true;
	preference->n_thread_idx = min(32, (int)thread::hardware_concurrency());
	preference->hint_num = HW2;
	preference->show_log = true;
	preference->use_graph_flag = true;
	preference->auto_update_check = true;
	preference->show_over_joseki = true;
	preference->lang_name = "japanese";
	preference->book_file = document_dir + "Egaroucid/book.egbk";
}
