#pragma once

#include <Siv3D.hpp> // OpenSiv3D v0.6.3
#include <iostream>
#include <string>
#include <fstream>
#include "rapidjson/document.h"

using namespace std;

class language {
private:
	rapidjson::Document lang;
	bool ready;

public:
	bool is_ready() {
		return ready;
	}

	bool init(string file) {
		ready = false;
		ifstream ifs(file);
		if (ifs.fail()) {
			cerr << "can't open language.json" << endl;
			return false;
		}
		string raw_data((istreambuf_iterator<char>(ifs)), istreambuf_iterator<char>());
		lang.Parse(raw_data.c_str());
		if (lang.HasParseError()) {
			cerr << "can't parce language.json" << endl;
			return false;
		}
		cerr << "language name: " << lang["lang_name"].GetString() << endl;
		ready = true;
		return true;
	}

	String get(const char v0[]) {
		if (!lang.HasMember(v0))
			return U"?";
		if (lang[v0].IsObject())
			return U"?";
		return Unicode::Widen(lang[v0].GetString());
	}

	String get(const char v0[], const char v1[]) {
		//cerr << v0 << " " << v1 << endl;
		if (!lang.HasMember(v0))
			return U"?";
		if (!lang[v0].IsObject())
			return U"?";
		if (!lang[v0].HasMember(v1))
			return U"?";
		if (lang[v0][v1].IsObject())
			return U"?";
		return Unicode::Widen(lang[v0][v1].GetString());
	}

	String get(const char v0[], const char v1[], const char v2[]) {
		//cerr << v0 << " " << v1 << " " << v2 << endl;
		if (!lang.HasMember(v0))
			return U"?";
		if (!lang[v0].IsObject())
			return U"?";
		if (!lang[v0].HasMember(v1))
			return U"?";
		if (!lang[v0][v1].IsObject())
			return U"?";
		if (!lang[v0][v1].HasMember(v2))
			return U"?";
		if (lang[v0][v1][v2].IsObject())
			return U"?";
		return Unicode::Widen(lang[v0][v1][v2].GetString());
	}
};

language language;
