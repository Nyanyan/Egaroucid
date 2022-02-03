#pragma once

#include <Siv3D.hpp> // OpenSiv3D v0.6.3
#include <iostream>
#include <string>
#include <fstream>

using namespace std;

class language {
private:
	JSON lang;

public:

	bool init(string file) {
		lang.clear();
		lang = JSON::Load(Unicode::Widen(file));
		if (not lang) {
			cerr << "can't open `language_name`.json" << endl;
			return false;
		}
		return true;
	}

	String get(string v0) {
		String v0s = Unicode::Widen(v0);
		if (lang[v0s].getType() != JSONValueType::String)
			return U"?";
		return lang[v0s].getString();
	}

	String get(string v0, string v1) {
		String v0s = Unicode::Widen(v0);
		String v1s = Unicode::Widen(v1);
		if (lang[v0s][v1s].getType() != JSONValueType::String)
			return U"?";
		return lang[v0s][v1s].getString();
	}

	String get(string v0, string v1, string v2) {
		String v0s = Unicode::Widen(v0);
		String v1s = Unicode::Widen(v1);
		String v2s = Unicode::Widen(v2);
		if (lang[v0s][v1s][v2s].getType() != JSONValueType::String)
			return U"?";
		return lang[v0s][v1s][v2s].getString();
	}
};

language language;

class language_name {
private:
	JSON lang;

public:
	bool init() {
		lang = JSON::Load(U"resources/languages/languages.json");
		if (not lang) {
			cerr << "can't open languages.json" << endl;
			return false;
		}
		return true;
	}

	String get(string v0) {
		String v0s = Unicode::Widen(v0);
		if (lang[v0s].getType() != JSONValueType::String)
			return U"?";
		return lang[v0s].getString();
	}
};

language_name language_name;
