/*
    Egaroucid Project

    @file load.hpp
        Loading scene
    @date 2021-2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

#pragma once
#include <iostream>
#include <future>
#include "./../engine/engine_all.hpp"
#include "function/function_all.hpp"

#ifdef __APPLE__
#include <sys/types.h>
#include <sys/sysctl.h>
#include <mach/mach.h>

struct MacMemoryStatusEx {
    uint64_t totalPhysicalMemory;
    uint64_t freeMemory;
    uint64_t usedMemory;
    uint64_t activeMemory;
    uint64_t inactiveMemory;
    uint64_t wiredMemory;

    MacMemoryStatusEx() {
        totalPhysicalMemory = 0;
        freeMemory = 0;
        usedMemory = 0;
        activeMemory = 0;
        inactiveMemory = 0;
        wiredMemory = 0;
    }

    void updateMemoryStatus() {
        size_t length = sizeof(totalPhysicalMemory);
        sysctlbyname("hw.memsize", &totalPhysicalMemory, &length, nullptr, 0);
        
        mach_msg_type_number_t count = HOST_VM_INFO_COUNT;
        vm_statistics64_data_t vm_stat;
        if (host_statistics64(mach_host_self(), HOST_VM_INFO, (host_info64_t)&vm_stat, &count) == KERN_SUCCESS) {
            freeMemory = vm_stat.free_count * vm_page_size;
            activeMemory = vm_stat.active_count * vm_page_size;
            inactiveMemory = vm_stat.inactive_count * vm_page_size;
            wiredMemory = vm_stat.wire_count * vm_page_size;
            usedMemory = (activeMemory + inactiveMemory + wiredMemory);
        }
    }
};
#else // Windows
#include <windows.h>
#include <shlwapi.h>
#endif


void init_shortcut_keys(const Directories* directories) {
    String file = U"{}shortcut_key.json"_fmt(Unicode::Widen(directories->appdata_dir));
    shortcut_keys.init(file);
}

int check_update(const Directories* directories, String *new_version) {
    const FilePath version_save_path = U"{}version.txt"_fmt(Unicode::Widen(directories->appdata_dir));
    AsyncHTTPTask task = SimpleHTTP::SaveAsync(VERSION_URL, version_save_path);
    uint64_t strt = tim();
    while (tim() - strt < 1000) { // timeout 1000 ms
        if (task.isReady()) {
            if (task.getResponse().isOK()) {
                TextReader reader(version_save_path);
                if (reader) {
                    reader.readLine(*new_version);
                    if (EGAROUCID_NUM_VERSION != *new_version) { // new version found
                        return UPDATE_CHECK_UPDATE_FOUND;
                    } else {
                        return UPDATE_CHECK_ALREADY_UPDATED; // already latest
                    }
                }
            }
        }
    }
    if (task.getStatus() != HTTPAsyncStatus::Succeeded) { // cancel task
        task.cancel();
    }
    return UPDATE_CHECK_FAILED; // update check failed
}

int init_resources_load(Resources* resources, Settings* settings, bool *stop_loading) {
    // texture
    std::cerr << "loading textures (2)" << std::endl;
    Texture checkbox(Unicode::Widen(EXE_DIRECTORY_PATH + "resources/img/checked.png"), TextureDesc::Mipped);
    Texture unchecked(Unicode::Widen(EXE_DIRECTORY_PATH + "resources/img/unchecked.png"), TextureDesc::Mipped);
    Texture laser_pointer(Unicode::Widen(EXE_DIRECTORY_PATH + "resources/img/laser_pointer.png"), TextureDesc::Mipped);
    Texture cross(Unicode::Widen(EXE_DIRECTORY_PATH + "resources/img/cross.png"), TextureDesc::Mipped);
    Texture flip_horizontal(Unicode::Widen(EXE_DIRECTORY_PATH + "resources/img/flip_horizontal.png"), TextureDesc::Mipped);
    Texture flip_vertical(Unicode::Widen(EXE_DIRECTORY_PATH + "resources/img/flip_vertical.png"), TextureDesc::Mipped);
    Texture rotate_cw(Unicode::Widen(EXE_DIRECTORY_PATH + "resources/img/rotate_cw.png"), TextureDesc::Mipped);
    Texture rotate_ccw(Unicode::Widen(EXE_DIRECTORY_PATH + "resources/img/rotate_ccw.png"), TextureDesc::Mipped);
    Texture rotate_180(Unicode::Widen(EXE_DIRECTORY_PATH + "resources/img/rotate_180.png"), TextureDesc::Mipped);
    Texture mirror_white_line(Unicode::Widen(EXE_DIRECTORY_PATH + "resources/img/mirror_white_line.png"), TextureDesc::Mipped);
    Texture mirror_black_line(Unicode::Widen(EXE_DIRECTORY_PATH + "resources/img/mirror_black_line.png"), TextureDesc::Mipped);
    Texture check(Unicode::Widen(EXE_DIRECTORY_PATH + "resources/img/check.png"), TextureDesc::Mipped);
    Texture folder(Unicode::Widen(EXE_DIRECTORY_PATH + "resources/img/folder.png"), TextureDesc::Mipped);
    Texture arrow_left(Unicode::Widen(EXE_DIRECTORY_PATH + "resources/img/arrow_left.png"), TextureDesc::Mipped);
    Texture pencil(Unicode::Widen(EXE_DIRECTORY_PATH + "resources/img/pencil.png"), TextureDesc::Mipped);
    if (checkbox.isEmpty() || unchecked.isEmpty() || laser_pointer.isEmpty() || cross.isEmpty() ||
        flip_horizontal.isEmpty() || flip_vertical.isEmpty() || rotate_cw.isEmpty() || rotate_ccw.isEmpty() ||
        rotate_180.isEmpty() || mirror_white_line.isEmpty() || mirror_black_line.isEmpty() ||
        check.isEmpty() || folder.isEmpty() || arrow_left.isEmpty() || pencil.isEmpty()
    ) {
        return ERR_LOAD_TEXTURE_NOT_LOADED;
    }
    resources->checkbox = checkbox;
    resources->unchecked = unchecked;
    resources->laser_pointer = laser_pointer;
    resources->cross = cross;
    resources->flip_horizontal = flip_horizontal;
    resources->flip_vertical = flip_vertical;
    resources->rotate_cw = rotate_cw;
    resources->rotate_ccw = rotate_ccw;
    resources->rotate_180 = rotate_180;
    resources->mirror_white_line = mirror_white_line;
    resources->mirror_black_line = mirror_black_line;
    resources->check = check;
    resources->folder = folder;
    resources->arrow_left = arrow_left;
    resources->pencil = pencil;

    if (*stop_loading) {
        return ERR_LOAD_TERMINATED;
    }

    // language image
    std::cerr << "loading language images" << std::endl;
    std::vector<Texture> lang_img;
    for (int i = 0; i < (int)resources->language_names.size() && !(*stop_loading); ++i) {
        Texture limg(Unicode::Widen(EXE_DIRECTORY_PATH + "resources/languages/" + resources->language_names[i] + ".png"), TextureDesc::Mipped);
        if (limg.isEmpty()) {
            return ERR_LOAD_TEXTURE_NOT_LOADED;
        }
        lang_img.emplace_back(limg);
    }
    resources->lang_img = lang_img;

    if (*stop_loading) {
        return ERR_LOAD_TERMINATED;
    }

    // opening
    std::cerr << "loading openings" << std::endl;
    if (!opening_init(settings->lang_name)) {
        std::cerr << "opening file not found. use alternative opening file" << std::endl;
        if (!opening_init(DEFAULT_OPENING_LANG_NAME)) {
            std::cerr << "opening file not found" << std::endl;
            return ERR_LOAD_OPENING_NOT_LOADED;
        }
    }

    if (*stop_loading) {
        return ERR_LOAD_TERMINATED;
    }

    // license
    std::cerr << "loading license" << std::endl;
    TextReader reader{Unicode::Widen(EXE_DIRECTORY_PATH + "LICENSE")};
    if (not reader) {
        return ERR_LOAD_LICENSE_FILE_NOT_LOADED;
    }
    String copyright = Unicode::Widen("(C) " + (std::string)EGAROUCID_DATE + " " + (std::string)EGAROUCID_AUTHOR);
    String license = reader.readAll();
    LicenseManager::AddLicense({
        .title = U"Egaroucid",
        .copyright = copyright,
        .text = license
    });

    return ERR_OK;
}

int init_ai(Settings* settings, const Directories* directories, bool *stop_loading) {
    thread_pool.resize(settings->n_threads - 1);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    std::cerr << "there are " << thread_pool.size() << " additional threads" << std::endl;
    bit_init();
    mobility_init();
    flip_init();
    last_flip_init();
    endsearch_init();
#if USE_MPC_PRE_CALCULATION
    mpc_init();
#endif
    move_ordering_init();
#ifndef __APPLE__
    MEMORYSTATUSEX msex = { sizeof(MEMORYSTATUSEX) };
    GlobalMemoryStatusEx( &msex );
    double free_mb = (double)msex.ullAvailPhys / 1024 / 1024;
#else // Windows
    MacMemoryStatusEx msex;
    msex.updateMemoryStatus();
    double free_mb = static_cast<double>(msex.freeMemory) / 1024 / 1024;
#endif
    double size_mb = (double)sizeof(Hash_node) / 1024 / 1024 * hash_sizes[MAX_HASH_LEVEL];
    std::cerr << "memory " << free_mb << " " << size_mb << std::endl;
    while (free_mb <= size_mb && MAX_HASH_LEVEL > 26) {
        --MAX_HASH_LEVEL;
        size_mb = (double)sizeof(Hash_node) / 1024 / 1024 * hash_sizes[MAX_HASH_LEVEL];
    }
    settings->hash_level = std::min(settings->hash_level, MAX_HASH_LEVEL);
    std::cerr << "max hash level " << MAX_HASH_LEVEL << std::endl;
#if USE_CHANGEABLE_HASH_LEVEL
    if (!hash_resize(DEFAULT_HASH_LEVEL, settings->hash_level, true)) {
        std::cerr << "hash resize failed. use default setting" << std::endl;
        settings->hash_level = DEFAULT_HASH_LEVEL;
    }
#else
    hash_tt_init(true);
#endif
    stability_init();
    if (!evaluate_init(directories->eval_file, directories->eval_mo_end_file, true)) {
        return ERR_LOAD_EVAL_FILE_NOT_IMPORTED;
    }
    if (!book_init(settings->book_file, true, stop_loading)) {
        return ERR_LOAD_BOOK_FILE_NOT_IMPORTED;
    }
    std::string ext = get_extension(settings->book_file);
    if (ext == "egbk") {
        settings->book_file += "2"; // force book version 3
        book.save_egbk3(settings->book_file, settings->book_file + ".bak");
    } else if (ext == "egbk2") {
        settings->book_file[settings->book_file.size() - 1] = '3'; // force book version 3
        book.save_egbk3(settings->book_file, settings->book_file + ".bak");
    }
    return ERR_OK;
}

// Helper function to load openings from CSV files in forced_openings directory
void load_openings_from_csv_files(Forced_openings* forced_openings, const std::string& forced_openings_dir) {
    String base_dir = Unicode::Widen(forced_openings_dir);
    String settings_path = base_dir + U"/settings.txt";
    
    // Load enabled states from settings file
    std::unordered_map<String, bool> enabled_map;
    if (FileSystem::Exists(settings_path)) {
        TextReader reader(settings_path);
        if (reader) {
            String line;
            while (reader.readLine(line)) {
                auto parts = line.split(U'\t');
                if (parts.size() >= 2) {
                    String filename = parts[0];
                    bool enabled = ParseOr<bool>(parts[1], true);
                    enabled_map[filename] = enabled;
                }
            }
        }
    }
    
    // Enumerate all CSV files
    Array<FilePath> list = FileSystem::DirectoryContents(base_dir);
    std::vector<String> csv_filenames;
    for (const auto& path : list) {
        if (FileSystem::IsFile(path) && path.ends_with(U".csv")) {
            String filename = FileSystem::FileName(path);
            csv_filenames.emplace_back(filename);
        }
    }
    std::sort(csv_filenames.begin(), csv_filenames.end());
    
    // Load openings from enabled CSV files
    for (const auto& filename : csv_filenames) {
        bool enabled = enabled_map.count(filename) ? enabled_map[filename] : true;
        if (enabled) {
            String csv_path = base_dir + U"/" + filename;
            const CSV csv{ csv_path };
            if (csv) {
                for (size_t row = 0; row < csv.rows(); ++row) {
                    if (csv[row].size() >= 2) {
                        String transcript = csv[row][0];
                        double weight = ParseOr<double>(csv[row][1], 1.0);
                        forced_openings->openings.emplace_back(std::make_pair(transcript.narrow(), weight));
                    }
                }
            }
        }
    }
}

int load_app(Directories* directories, Resources* resources, Settings* settings, Forced_openings *forced_openings, Menu_elements *menu_elements, bool* update_found, String *new_version, bool *stop_loading) {
    // auto update check
    if (settings->auto_update_check) {
        if (check_update(directories, new_version) == UPDATE_CHECK_UPDATE_FOUND) {
            *update_found = true;
        }
    }
    // resources
    int code = init_resources_load(resources, settings, stop_loading);
    if (code == ERR_OK) { // when resources initialized
        // shortcut keys
        init_shortcut_keys(directories);
        // settings -> menu elements
        menu_elements->init(settings, resources);
        // AI
        code = init_ai(settings, directories, stop_loading);
        // forced openings for AI
        std::string forced_openings_dir = directories->document_dir + "/forced_openings";
        String forced_openings_dir_wide = Unicode::Widen(forced_openings_dir);
        
        // Create forced_openings directory if it doesn't exist
        if (!FileSystem::Exists(forced_openings_dir_wide)) {
            FileSystem::CreateDirectories(forced_openings_dir_wide);
        }
        
        // Migrate from old forced_openings.txt in appdata to default.csv in document directory
        std::string old_forced_openings_file = directories->appdata_dir + "/forced_openings.txt";
        String old_file_wide = Unicode::Widen(old_forced_openings_file);
        String new_csv_path = forced_openings_dir_wide + U"/summary.csv";
        
        if (FileSystem::Exists(old_file_wide)) {
            // Migrate old format to new CSV format (summary.csv) in document directory
            std::ifstream ifs(old_forced_openings_file);
            if (ifs) {
                CSV csv;
                // Load existing CSV if it exists (append mode)
                if (FileSystem::Exists(new_csv_path)) {
                    csv = CSV(new_csv_path);
                }
                
                std::string line;
                while (std::getline(ifs, line)) {
                    std::istringstream iss(line);
                    std::string transcript, weight_str;
                    iss >> transcript >> weight_str;
                    double weight;
                    try {
                        weight = stoi(weight_str);
                        if (is_valid_transcript(transcript)) {
                            csv.write(Unicode::Widen(transcript));
                            csv.write(Format(weight));
                            csv.write(Format(true));
                            csv.newLine();
                        }
                    } catch (const std::exception& e) {
                        // Skip invalid lines
                    }
                }
                csv.save(new_csv_path);
                ifs.close();
            }
            // Delete old file after migration
            FileSystem::Remove(old_file_wide);
            std::cerr << "Migrated forced_openings.txt from appdata to summary.csv in document directory" << std::endl;
        }
        
        // Load openings from CSV files
        forced_openings->openings.clear();
        load_openings_from_csv_files(forced_openings, forced_openings_dir);
        forced_openings->init();
    }
    return code;
}

class Load : public App::Scene {
private:
    bool load_failed;
    int load_code;
    String tips;
    bool update_found;
    std::future<int> load_future;
    Button skip_button;
    Button update_button;
    Button book_ignore_button;
    String new_version;
    bool stop_loading;

public:
    Load(const InitData& init) : IScene{ init } {
        skip_button.init(GO_BACK_BUTTON_BACK_SX, GO_BACK_BUTTON_SY, GO_BACK_BUTTON_WIDTH, GO_BACK_BUTTON_HEIGHT, GO_BACK_BUTTON_RADIUS, language.get("help", "skip"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
        update_button.init(GO_BACK_BUTTON_GO_SX, GO_BACK_BUTTON_SY, GO_BACK_BUTTON_WIDTH, GO_BACK_BUTTON_HEIGHT, GO_BACK_BUTTON_RADIUS, language.get("help", "download"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
        book_ignore_button.init(BACK_BUTTON_SX, BACK_BUTTON_SY, BACK_BUTTON_WIDTH, BACK_BUTTON_HEIGHT, BACK_BUTTON_RADIUS, language.get("loading", "launch"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
        load_failed = false;
        tips = language.get_random("tips", "tips");
        update_found = false;
        stop_loading = false;
        load_future = std::async(std::launch::async, load_app, &getData().directories, &getData().resources, &getData().settings, &getData().forced_openings, &getData().menu_elements, &update_found, &new_version, &stop_loading);
    }

    void update() override {
        if (System::GetUserActions() & UserAction::CloseButtonClicked) {
            stop_loading = true;
            if (load_future.valid()) {
                load_future.get();
            }
            changeScene(U"Close", SCENE_FADE_TIME);
            return;
        }
        Scene::SetBackground(getData().colors.green);
        if (update_found) {
            const int icon_width = (LEFT_RIGHT - LEFT_LEFT) / 2;
            getData().resources.icon.scaled((double)icon_width / getData().resources.icon.width()).draw(X_CENTER - icon_width / 2, 20);
            getData().resources.logo.scaled((double)icon_width / getData().resources.logo.width()).draw(X_CENTER - icon_width / 2, 20 + icon_width);
            int sy = 20 + icon_width + 50;
            getData().fonts.font(language.get("help", "new_version_available")).draw(25, Arg::topCenter(X_CENTER, sy), getData().colors.white);
            sy += 35;
            getData().fonts.font(language.get("help", "download?")).draw(25, Arg::topCenter(X_CENTER, sy), getData().colors.white);
            skip_button.draw();
            update_button.draw();
            if (skip_button.clicked() || KeyEscape.down()) {
                update_found = false;
            }
            if (update_button.clicked() || KeyEnter.down()) {
                if (language.get("lang_name") == U"日本語") {
                    System::LaunchBrowser(U"https://www.egaroucid.nyanyan.dev/ja/download/");
                } else {
                    System::LaunchBrowser(U"https://www.egaroucid.nyanyan.dev/en/download/");
                }
                changeScene(U"Close", SCENE_FADE_TIME);
                return;
            }
        } else {
            const int icon_width = (LEFT_RIGHT - LEFT_LEFT);
            getData().resources.icon.scaled((double)icon_width / getData().resources.icon.width()).draw(LEFT_LEFT, Y_CENTER - icon_width / 2);
            getData().resources.logo.scaled((double)icon_width * 0.8 / getData().resources.logo.width()).draw(RIGHT_LEFT, Y_CENTER - 40);
            if (load_future.valid()) {
                if (load_future.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                    load_code = load_future.get();
                    if (load_code == ERR_OK) {
                        std::cerr << "loaded" << std::endl;
                        getData().window_state.loading = false;
                        // changeScene(U"Main_scene", SCENE_FADE_TIME);
                        changeScene(U"Main_scene", 0);
                    } else {
                        load_failed = true;
                    }
                }
            }
            if (load_failed) {
                if (load_code == ERR_LOAD_BOOK_FILE_NOT_IMPORTED) {
                    getData().fonts.font(language.get("loading", "book_failed")).draw(20, RIGHT_LEFT, Y_CENTER + 50, getData().colors.white);
                    book_ignore_button.draw();
                    if (book_ignore_button.clicked()) {
                        std::cerr << "loaded" << std::endl;
                        getData().window_state.loading = false;
                        // changeScene(U"Main_scene", SCENE_FADE_TIME);
                        changeScene(U"Main_scene", 0);
                    }
                } else {
                    String err_str = language.get("loading", "load_failed") + U"\nERROR CODE: " + Format(load_code);
                    getData().fonts.font(err_str).draw(20, RIGHT_LEFT, Y_CENTER + 50, getData().colors.white);
                    if (System::GetUserActions() & UserAction::CloseButtonClicked) {
                        System::Exit();
                    }
                }
            } else {
                getData().fonts.font(language.get("loading", "loading")).draw(50, RIGHT_LEFT, Y_CENTER + 40, getData().colors.white);
                getData().fonts.font(language.get("tips", "do_you_know")).draw(20, RIGHT_LEFT, Y_CENTER + 110, getData().colors.white);
                getData().fonts.font(tips).draw(15, RIGHT_LEFT, Y_CENTER + 140, getData().colors.white);
            }
        }
    }

    void draw() const override {

    }
};