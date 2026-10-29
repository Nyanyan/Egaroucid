/*
    Issue #652: opt-in checks inside the real GUI runtime.
    Included after gui_all.hpp. No OS input injection or private user data.
*/
#pragma once
#include <set>
#include <stdexcept>
#include <type_traits>

namespace operation_check {

// Desktop tests drive the normal OS controls. This observer only records state;
// it never changes scenes, settings, game history, or input events.
inline bool desktop_observation_enabled = false;
inline String observed_scene;
inline String observed_status;
inline int observed_list_size = -1;
inline int observed_ready_details = -1;

#if SIV3D_PLATFORM(WINDOWS)
struct Mouse_message {
    ULONGLONG ticks;
    UINT message;
    int x, y;
    WPARAM flags;
};
inline std::mutex mouse_message_mutex;
inline std::vector<Mouse_message> mouse_messages;
inline void observe_mouse_message(HWND, UINT message, WPARAM flags, LPARAM position) {
    if (message != WM_LBUTTONDOWN && message != WM_LBUTTONUP &&
        !(message == WM_MOUSEMOVE && (flags & MK_LBUTTON))) return;
    std::lock_guard<std::mutex> lock(mouse_message_mutex);
    if (mouse_messages.size() < 4096) mouse_messages.push_back({GetTickCount64(), message,
        static_cast<short>(LOWORD(position)), static_cast<short>(HIWORD(position)), flags});
}

inline void write_mouse_trace(int frame) {
    std::vector<Mouse_message> messages;
    {
        std::lock_guard<std::mutex> lock(mouse_message_mutex);
        messages.swap(mouse_messages);
    }
    if (messages.empty() && !MouseL.down() && !MouseL.pressed() && !MouseL.up()) return;
    std::ofstream stream("desktop-mouse.tsv", std::ios::app);
    for (const auto& message : messages) {
        stream << "message\t" << frame << '\t' << message.ticks << '\t' << observed_scene.narrow()
            << '\t' << message.message << '\t' << message.x << '\t' << message.y << '\t' << message.flags << '\n';
    }
    const auto cursor = Cursor::Pos();
    stream << "frame\t" << frame << '\t' << GetTickCount64() << '\t' << observed_scene.narrow()
        << '\t' << cursor.x << '\t' << cursor.y << '\t' << MouseL.down() << '\t' << MouseL.pressed() << '\t' << MouseL.up() << '\n';
}
#endif

template<class SceneType>
class Observed_scene : public SceneType {
    String scene_name;
public:
    Observed_scene(const App::Scene::InitData& init) : SceneType(init), scene_name(init.state) {
        observed_scene = init.state;
        observed_status.clear();
        observed_list_size = observed_ready_details = -1;
    }
    void update() override {
        SceneType::update();
        if (observed_scene != scene_name) return;
        if constexpr (std::is_same_v<SceneType, Import_book>) {
            observed_status = (this->book_deleting || this->book_importing || this->book_restoring) ? U"running" :
                this->done ? (this->failed ? U"failed" : U"done") : U"ready";
        } else if constexpr (std::is_same_v<SceneType, Merge_book>) {
            observed_status = this->imported ? (this->failed ? U"failed" : U"done") : this->importing ? U"running" : U"ready";
        } else if constexpr (std::is_same_v<SceneType, Enhance_book> || std::is_same_v<SceneType, Store_book> ||
            std::is_same_v<SceneType, Fix_book> || std::is_same_v<SceneType, Fix_book_edax> ||
            std::is_same_v<SceneType, Reduce_book> || std::is_same_v<SceneType, Leaf_recalculate_book> ||
            std::is_same_v<SceneType, N_lines_recalculate_book>) {
            observed_status = this->before_start ? U"ready" : this->done ? U"done" : U"running";
        } else if constexpr (std::is_same_v<SceneType, Update_check>) {
            observed_status = this->update_status == UPDATE_CHECK_NONE ? U"running" :
                this->update_status == UPDATE_CHECK_ALREADY_UPDATED ? U"already-latest" :
                this->update_status == UPDATE_CHECK_UPDATE_FOUND ? U"update-found" : U"failed";
        } else if constexpr (std::is_same_v<SceneType, Import_othello_quest>) {
            observed_status = this->list_loading ? U"loading-list" : this->status_message;
            observed_list_size = static_cast<int>(this->games.size());
            observed_ready_details = 0;
            for (const auto& game : this->games) {
                if (game.detail_state == SceneType::Detail_state::Ready) ++observed_ready_details;
            }
        }
    }
};

class Observed_app : public App {
public:
    template<class SceneType>
    Observed_app& add(const String& name) {
        if (desktop_observation_enabled) App::add<Observed_scene<SceneType>>(name);
        else App::add<SceneType>(name);
        return *this;
    }
};

inline void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

class Gui_probe {
    bool enabled = false;
    bool started = false;
    bool closing = false;
    bool capturing = false;
    int frames = 0;
    size_t scene_index = 0;
    String case_name;
    String token;
    JSON result;
    int desktop_frames = 0;
    const Array<String> scenes = {U"Main_scene", U"Import_text", U"Edit_board",
        U"Game_information_scene", U"Export_game", U"Board_image",
        U"Opening_setting", U"Shortcut_key_setting", U"Game_library"};

    void write_result(bool passed, const String& error = U"") {
        result[U"protocol"] = 1;
        result[U"case"] = case_name;
        result[U"token"] = token;
        result[U"version"] = EGAROUCID_VERSION;
        result[U"portable"] = GUI_PORTABLE_MODE;
        result[U"passed"] = passed;
        result[U"error"] = error;
        require(result.save(U"native-result.json"), "Could not save native result");
    }

    void finish(App& manager, bool passed, const String& error = U"") {
        write_result(passed, error);
        closing = true;
        // Use the normal shutdown path, including settings/profile/shortcut saves.
        require(manager.changeScene(U"Close", 0), "Could not enter Close scene");
    }

    static std::vector<History_elem> sample_history() {
        History_elem initial;
        bool failed = false;
        auto history = import_transcript_processing({initial}, initial, "f5d6c3d3", &failed);
        require(!failed && history.size() == 5, "Valid transcript import failed");
        return history;
    }

    void check_case(Common_resources& data) {
        require(!data.resources.icon.isEmpty() && !data.resources.logo.isEmpty(), "Textures not loaded");
        require(!data.fonts.font.isEmpty(), "Font not loaded");
        require(data.history_elem.board.n_discs() == 4, "Initial board has wrong disc count");
        require(pop_count_ull(data.history_elem.board.get_legal()) == 4, "Initial legal moves differ");
        require(data.graph_resources.nodes[0].size() == 1, "Initial graph missing");
        result[U"language"] = Unicode::Widen(data.settings.lang_name);

        if (case_name == U"startup" || case_name == U"scenes") {
            return;
        } else if (case_name == U"settings-defaults" || case_name == U"settings-invalid") {
            const auto& settings = data.settings;
            require(settings.board_image_mark == (case_name == U"settings-invalid" ? BOARD_IMAGE_MARK_MAX : 0), "Mark default/clamp differs");
            require(settings.board_image_coordinate == 0 && settings.board_image_color == 0, "Coordinate/color default differs");
            require(settings.board_image_outer_frame_color == (case_name == U"settings-invalid" ? 1 : 0), "Frame default/clamp differs");
        } else if (case_name == U"settings-restored") {
            require(data.settings.board_image_mark == BOARD_IMAGE_FLIP && data.settings.board_image_coordinate == 1 &&
                data.settings.board_image_color == 1 && data.settings.board_image_outer_frame_color == 1,
                "Restart did not restore editor settings");
            require(!data.menu_elements.show_legal && data.menu_elements.timer_time_limit_min == 7,
                "Restart did not restore display settings");
        } else if (case_name == U"settings") {
            data.settings.board_image_mark = BOARD_IMAGE_FLIP;
            data.settings.board_image_coordinate = 1;
            data.settings.board_image_color = 1;
            data.settings.board_image_outer_frame_color = 1;
            data.menu_elements.show_legal = false;
            data.menu_elements.timer_time_limit_min = 7;
            save_app_data(data.menu_elements, &data.settings, data.directories,
                data.user_settings, data.book_information, data.window_state);
            Settings loaded;
            init_settings(&data.directories, &data.resources, &loaded);
            require(loaded.board_image_mark == BOARD_IMAGE_FLIP && loaded.board_image_coordinate == 1 &&
                loaded.board_image_color == 1 && loaded.board_image_outer_frame_color == 1, "Board image settings roundtrip differs");
            require(!loaded.show_legal && loaded.timer_time_limit_min == 7, "Display settings roundtrip differs");
        } else if (case_name == U"transcript") {
            auto history = sample_history();
            Graph_resources graph;
            graph.nodes[0] = history;
            require(get_transcript(graph, history.back()) == "f5d6c3d3", "Transcript export differs");
            bool failed = false;
            auto imported = import_any_format_processing("f5d6c3d3", {History_elem{}}, History_elem{}, &failed);
            require(!failed && imported.history.back().board == history.back().board, "Automatic format detection differs");
            import_transcript_processing({History_elem{}}, History_elem{}, "a1", &failed);
            require(failed, "Illegal transcript accepted");
            const std::string board_text = history.back().board.to_str(history.back().player);
            auto board = import_board_processing(board_text, &failed);
            require(!failed && board.first == history.back().board && board.second == history.back().player, "Board text roundtrip differs");
            result[U"board"] = Unicode::Widen(board_text);
        } else if (case_name == U"profiles") {
            AI_profile_values ai_values = to_ai_profile_values(data.menu_elements);
            ai_values.level = 3;
            ai_values.n_threads = 2;
            ai_values.max_loss_by_move[10] = 4;
            const String name = U"operation check 日本語 中文 한글";
            require(save_ai_profile_values(U"ai-test.json", ai_values, name), "AI profile save failed");
            AI_profile_values loaded_ai;
            String loaded_name;
            require(load_ai_profile_values(U"ai-test.json", &loaded_ai, &loaded_name), "AI profile load failed");
            require(equals_ai_profile_values(ai_values, loaded_ai) && loaded_name == name, "AI profile roundtrip differs");
            Display_profile_values display = to_display_profile_values(data.menu_elements);
            display.show_timer = true;
            display.timer_time_limit_min = 7;
            display.show_legal = false;
            require(save_display_profile_values(U"display-test.json", display, name), "Display profile save failed");
            Display_profile_values loaded_display;
            require(load_display_profile_values(U"display-test.json", &loaded_display, &loaded_name), "Display profile load failed");
            require(equals_display_profile_values(display, loaded_display) && loaded_name == name, "Display profile roundtrip differs");
        } else if (case_name == U"game-files") {
            auto history = sample_history();
            const String folder = U"./document/games/";
            const String date = U"2000_01_02_03_04_05";
            const String memo = U"memo, comma\n日本語と中文";
            game_save_helper::save_game_to_file(folder, date, U"Black 日本語", U"White 中文", memo, history, U"2000-01-02");
            Opening test_opening;
            require(load_game_from_json(data, test_opening, folder + date + U".json", date, ""), "Saved game load failed");
            require(data.graph_resources.nodes[0].size() == history.size(), "Saved game history size differs");
            require(data.graph_resources.nodes[0].back().board == history.back().board, "Saved game board differs");
            require(data.game_information.memo == memo && data.game_information.black_player_name == U"Black 日本語", "Game metadata differs");
            require(FileSystem::Exists(folder + U"summary.csv"), "Game CSV summary missing");
            result[U"board"] = Unicode::Widen(history.back().board.to_str(history.back().player));
        } else if (case_name == U"annotations") {
            auto history = sample_history();
            auto key = [](const History_elem& node) { return Board_image_position{node.board.player, node.board.opponent, node.player}; };
            Graph_resources graph;
            graph.nodes[0] = history;
            Board_image_edit edit;
            edit.marks[0] = BOARD_IMAGE_FLIP;
            edit.values[1] = 12;
            Board_image_arrow_input arrow;
            require(!arrow.click(0, &edit) && arrow.click(63, &edit), "Two-click arrow failed");
            require(!arrow.click(0, &edit) && !arrow.click(63, &edit), "Duplicate arrow accepted");
            graph.board_image_edits.save(key(history[0]), edit);
            graph.prune_board_image_edits();
            const auto restored = graph.board_image_edits.load(key(history[0]));
            require(restored.marks[0] == BOARD_IMAGE_FLIP && restored.values[1] == 12 && restored.arrows.size() == 1, "Annotation restore differs");
            require(graph.board_image_edits.load(key(history[1])).arrows.empty(), "Annotations leaked to another board");
            graph.nodes[0].erase(graph.nodes[0].begin());
            graph.prune_board_image_edits();
            require(graph.board_image_edits.positions.empty(), "Removed position kept its annotations");
            graph.board_image_edits.save(key(history[0]), edit);
            graph.init();
            require(graph.board_image_edits.positions.empty(), "New progression kept annotations");
            edit.clear();
            require(edit.arrows.empty() && !edit.values[1] && edit.marks[0] == BOARD_IMAGE_NOMARK, "Clear left annotations behind");
            require(board_image_cell_at(1, 1, 50) == board_image_cell_at(49, 49, 50) &&
                board_image_cell_at(50, 0, 50) == 1 && board_image_cell_at(400, 0, 50) == -1, "Grid snapping differs");
        } else if (case_name == U"forced-openings") {
            Forced_openings openings;
            openings.openings.emplace_back("f5d6c3d3", 1);
            openings.init();
            for (const auto& node : sample_history()) {
                if (node.board.n_discs() == 8) break;
                const int policy = openings.get_one(node.board);
                require(0 <= policy && policy < 64 && (node.board.get_legal() & (1ULL << policy)), "Forced opening returned illegal/undefined move");
            }
        } else if (case_name == U"search") {
            const auto search = ai(data.history_elem.board, 1, false, 0, false, false);
            require(0 <= search.policy && search.policy < 64 &&
                (data.history_elem.board.get_legal() & (1ULL << search.policy)), "GUI AI returned illegal move");
            result[U"move"] = Unicode::Widen(idx_to_coord(search.policy));
        } else {
            throw std::runtime_error("Unknown GUI operation-check case");
        }
    }

    void capture_scene(App& manager) {
        if (++frames < 8) return; // Render the new scene before requesting its frame.
        if (!capturing) {
            ScreenCapture::RequestCurrentFrame();
            capturing = true;
            return;
        }
        if (!ScreenCapture::HasNewFrame()) return;
        const Image& frame = ScreenCapture::GetFrame();
        require(!frame.isEmpty() && frame.width() >= 600 && frame.height() >= 400, "Missing/undersized rendered frame");
        std::set<uint32_t> colors;
        for (int y = 0; y < frame.height(); y += 3) {
            for (int x = 0; x < frame.width(); x += 3) {
                const auto pixel = frame[y][x];
                colors.insert((pixel.r << 16) | (pixel.g << 8) | pixel.b);
            }
        }
        require(colors.size() >= 8, "Rendered frame is blank or nearly uniform");
        const String name = case_name == U"scenes" ? scenes[scene_index] : U"Main_scene";
        require(frame.save(U"frame-" + name + U".png"), "Could not save rendered frame");
        if (case_name == U"scenes" && ++scene_index < scenes.size()) {
            require(manager.changeScene(scenes[scene_index], 0), "Scene transition failed");
            frames = 0;
            capturing = false;
        } else {
            finish(manager, true);
        }
    }

    void observe_desktop(const Common_resources& data) {
        ++desktop_frames;
#if SIV3D_PLATFORM(WINDOWS)
        write_mouse_trace(desktop_frames);
#endif
        if (desktop_frames % 6 != 0 && observed_scene != U"Close") return;
        JSON state;
        state[U"marker"] = U"egaroucid-desktop-check-v1";
        state[U"token"] = token;
        state[U"frame"] = desktop_frames;
        state[U"scene"] = observed_scene;
        state[U"operation_status"] = observed_status;
        state[U"list_size"] = observed_list_size;
        state[U"ready_details"] = observed_ready_details;
        state[U"loading"] = data.window_state.loading;
        state[U"focused"] = Window::GetState().focused;
        if (data.window_state.loading) {
            state.save(U"desktop-state.json");
            return; // The asynchronous loader still owns settings/resources.
        }
        state[U"ime_enabled_for_scene"] = gui_scene_ime::desired_enabled;
        state[U"ime_composition"] = TextInput::GetEditingText();
        state[U"board"] = Unicode::Widen(data.history_elem.board.to_str(data.history_elem.player));
        state[U"discs"] = data.history_elem.board.n_discs();
        state[U"history_size"] = data.graph_resources.nodes[0].size();
        state[U"branch"] = data.graph_resources.branch;
        state[U"black_name"] = data.game_information.black_player_name;
        state[U"white_name"] = data.game_information.white_player_name;
        state[U"memo"] = data.game_information.memo;
        state[U"annotations"] = data.graph_resources.board_image_edits.positions.size();
        state[U"book_changed"] = data.book_information.changed;
        state[U"board_image_mark"] = data.settings.board_image_mark;
        state[U"board_image_color"] = data.settings.board_image_color;
        state[U"board_image_coordinate"] = data.settings.board_image_coordinate;
        state[U"show_legal"] = data.menu_elements.show_legal;
        state[U"level"] = data.menu_elements.level;
        state[U"ai_black"] = data.menu_elements.ai_put_black;
        state[U"ai_white"] = data.menu_elements.ai_put_white;
        state[U"version"] = EGAROUCID_VERSION;
        state.save(U"desktop-state.json");
    }

public:
    bool prepare() {
        const auto& args = System::GetCommandLineArgs();
        for (size_t i = 0; i < args.size(); ++i) {
            if (args[i] != U"--operation-check") continue;
            enabled = true;
            if (i + 1 == args.size()) return false;
            const String folder = FileSystem::FullPath(args[i + 1]);
            const JSON request = JSON::Load(folder + U"/request.json");
            if (request[U"marker"].getOr<String>(U"") != U"egaroucid-operation-check-v1") return false;
            if (!FileSystem::ChangeCurrentDirectory(folder)) return false;
            case_name = request[U"case"].getOr<String>(U"");
            token = request[U"token"].getOr<String>(U"");
            // Reject installed builds before they can access Documents/LocalAppData.
            if (!GUI_PORTABLE_MODE) {
                write_result(false, U"Use a GUI Portable build for isolated operation checks");
                return false;
            }
            desktop_observation_enabled = (case_name == U"desktop");
#if SIV3D_PLATFORM(WINDOWS)
            if (desktop_observation_enabled) {
                std::lock_guard<std::mutex> lock(windows_shortcut_message_state_mutex());
                windows_shortcut_message_state().diagnostic_observer = &observe_mouse_message;
            }
#endif
            JSON settings = JSON::Load(U"appdata/setting.json");
            settings[AUTO_UPDATE_CHECK_SETTING_KEY] = false;
            require(settings.save(U"appdata/setting.json"), "Could not prepare isolated settings");
            log(U"Waiting for normal GUI loading");
            return true;
        }
        return true; // No flag: no test files, scene changes, or settings changes.
    }

    void update(App& manager) {
        if (!enabled || closing) return;
        auto& data = *manager.get();
        if (desktop_observation_enabled) {
            observe_desktop(data);
            return;
        }
        if (!started && (data.window_state.loading || data.graph_resources.nodes[0].empty())) return;
        try {
            if (!started) {
                started = true;
                log(U"Running case: " + case_name);
                data.graph_resources.need_init = false;
                check_case(data);
                if (case_name != U"scenes" && case_name != U"startup") {
                    finish(manager, true);
                    return;
                }
            }
            capture_scene(manager);
        } catch (const std::exception& error) {
            finish(manager, false, Unicode::FromUTF8(error.what()));
        }
    }

    void log(const String& line) const {
        if (!enabled || line.isEmpty()) return;
        std::ofstream stream("native.log", std::ios::app);
        stream << line.toUTF8() << '\n';
    }
};
}
