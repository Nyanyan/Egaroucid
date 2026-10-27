#include <future>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

struct Test_data {
    struct { std::string book_file = "old.egbk3"; } settings;
    struct { bool changed = true; } book_information;
};

struct Test_book {
    std::vector<int> entries{42};
    std::vector<int> saved_entries;
    std::string saved_file;
    bool save_succeeds = false;
    int save_calls = 0;
    std::shared_future<void> save_gate;

    bool save_egbk3(const std::string& file, const std::string&) {
        ++save_calls;
        if (save_gate.valid()) save_gate.wait();
        if (!save_succeeds) return false;
        saved_entries = entries;
        saved_file = file;
        return true;
    }
} book;

void delete_book() { book.entries.clear(); }

// The runner includes the actual Refer_book transition methods from book_scene.hpp.
// GUI drawing and file writing are replaced so failures and pending saves are deterministic.
struct Test_scene {
    Test_data data;
    std::string book_file = "new.egbk3";
    std::future<bool> save_book_future;
    std::future<void> delete_book_future;
    bool book_saving = false, book_deleting = false, save_failed = false;
    Test_data& getData() { return data; }
#include "book_reference_methods.hpp"
};

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

int main() {
    try {
        Test_scene scene;
        std::promise<void> release_save;
        book.save_gate = release_save.get_future().share();
        scene.start_reference_change();
        const bool old_reference_kept = scene.data.settings.book_file == "old.egbk3";
        const bool dirty_kept = scene.data.book_information.changed;
        release_save.set_value();
        if (scene.save_book_future.valid()) {
            scene.save_book_future.wait();
            scene.finish_book_save();
        }
        if (scene.delete_book_future.valid()) scene.delete_book_future.get();
        require(old_reference_kept && dirty_kept, "reference or dirty flag changed before saving succeeded");
        require(scene.save_failed && !scene.book_saving && !scene.book_deleting, "save failure did not stop replacement");
        require(scene.data.settings.book_file == "old.egbk3" && scene.data.book_information.changed, "save failure lost the old settings");
        require(book.entries == std::vector<int>{42}, "save failure deleted unsaved entries");

        book.save_succeeds = true;
        scene.start_reference_change();
        scene.save_book_future.wait();
        scene.finish_book_save();
        scene.delete_book_future.get();
        require(book.saved_file == "old.egbk3" && book.saved_entries == std::vector<int>{42}, "retry did not save the old book");
        require(scene.data.settings.book_file == "new.egbk3" && !scene.data.book_information.changed, "successful save did not switch the reference");
        require(!scene.save_failed && book.entries.empty(), "successful retry left a failure or did not replace the book");

        Test_scene clean;
        clean.data.book_information.changed = false;
        book.entries = {99};
        const int previous_calls = book.save_calls;
        clean.start_reference_change();
        clean.delete_book_future.get();
        require(book.save_calls == previous_calls && !clean.book_saving, "unchanged book was saved unnecessarily");
        require(clean.data.settings.book_file == "new.egbk3" && book.entries.empty(), "unchanged book could not be replaced");
        std::cout << "PASS: pending save, failed save retention, retry, successful switch, unchanged book\n";
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
