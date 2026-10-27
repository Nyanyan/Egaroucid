#include <Siv3D.hpp>
#include <Windows.h>
#include <iostream>
#include <stdexcept>
using namespace s3d;
#include "folder_empty_function.hpp"

void Main() {}

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

struct File_lock {
    HANDLE handle;
    explicit File_lock(const String& path)
        : handle(CreateFileW(path.toWstr().c_str(), GENERIC_WRITE,
                            FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING, 0, nullptr)) {
        require(handle != INVALID_HANDLE_VALUE, "could not lock summary.csv against readers");
    }
    ~File_lock() { CloseHandle(handle); }
};

int main(int argc, char** argv) {
    try {
        require(argc == 2, "fixture directory argument required");
        const String root = Unicode::FromUTF8(argv[1]) + U"/";
        require(FileSystem::CreateDirectories(root + U"empty/"), "could not create fixture");
        require(is_list_folder_empty(root + U"empty/"), "empty folder cannot be deleted");
        require(!is_list_folder_empty(root + U"missing/"), "missing folder counted as empty");

        FileSystem::CreateDirectories(root + U"nested/child/");
        require(!is_list_folder_empty(root + U"nested/"), "folder with a subfolder counted as empty");
        FileSystem::CreateDirectories(root + U"zero/");
        { BinaryWriter file(root + U"zero/summary.csv"); }
        require(is_list_folder_empty(root + U"zero/"), "zero-byte summary counted as nonempty");
        FileSystem::CreateDirectories(root + U"bom/");
        { TextWriter file(root + U"bom/summary.csv", TextEncoding::UTF8_WITH_BOM); }
        require(is_list_folder_empty(root + U"bom/"), "BOM-only summary counted as nonempty");

        FileSystem::CreateDirectories(root + U"한글😀/");
        CSV entries;
        entries.writeRow(U"game", U"player", U"memo");
        require(entries.save(root + U"한글😀/summary.csv"), "could not save CSV fixture");
        require(!is_list_folder_empty(root + U"한글😀/"), "Unicode folder with entries counted as empty");

        FileSystem::CreateDirectories(root + U"locked/");
        require(entries.save(root + U"locked/summary.csv"), "could not save locked CSV fixture");
        {
            // Reads are denied while deletion remains permitted: the old empty check exposed a delete button.
            File_lock lock(root + U"locked/summary.csv");
            require(!is_list_folder_empty(root + U"locked/"), "unreadable summary counted as empty");
        }
        require(!is_list_folder_empty(root + U"locked/"), "unlocked populated summary counted as empty");
        std::cout << "PASS: empty, zero-byte/BOM-only CSV, Unicode, subfolders, missing path, read-denied CSV\n";
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
