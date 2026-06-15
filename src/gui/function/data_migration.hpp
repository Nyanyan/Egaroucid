/*
    Egaroucid Project

    @file data_migration.hpp
        Export and import user data for migration
    @date 2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

#pragma once
#include <Siv3D.hpp>
#include <algorithm>
#include "const/gui_common.hpp"

enum class Data_migration_error {
    none,
    invalid_destination,
    invalid_source,
    unsupported_zip,
    unsafe_source,
    copy_failed
};

struct Data_migration_result {
    bool succeeded{ false };
    Data_migration_error error{ Data_migration_error::none };
    String path;
};

inline String data_migration_slash_path(String path) {
    return path.replaced(U"\\", U"/");
}

inline String data_migration_with_trailing_separator(String path) {
    path = data_migration_slash_path(path);
    if (path.size() && path.back() != U'/') {
        path += U"/";
    }
    return path;
}

inline String data_migration_join_path(String dir, const String& name) {
    return data_migration_with_trailing_separator(dir) + name;
}

inline String data_migration_path_leaf(String path) {
    path = data_migration_slash_path(path);
    while (path.size() && path.back() == U'/') {
        path.pop_back();
    }
    const size_t pos = path.lastIndexOf(U'/');
    if (pos == String::npos) {
        return path;
    }
    return path.substr(pos + 1);
}

inline String data_migration_normalized_full_dir(String path) {
    path = data_migration_with_trailing_separator(FileSystem::FullPath(path));
    return path.lowercased();
}

inline bool data_migration_is_same_or_inside_dir(const String& path, const String& dir) {
    if (path.isEmpty() || dir.isEmpty()) {
        return false;
    }
    const String normalized_path = data_migration_normalized_full_dir(path);
    const String normalized_dir = data_migration_normalized_full_dir(dir);
    return normalized_path == normalized_dir || normalized_path.starts_with(normalized_dir);
}

inline String data_migration_unique_path(String path) {
    if (!FileSystem::Exists(path)) {
        return path;
    }

    const String parent = FileSystem::ParentPath(path);
    const String base = FileSystem::BaseName(path);
    const String extension = FileSystem::Extension(path);
    for (int i = 1; i < 10000; ++i) {
        String candidate = data_migration_join_path(parent, base + U"_imported_" + Format(i));
        if (!extension.isEmpty()) {
            candidate += U"." + extension;
        }
        if (!FileSystem::Exists(candidate)) {
            return candidate;
        }
    }
    return path;
}

inline bool data_migration_ensure_parent_dir(const String& path) {
    const String parent = FileSystem::ParentPath(path);
    if (parent.isEmpty() || FileSystem::Exists(parent)) {
        return true;
    }
    return FileSystem::CreateDirectories(parent) || FileSystem::Exists(parent);
}

inline bool data_migration_copy_file_overwrite(const String& source, const String& target) {
    if (!data_migration_ensure_parent_dir(target)) {
        return false;
    }
    if (FileSystem::Exists(target) && !FileSystem::Remove(target, AllowUndo::No)) {
        return false;
    }
    return FileSystem::Copy(source, target);
}

inline bool data_migration_copy_file_preserve_conflict(const String& source, const String& target) {
    String actual_target = target;
    if (FileSystem::Exists(actual_target)) {
        actual_target = data_migration_unique_path(actual_target);
    }
    if (!data_migration_ensure_parent_dir(actual_target)) {
        return false;
    }
    return FileSystem::Copy(source, actual_target);
}

inline bool data_migration_copy_tree_overwrite(
    const String& source,
    const String& target,
    const String& skip_source_dir = U""
) {
    if (!FileSystem::Exists(source)) {
        return FileSystem::CreateDirectories(target) || FileSystem::Exists(target);
    }

    if (!skip_source_dir.isEmpty() && data_migration_is_same_or_inside_dir(source, skip_source_dir)) {
        return true;
    }

    if (FileSystem::IsDirectory(source)) {
        if (FileSystem::Exists(target) && !FileSystem::IsDirectory(target)) {
            if (!FileSystem::Remove(target, AllowUndo::No)) {
                return false;
            }
        }
        if (!FileSystem::CreateDirectories(target) && !FileSystem::Exists(target)) {
            return false;
        }
        for (const auto& child : FileSystem::DirectoryContents(source)) {
            const String child_name = data_migration_path_leaf(child);
            if (child_name.isEmpty()) {
                continue;
            }
            if (!data_migration_copy_tree_overwrite(
                    child,
                    data_migration_join_path(target, child_name),
                    skip_source_dir
                )) {
                return false;
            }
        }
        return true;
    }

    return data_migration_copy_file_overwrite(source, target);
}

inline bool data_migration_copy_tree_preserve_conflicts(const String& source, const String& target) {
    if (!FileSystem::Exists(source)) {
        return true;
    }

    if (FileSystem::IsDirectory(source)) {
        String actual_target = target;
        if (FileSystem::Exists(actual_target) && !FileSystem::IsDirectory(actual_target)) {
            actual_target = data_migration_unique_path(actual_target);
        }
        if (!FileSystem::CreateDirectories(actual_target) && !FileSystem::Exists(actual_target)) {
            return false;
        }
        for (const auto& child : FileSystem::DirectoryContents(source)) {
            const String child_name = data_migration_path_leaf(child);
            if (child_name.isEmpty()) {
                continue;
            }
            if (!data_migration_copy_tree_preserve_conflicts(
                    child,
                    data_migration_join_path(actual_target, child_name)
                )) {
                return false;
            }
        }
        return true;
    }

    return data_migration_copy_file_preserve_conflict(source, target);
}

inline bool data_migration_save_manifest(const String& root, const Directories& directories) {
    JSON manifest;
    manifest[U"format"] = U"Egaroucid_Settings_Folder";
    manifest[U"version"] = Unicode::Widen(EGAROUCID_VERSION);
    manifest[U"exported_at"] = DateTime::Now().format(U"yyyy-MM-dd HH:mm:ss");
    manifest[U"source_document_dir"] = Unicode::Widen(directories.document_dir);
    manifest[U"source_appdata_dir"] = Unicode::Widen(directories.appdata_dir);
    const String manifest_path = data_migration_join_path(root, U"manifest.json");
    return manifest.save(manifest_path);
}

inline String data_migration_make_unique_export_root(const String& destination_dir) {
    const String timestamp = DateTime::Now().format(U"yyyyMMdd_HHmmss");
    const String base = data_migration_join_path(destination_dir, U"Egaroucid_Settings_" + timestamp);
    if (!FileSystem::Exists(base)) {
        return base;
    }
    for (int i = 1; i < 10000; ++i) {
        const String candidate = base + U"_" + Format(i);
        if (!FileSystem::Exists(candidate)) {
            return candidate;
        }
    }
    return base;
}

inline Data_migration_result export_egaroucid_settings_data(
    const Directories& directories,
    const String& destination_dir
) {
    Data_migration_result result;
    if (!FileSystem::IsDirectory(destination_dir)) {
        result.error = Data_migration_error::invalid_destination;
        return result;
    }

    const String root = data_migration_make_unique_export_root(destination_dir);
    if (!FileSystem::CreateDirectories(root) && !FileSystem::Exists(root)) {
        result.error = Data_migration_error::copy_failed;
        return result;
    }
    if (!data_migration_save_manifest(root, directories)) {
        result.error = Data_migration_error::copy_failed;
        result.path = root;
        return result;
    }

    const String appdata_source = Unicode::Widen(directories.appdata_dir);
    const String document_source = Unicode::Widen(directories.document_dir);
    const bool appdata_ok = data_migration_copy_tree_overwrite(
        appdata_source,
        data_migration_join_path(root, U"appdata"),
        root
    );
    const bool document_ok = data_migration_copy_tree_overwrite(
        document_source,
        data_migration_join_path(root, U"document"),
        root
    );

    result.succeeded = appdata_ok && document_ok;
    result.error = result.succeeded ? Data_migration_error::none : Data_migration_error::copy_failed;
    result.path = root;
    return result;
}

inline bool data_migration_rewrite_path_value(JSON& json, const String& key, String old_root, String new_root) {
    if (json[key].getType() != JSONValueType::String || old_root.isEmpty() || new_root.isEmpty()) {
        return false;
    }

    String value = data_migration_slash_path(json[key].getString());
    old_root = data_migration_with_trailing_separator(old_root);
    new_root = data_migration_with_trailing_separator(new_root);

    if (!value.lowercased().starts_with(old_root.lowercased())) {
        return false;
    }

    json[key] = new_root + value.substr(old_root.size());
    return true;
}

inline void data_migration_rewrite_imported_setting_paths(
    const String& backup_root,
    const Directories& directories
) {
    const JSON manifest = JSON::Load(data_migration_join_path(backup_root, U"manifest.json"));
    if (!manifest || manifest[U"source_document_dir"].getType() != JSONValueType::String) {
        return;
    }

    const String source_document_dir = manifest[U"source_document_dir"].getString();
    const String settings_path = Unicode::Widen(directories.appdata_dir) + U"setting.json";
    JSON setting_json = JSON::Load(settings_path);
    if (!setting_json) {
        return;
    }

    bool changed = false;
    changed |= data_migration_rewrite_path_value(setting_json, U"book_file", source_document_dir, Unicode::Widen(directories.document_dir));
    changed |= data_migration_rewrite_path_value(setting_json, U"screenshot_saving_dir", source_document_dir, Unicode::Widen(directories.document_dir));
    if (changed) {
        setting_json.save(settings_path);
    }
}

inline bool data_migration_copy_document_for_import(const String& source_document, const String& target_document) {
    if (!FileSystem::Exists(source_document)) {
        return true;
    }
    if (!FileSystem::CreateDirectories(target_document) && !FileSystem::Exists(target_document)) {
        return false;
    }

    for (const auto& child : FileSystem::DirectoryContents(source_document)) {
        const String child_name = data_migration_path_leaf(child);
        if (child_name.isEmpty()) {
            continue;
        }
        const String target = data_migration_join_path(target_document, child_name);
        if (child_name.lowercased() == U"games" && FileSystem::IsDirectory(child)) {
            if (!data_migration_copy_tree_preserve_conflicts(child, target)) {
                return false;
            }
        } else {
            if (!data_migration_copy_tree_overwrite(child, target)) {
                return false;
            }
        }
    }
    return true;
}

inline Data_migration_result import_egaroucid_settings_data(
    const Directories& directories,
    const String& backup_root
) {
    Data_migration_result result;
    if (backup_root.lowercased().ends_with(U".zip")) {
        result.error = Data_migration_error::unsupported_zip;
        return result;
    }
    if (!FileSystem::IsDirectory(backup_root)) {
        result.error = Data_migration_error::invalid_source;
        return result;
    }

    const String source_appdata = data_migration_join_path(backup_root, U"appdata");
    const String source_document = data_migration_join_path(backup_root, U"document");
    if (!FileSystem::IsDirectory(source_appdata) || !FileSystem::IsDirectory(source_document)) {
        result.error = Data_migration_error::invalid_source;
        return result;
    }

    const String target_appdata = Unicode::Widen(directories.appdata_dir);
    const String target_document = Unicode::Widen(directories.document_dir);
    if (data_migration_is_same_or_inside_dir(source_appdata, target_appdata)) {
        result.error = Data_migration_error::unsafe_source;
        return result;
    }

    if (FileSystem::Exists(target_appdata) && !FileSystem::Remove(target_appdata, AllowUndo::No)) {
        result.error = Data_migration_error::copy_failed;
        return result;
    }
    if (!data_migration_copy_tree_overwrite(source_appdata, target_appdata)) {
        result.error = Data_migration_error::copy_failed;
        return result;
    }
    data_migration_rewrite_imported_setting_paths(backup_root, directories);

    if (!data_migration_copy_document_for_import(source_document, target_document)) {
        result.error = Data_migration_error::copy_failed;
        return result;
    }

    result.succeeded = true;
    result.error = Data_migration_error::none;
    result.path = backup_root;
    return result;
}
