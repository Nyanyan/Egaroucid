# GitHub PR Steps

This change is easiest to submit as stacked PRs:

1. `pr/xot-identification`
2. `pr/game-library`
3. `pr/oq-input`

Each PR should target the previous one:

- `pr/xot-identification` targets the upstream base branch.
- `pr/game-library` targets `pr/xot-identification`.
- `pr/oq-input` targets `pr/game-library`.

## Before Creating Branches

Run:

```powershell
git status --short
git diff --check
& 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\MSBuild\Current\Bin\amd64\MSBuild.exe' Egaroucid.sln /m /p:Configuration=Release /p:Platform=x64 /verbosity:minimal
```

## Suggested File Groups

XOT identification:

```text
src/engine/xot.hpp
src/engine/xot_keys.hpp
src/engine/engine_all.hpp
src/gui/function/util.hpp
src/gui/function/graph.hpp
src/gui/main_scene.hpp
src/gui/silent_load_scene.hpp
src/gui/close_scene.hpp
src/gui/function/const/gui_common.hpp
src/gui/function/menu_definition.hpp
bin/resources/languages/*.json
src/tools/release_script/format_files/0_common_files/languages/*.json
docs/pr/xot-identification.md
docs/pr/images/xot-identification/*
```

Game Library:

```text
src/gui/input_scene.hpp
src/gui/save_location_picker_scene.hpp
src/gui/close_scene.hpp
src/gui/function/const/gui_common.hpp
src/gui/function/menu_definition.hpp
src/gui/silent_load_scene.hpp
bin/resources/languages/*.json
src/tools/release_script/format_files/0_common_files/languages/*.json
src/tools/release_script/format_files/GUI_Installer_files/Documents/Egaroucid/games/summary.csv
src/tools/release_script/format_files/GUI_Portable_files/document/games/summary.csv
docs/pr/game-library.md
docs/pr/images/game-library/*
```

OQ input:

```text
src/gui/input_scene.hpp
src/gui/main_scene.hpp
src/gui/function/const/gui_common.hpp
src/gui/function/menu_definition.hpp
src/gui/silent_load_scene.hpp
src/gui/close_scene.hpp
bin/resources/languages/*.json
src/tools/release_script/format_files/0_common_files/languages/*.json
docs/pr/oq-input.md
docs/pr/images/oq-input/*
```

Some files are shared between PRs. For those files, use `git add -p` and stage only the hunks for the current PR.

## Local Git Commands

Create the first branch:

```powershell
git switch -c pr/xot-identification
git add -p
git commit -m "Add XOT identification"
git push -u origin pr/xot-identification
```

Create the second branch from the first:

```powershell
git switch -c pr/game-library
git add -p
git commit -m "Add Game Library"
git push -u origin pr/game-library
```

Create the third branch from the second:

```powershell
git switch -c pr/oq-input
git add -p
git commit -m "Add Othello Quest import"
git push -u origin pr/oq-input
```

## On GitHub

For each pushed branch:

1. Open the repository page on GitHub.
2. Click "Compare & pull request".
3. Check the base branch:
   - XOT PR: base is the upstream branch, usually `main` or `master`.
   - Game Library PR: base is `pr/xot-identification`.
   - OQ PR: base is `pr/game-library`.
4. Copy the matching draft from:
   - `docs/pr/xot-identification.md`
   - `docs/pr/game-library.md`
   - `docs/pr/oq-input.md`
5. Paste it into the PR description.
6. Submit the PR.

## Important

Do not use plain `git add .` for these PRs. The current work touches shared files, especially `src/gui/input_scene.hpp` and language JSON files, so staging everything will mix the three PRs together.
