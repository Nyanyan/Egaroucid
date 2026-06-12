# Game Library PR Draft

## Summary

Add a Game Library scene for browsing, organizing, importing, and managing saved games.

## What Changed

- Added a Game Library entry under In/Out.
- Added folder navigation for `games/`, including system folders for Othello Quest imports and recycle bin.
- Added copy, cut, paste, drag-and-drop move, reorder, rename, and delete flows.
- Added recycle-bin support with permanent delete inside the recycle bin.
- Added save-location picker support for choosing where a game should be saved.
- Hardened CSV handling:
  - Skips malformed `summary.csv` rows with too few columns.
  - Preserves all CSV columns when removing rows.
  - Creates missing root `summary.csv` release templates.
- Prevents overwriting existing JSON files when moving games into a folder with a matching filename.

## Why

The previous save/load flow had no central library UI for organizing saved games. The new scene gives users a direct way to browse and manage saved games while preserving existing game metadata.

The review also found real data-safety bugs in the library flow: moving a game could overwrite an existing JSON file, and CSV records could be changed even when the file move failed. Those cases are now guarded.

## Screenshots

![Game Library root folders](images/game-library/root-folders.png)

![Game Library recycle bin](images/game-library/recycle-bin.png)

## Validation

- Release x64 GUI build passed with MSBuild.
- Runtime and release language JSON files are synchronized.
- Release templates contain 12 empty `summary.csv` files.
- Manually verified Game Library root navigation and recycle-bin UI with the screenshots above.

## Suggested PR Title

Add Game Library management UI
