# Codex Instructions for Egaroucid

This note is for Codex-specific operation in this repository.

## Source of Truth
- Follow [`copilot-instructions.md`](./copilot-instructions.md) as the primary coding guideline.
- If there is a conflict, prioritize project safety and existing repository conventions.

## Required Practices
- Use `language.get()` for all new user-facing UI text.
- When adding a new UI text key, update all 4 language files:
  - `bin/resources/languages/english.json`
  - `bin/resources/languages/japanese.json`
  - `bin/resources/languages/simplified_chinese.json`
  - `bin/resources/languages/traditional_chinese_taiwan.json`
- Keep release-script language templates in sync when relevant:
  - `src/tools/release_script/format_files/0_common_files/languages/*.json`

## Build Notes
- GUI build should follow release-script based workflow described in `copilot-instructions.md`.
- Console quick compile may use the command examples in `copilot-instructions.md`.

## Editing Safety
- Do not hardcode localized UI labels in scene code.
- Prefer minimal diffs and avoid unrelated refactors.
- Do not revert user changes outside your task scope.
