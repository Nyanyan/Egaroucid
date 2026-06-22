# Umigame Condition Settings

## Terminology and Documentation Constraints

Per the maintainer's request, this proposal should use common game AI and
computer game research terminology where possible. Any custom terms must be
defined before they are used and related to standard terminology.

Terms used in this PR:

- `score` / `evaluation`: a book value in disc units. Positive values are good
  for the side indicated by the stated perspective.
- `mover-perspective score`: a child move score from the player-to-move's
  perspective before that move is played.
- `black-score`: the same score converted to Black's perspective. Positive
  values are favorable for Black; negative values are favorable for White.
- `move loss`: the score loss, or regret, of a child move relative to the best
  child move at the same node.
- `score window`: the accepted black-score interval for child moves.
- `Umigame number` / `minimum memorization number`: the existing Egaroucid
  display value computed for Black and White.

## Summary

This PR adds configurable display-side score filters for Umigame numbers.

The feature request came from mainland Chinese user MangWu, who wanted the next
Egaroucid version to allow wider and more practical conditions when calculating
and displaying `海龟数` / Umigame numbers.

MangWu's original request:

> 希望下一个版本的EG能对“海龟数”的限制进行调整：例如估值为0的某个棋步，海龟数随机值放宽至±2，在计算海龟数时，将所有己方为0或-2的后续分支全部计算在内；将对方0或+2的分支计算在内。

In this PR, that request is implemented as two display-side controls:

- `Errors per Move`: maximum per-node move loss.
- `Integration Errors`: side-specific black-score window, shown as
  `B{black} W{white}`.

The internal black-score window for `Integration Errors` is `[-B, +W]`. For
example, `B3 W8` accepts child moves with black-scores from `-3` through `+8`.

## Scope

This PR only changes Umigame number display conditions.

It does not change book learning behavior, book revision handling, version
numbers, or the underlying lower-limit style design discussed elsewhere. The
existing Umigame result model remains non-negative `B{black} W{white}` counts.

## User-Facing Changes

- Adds `Errors per Move` under `Display > On Cell > Min Memorize Number`.
- Adds `Integration Errors` under the same menu, with independent black and
  white knobs.
- Adds `Apply Setting`; changing the sliders alone does not recalculate Umigame
  numbers until the setting is applied.
- Displays the `Integration Errors` slider value as `B{black} W{white}`.
- Draws the white player-loss knob as a pure white ball.
- Draws a single orange knob when the black and white knobs overlap, making it
  clear that both knobs are at the same value.

## Implementation Notes

- `Book::get_all_moves_within_child_loss()` gathers child moves whose
  mover-perspective score is within `max_move_loss` of the node's best child
  score.
- `Umigame_condition` carries `max_move_loss`, `black_max_loss`, and
  `white_max_loss`. These are display-side search filters, not book-learning
  parameters.
- The Umigame search converts each mover-perspective child score to a
  black-score before applying the `[-black_max_loss, +white_max_loss]` score
  window.
- Umigame cache entries are protected by condition context. A change to
  `depth`, `max_move_loss`, `black_max_loss`, or `white_max_loss` clears the
  cache and increments the generation.
- Cache reads and writes are tied to the same generation so an older async
  request cannot reuse cache entries from a newer condition context.
- Async Umigame UI jobs use request IDs. Old results and undefined interrupted
  results are ignored, so stale jobs cannot mark the current UI as complete.
- If the current request returns an undefined result, the UI advances the
  request ID and clears displayed values before retrying.
- No `lower_limit`, `No range intersection`, `book_revision`, or
  `umigame_condition.hpp` design is introduced in this PR.

## Screenshots

### Condition Menu

![Umigame condition menu](images/umigame-condition-settings/condition-menu.png)

### Displayed Umigame Values

![Displayed Umigame values](images/umigame-condition-settings/umigame-values.png)

## Relation To Issue #612

This PR is conceptually related to issue #612 because both discuss how Umigame
conditions should be interpreted. This PR is intentionally narrower: it only
changes display conditions for Umigame numbers and does not modify book
learning.

## Build Compatibility Note

This branch keeps the existing OpenSiv3D IME compatibility fix needed by the
local Windows GUI build: `TextInput::EnableIME()` is not available in the
tested OpenSiv3D environment, so the Windows helper only calls `DisableIME()`
when disabling IME input. This is independent from the Umigame feature and can
be split out by maintainers if preferred.

## Validation

- `git diff --check`
- Parsed language JSON files under:
  - `bin/resources/languages/`
  - `src/tools/release_script/format_files/0_common_files/languages/`
- `python src/tools/release_script/build_console.py -c Generic`
- `python src/tools/release_script/build_gui.py -c Generic`

The Generic console and GUI builds passed locally with:

`MSBUILD_PATH=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\MSBuild\Current\Bin\MSBuild.exe`

Manual UI checks covered:

- `Integration Errors = B10 W10` shows one orange overlap knob.
- `B10 W9` / `B9 W10` separates the black and pure-white knobs.
- Slider changes do not recalculate until `Apply Setting` is clicked.
- Applied condition changes clear displayed Umigame values and recalculate.

## Version

No version number is changed in this PR.
