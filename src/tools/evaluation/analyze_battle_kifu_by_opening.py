import argparse
import csv
import os
import sys
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class BattleSummary:
    battle: int
    opening_idx: int
    disc_sum: int

    @property
    def avg_disc(self):
        return self.disc_sum / 2.0


@dataclass
class OpeningSummary:
    opening_idx: int
    n_battles: int
    disc_sum: int
    wins: int
    draws: int
    losses: int

    @property
    def avg_disc(self):
        return self.disc_sum / (2.0 * self.n_battles)


@dataclass
class RunSummary:
    name: str
    path: str
    n_rows: int
    battles: list
    openings: list


def parse_run_arg(value):
    name, sep, path = value.partition("=")
    if sep:
        if not name or not path:
            raise argparse.ArgumentTypeError("run must be NAME=PATH or PATH")
        return name, path
    path = value
    base = os.path.basename(path)
    return os.path.splitext(base)[0], path


def read_rows(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        required = {"battle", "opening_idx", "p0_disc_diff"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError("{} missing columns: {}".format(path, ",".join(sorted(missing))))
        return list(reader)


def summarize_run(name, path):
    rows = read_rows(path)
    by_battle = defaultdict(list)
    for row in rows:
        by_battle[int(row["battle"])].append(row)

    battles = []
    for battle, battle_rows in sorted(by_battle.items()):
        opening_indices = {int(row["opening_idx"]) for row in battle_rows}
        if len(opening_indices) != 1:
            raise ValueError("{} battle {} has multiple opening_idx values".format(path, battle))
        if len(battle_rows) != 2:
            raise ValueError("{} battle {} has {} rows, expected 2".format(path, battle, len(battle_rows)))
        disc_sum = sum(int(row["p0_disc_diff"]) for row in battle_rows)
        battles.append(BattleSummary(battle, opening_indices.pop(), disc_sum))

    by_opening = defaultdict(list)
    for battle in battles:
        by_opening[battle.opening_idx].append(battle)

    openings = []
    for opening_idx, opening_battles in sorted(by_opening.items()):
        disc_sum = sum(battle.disc_sum for battle in opening_battles)
        wins = sum(1 for battle in opening_battles if battle.disc_sum > 0)
        draws = sum(1 for battle in opening_battles if battle.disc_sum == 0)
        losses = sum(1 for battle in opening_battles if battle.disc_sum < 0)
        openings.append(OpeningSummary(opening_idx, len(opening_battles), disc_sum, wins, draws, losses))

    return RunSummary(name, path, len(rows), battles, openings)


def wdl_and_rate_from_summaries(items):
    wins = sum(1 for item in items if item.disc_sum > 0)
    draws = sum(1 for item in items if item.disc_sum == 0)
    losses = sum(1 for item in items if item.disc_sum < 0)
    n = wins + draws + losses
    winrate = (wins + 0.5 * draws) / n if n else 0.0
    avg_disc = sum(item.avg_disc for item in items) / n if n else 0.0
    return wins, draws, losses, winrate, avg_disc


def print_run_summary(run, top):
    battle_w, battle_d, battle_l, battle_wr, battle_avg = wdl_and_rate_from_summaries(run.battles)
    open_w, open_d, open_l, open_wr, open_avg = wdl_and_rate_from_summaries(run.openings)
    print("run\t{}\t{}".format(run.name, run.path))
    print(
        "summary\trows={}\tpaired_battles={}\topenings={}\tbattle_wdl={}/{}/{}\tbattle_wr={:.4f}\tbattle_avg={:+.2f}\topening_wdl={}/{}/{}\topening_wr={:.4f}\topening_avg={:+.2f}".format(
            run.n_rows,
            len(run.battles),
            len(run.openings),
            battle_w,
            battle_d,
            battle_l,
            battle_wr,
            battle_avg,
            open_w,
            open_d,
            open_l,
            open_wr,
            open_avg,
        )
    )

    nonzero = [opening for opening in run.openings if opening.disc_sum != 0]
    nonzero.sort(key=lambda item: (-abs(item.avg_disc), item.opening_idx))
    if top > 0:
        nonzero = nonzero[:top]
    print("nonzero_openings\t{}".format(len([opening for opening in run.openings if opening.disc_sum != 0])))
    for opening in nonzero:
        print(
            "opening\t{}\tn={}\twdl={}/{}/{}\tdisc_sum={}\tavg={:+.2f}".format(
                opening.opening_idx,
                opening.n_battles,
                opening.wins,
                opening.draws,
                opening.losses,
                opening.disc_sum,
                opening.avg_disc,
            )
        )


def print_comparison(runs, top):
    if len(runs) < 2:
        return
    by_name = {
        run.name: {opening.opening_idx: opening for opening in run.openings}
        for run in runs
    }
    opening_ids = sorted(set().union(*(set(openings.keys()) for openings in by_name.values())))
    rows = []
    base_name = runs[0].name
    for opening_idx in opening_ids:
        values = []
        has_nonzero = False
        for run in runs:
            opening = by_name[run.name].get(opening_idx)
            avg = opening.avg_disc if opening is not None else 0.0
            values.append(avg)
            has_nonzero = has_nonzero or abs(avg) > 1.0e-9
        if not has_nonzero:
            continue
        spread = max(values) - min(values)
        rows.append((opening_idx, spread, values))

    rows.sort(key=lambda row: (-row[1], row[0]))
    if top > 0:
        rows = rows[:top]

    print("comparison\tbase={}\truns={}".format(base_name, ",".join(run.name for run in runs)))
    header = ["opening_idx"] + [run.name for run in runs] + [
        "{}_delta".format(run.name) for run in runs[1:]
    ]
    print("\t".join(header))
    for opening_idx, _spread, values in rows:
        base = values[0]
        fields = [str(opening_idx)]
        fields.extend("{:+.2f}".format(value) for value in values)
        fields.extend("{:+.2f}".format(value - base) for value in values[1:])
        print("\t".join(fields))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        type=parse_run_arg,
        required=True,
        metavar="NAME=PATH",
        help="saved kifu TSV from battle_parallel_nonstop_gtp.py; repeatable",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="maximum nonzero openings/comparison rows to print; use 0 for no limit",
    )
    args = parser.parse_args()

    try:
        runs = [summarize_run(name, path) for name, path in args.run]
    except (OSError, ValueError) as e:
        print("[ERROR] {}".format(e), file=sys.stderr)
        return 1

    for idx, run in enumerate(runs):
        if idx:
            print()
        print_run_summary(run, args.top)

    if len(runs) >= 2:
        print()
        print_comparison(runs, args.top)

    return 0


if __name__ == "__main__":
    sys.exit(main())
