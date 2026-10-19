"""Append the Quest-trained policy to the completed GPW30 tournament.

Import all 21,750 existing sets unchanged and play 1,500 new color-swapped sets.
Re-running the identical command resumes the saved tournament.
"""
from pathlib import Path
import copy
import json
import sys

HERE = Path(__file__).resolve().parent
STRENGTH = HERE.parent / "40_test_strength"
sys.path.insert(0, str(STRENGTH))

import run_gpw_strength as gpw
import battle_blend_strength as battle
from extend_strength_tournament import ModelExtension
from strength_tournament import PlayerSpec, ResultStore, make_manifest, make_match_set_tasks

from experiment_common import validate_model, DEFAULT_RUN, DEFAULT_RESULTS, MODEL_ID

NEW_NAME = MODEL_ID


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


class QuestExtension(gpw.PaperPlayers):
    resource_alphas = (1.0,) * 12
    policy_server_count = 3
    missing_imports = ModelExtension.missing_imports

    def __init__(self, args, evidence):
        super().__init__(args, evidence)
        self.reference_dir = args.reference_dir.resolve()
        self.quest_model = args.quest_model.resolve()
        self.manifest = read(self.reference_dir / "strength_manifest.json")
        self.runtime = read(self.reference_dir / "strength_runtime_manifest.json")
        if make_manifest(self.manifest["configuration"]) != self.manifest:
            raise ValueError("Reference manifest fingerprint differs")
        if (make_manifest(self.runtime["configuration"]) != self.runtime
                or self.runtime["configuration"]["experiment_id"] != self.manifest["experiment_id"]):
            raise ValueError("Reference runtime fingerprint differs")
        original = gpw.PaperPlayers(args, evidence)
        levels, alphas = battle.validate_args(args, original.resource_alphas)
        openings = battle.read_openings(args.openings, args.seed)
        old_tasks = make_match_set_tasks(30, openings, args.match_sets_per_pair, args.seed)
        configuration = battle.build_manifest_configuration(args, levels, alphas, openings, old_tasks)
        original.update_configuration(configuration)
        if configuration != self.manifest["configuration"]:
            raise ValueError("Current GPW30 settings, inputs, code or package versions differ from the reference")
        expected_names = [spec.name for spec in original.add_specs(args, battle.build_player_specs(args, levels, alphas, 0), 0)]
        results = read(self.reference_dir / "strength_results.json")
        if (results["experiment_id"] != self.manifest["experiment_id"]
                or results["completed_match_sets"] != results["total_match_sets"]
                or results["total_match_sets"] != len(old_tasks)
                or results["total_match_sets"] != 21750
                or [player["name"] for player in results["players"]] != expected_names):
            raise ValueError("Reference must be the completed 30-player, 50-sets-per-pair tournament")
        self.reference_files = {name: battle.file_identity(self.reference_dir / name) for name in
                                ("strength_manifest.json", "strength_runtime_manifest.json",
                                 "strength_games.jsonl", "strength_results.json")}
        _, self.results = ResultStore(self.reference_dir).load({t.task_id: t for t in old_tasks}, read_only=True)
        if len(self.results) != len(old_tasks):
            raise ValueError("Reference game log is incomplete")
        self.old_names = expected_names
        self.quest_evidence = validate_model(self.quest_model)
        if args.output_dir.resolve() == self.reference_dir or self.reference_dir in args.output_dir.resolve().parents:
            raise ValueError("Output must be separate from the reference tournament")
        if args.max_match_sets is not None:
            raise ValueError("Do not limit the combined tournament task plan")
        print("Reference verified: 30 players, 21,750 saved sets. New schedule: 30 x 50 sets = 3,000 games.", flush=True)

    def add_specs(self, args, original_specs, port, original_port=0):
        specs = super().add_specs(args, original_specs, port, original_port)
        quest_port = self.extra_policy_servers["quest"].port if port and self.extra_policy_servers else 0
        command = (sys.executable, str(STRENGTH / "policy_opponent_gtp.py"),
                   "--mode", "geometric", "--beta", "1.0", "--host", args.policy_server_host,
                   "--weighted-port", str(quest_port), "--output", "logits",
                   "--timeout-sec", str(args.policy_server_timeout_sec))
        specs.append(PlayerSpec(NEW_NAME, command, args.blend_processes_per_player, True, 1.0))
        if [spec.name for spec in specs] != self.old_names + [NEW_NAME]:
            raise ValueError("The reference player order must remain unchanged")
        return specs

    def start_server(self, args, manager, run_id):
        loss_a = super().start_server(args, manager, run_id)
        settings = copy.copy(args)
        settings.policy_model = self.quest_model
        self.extra_policy_servers["quest"] = battle.start_policy_server(settings, manager, run_id + "_quest")
        return loss_a

    def validate_runtime(self, normal, loss_a):
        super().validate_runtime(normal, loss_a)
        ModelExtension.validate_runtime(self, normal, loss_a)

    def update_configuration(self, configuration):
        super().update_configuration(configuration)
        configuration["participants"]["additional_policy"] = dict(name=NEW_NAME, kind="standalone_policy",
                selection="argmax over legal moves", appended_after_original_players=True, training=self.quest_evidence)
        configuration["inputs"]["quest_model"] = battle.file_identity(self.quest_model)
        configuration["inputs"]["implementation_sources"]["quest_extension"] = battle.file_identity(Path(__file__))
        configuration["policy_inference"]["server_count"] = 4
        configuration["policy_inference"]["models"].append("quest")
        configuration["imported_reference"] = dict(directory=str(self.reference_dir),
                experiment_id=self.manifest["experiment_id"], files=self.reference_files,
                imported_match_sets=len(self.results), original_configuration=self.manifest["configuration"])


def make_args(argv=None):
    parser = gpw.make_argparser()
    parser.description = __doc__
    parser.set_defaults(output_dir=DEFAULT_RESULTS / "strength", resume=True)
    parser.add_argument("--reference-dir", type=Path, default=STRENGTH / "output/gpw30_independent_xot_50sets")
    parser.add_argument("--quest-model", type=Path, default=DEFAULT_RUN / "selected_model.h5")
    return parser.parse_args(argv)


def main():
    args = make_args()
    original = gpw.prepare(args)
    extension = QuestExtension(args, original.evidence)
    with battle.OutputRunLock(args.output_dir):
        battle.run_tournament(args, extension=extension)


if __name__ == "__main__":
    main()
