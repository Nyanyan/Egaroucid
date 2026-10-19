"""Input identities shared by the completed under-800 fine-tuning experiment."""
from pathlib import Path
import json
import sys

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
WORKSPACE = HERE.parents[4]
DEFAULT_RUN = HERE / "output/quest_under800_20261010"
DEFAULT_RESULTS = HERE / "output/quest_under800_20261010_analysis"
MODEL_ID = "quest_finetuned_under800"
TOURNAMENT = BASE / "40_test_strength/output/gpw30_independent_xot_50sets"
WTHOR_REFERENCE = BASE / "20_test_with_wthor/output/gpw30_human_match_20261008"
for folder in (BASE, BASE / "00_data_processing", BASE / "20_test_with_wthor", BASE / "40_test_strength"):
    sys.path.insert(0, str(folder))
from wthor_dataset import file_sha256


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def validate_model(model):
    model = Path(model).resolve()
    run = model.parent
    status, summary, identity, dataset = [read(run / name) for name in
        ("run_status.json", "selected_summary.json", "training_identity.json", "prepared_dataset.json")]
    if (status["state"] != "complete" or status["returncode"] != 0
            or summary["epochs_ran"] != identity["settings"]["epochs"]
            or summary["test_used_for_selection"] is not False
            or summary["rating_range"] != {"min_inclusive": None, "max_exclusive": 800}
            or dataset["augmentation_factor"] != dict(train=8, val=1, test=1)
            or not 1 <= summary["adopted_epoch"] <= summary["epochs_ran"]):
        raise ValueError("Expected completed under-800 fine-tuning selected without test data")
    if file_sha256(model) != summary["model_sha256"] or status["model_sha256"] != summary["model_sha256"]:
        raise ValueError("Saved model differs from the training result")
    base = identity["base_model"]
    if summary["base_model"] != base or file_sha256(Path(base["path"])) != base["sha256"]:
        raise ValueError("Pretrained model identity differs")
    source = Path(identity["source_directory"])
    if file_sha256(source / "manifest.json") != identity["source_manifest_sha256"]:
        raise ValueError("Quest dataset manifest differs from training")
    if read(source / "manifest.json")["splits"]["test"]["positions"] != 6692:
        raise ValueError("Expected the held-out 6692 Quest positions")
    return {name: dict(path=str(run / name), sha256=file_sha256(run / name)) for name in
            ("selected_model.h5", "selected_summary.json", "training_identity.json", "prepared_dataset.json", "run_status.json")}
