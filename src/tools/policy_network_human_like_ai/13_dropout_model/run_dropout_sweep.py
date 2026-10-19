"""Apply inference dropout to fixed pretrained WTHOR weights and evaluate Quest."""
import argparse
import datetime
import math
import os
from pathlib import Path
import time

from sweep_common import HERE, BASE, RATES, BANDS, data_root, read, write, sha256, write_csv, run_lock
from evaluate_dropout_quest import inspect_band, evaluate_band

DEFAULT_MODEL = HERE.parents[4] / "experiment_bat/20261002/results/train/selected_model.h5"


def make_argparser():
    root = data_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--quest-root", type=Path, default=root / "othello_quest_split" if root else None)
    parser.add_argument("--dropout-rates", nargs="+", type=float, default=list(RATES))
    parser.add_argument("--output-dir", type=Path, default=HERE / "output" /
                        ("inference_dropout_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")))
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=613)
    parser.add_argument("--progress-interval-sec", type=float, default=15)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def build_identity(args):
    if (args.quest_root is None or args.batch_size <= 0 or not math.isfinite(args.progress_interval_sec)
            or args.progress_interval_sec <= 0 or not args.dropout_rates
            or any(not math.isfinite(rate) or not 0 <= rate < 1 for rate in args.dropout_rates)
            or len(set(args.dropout_rates)) != len(args.dropout_rates)):
        raise ValueError("Specify Quest data, positive batch size/interval, and distinct rates in [0, 1)")
    model = args.base_model.resolve()
    selected, trained, prepared = [read(model.parent / name) for name in
                                  ("selected_summary.json", "training_identity.json", "prepared_dataset.json")]
    if (selected["spec"]["architecture"] != "conv8_bn" or selected["spec"]["dropout"] != 0
            or prepared.get("sample_weights") or not prepared["dataset"].startswith("WTHOR")
            or selected["epochs_ran"] != trained["settings"]["epochs"]):
        raise ValueError("Expected the existing unweighted WTHOR Conv8+BN model")
    datasets = [inspect_band(args.quest_root, band)[1] for band in BANDS]
    sources = [HERE / name for name in ("dropout_model.py", "sweep_common.py", "run_dropout_sweep.py", "evaluate_dropout_quest.py")]
    sources += [BASE / "00_data_processing" / name for name in
                ("policy_data.py", "pipeline_common.py", "wthor_dataset.py")]
    sources += [BASE / "policy_accuracy.py", BASE / "10_train_policy_network/prepared_policy_data.py"]
    return dict(version=2, operation="inference_only_no_training", dropout_rates=args.dropout_rates,
        base_model=dict(path=str(model), sha256=sha256(model), architecture="conv8_bn",
            adopted_epoch=selected["adopted_epoch"], original_precision=selected["precision"],
            dataset=prepared["dataset"], evidence={name: sha256(model.parent / name) for name in
            ("selected_summary.json", "training_identity.json", "prepared_dataset.json")}),
        quest_root=str(args.quest_root.resolve()), datasets=datasets, rating_bands=list(BANDS),
        batch_size=args.batch_size, seed=args.seed, stochastic_passes_per_position=1,
        inference_precision="float32_for_all_rates_to_avoid_half_precision_overflow",
        dropout="inverted element-wise dropout after all eight hidden convolution ReLUs and the dense ReLU",
        batch_normalization="frozen original moving statistics; inference mode",
        ranking="legal argmax of pre-softmax logits; ascending policy index breaks ties",
        agreement="board-preserving symmetries included", trainable_weights_changed=False,
        sources={str(path.relative_to(BASE)): sha256(path) for path in sources})


def write_results(output, rows, identity, complete=False):
    for band in BANDS:
        samples = {(row["positions"], row["test_records_sha256"]) for row in rows if row["rating_band"] == band}
        if len(samples) > 1:
            raise ValueError(f"Rates used different test records: {band}")
    write_csv(output / "summary.csv", rows)
    rates = identity["dropout_rates"]
    matrix = []
    lines = ["# WTHOR学習済みモデルへの推論時Dropout", "",
        "既存のWTHOR学習済みConv8＋BNの重みを固定し、推論時だけDropoutを適用した結果です。",
        "畳み込み層8個と隠れ全結合層1個のReLU直後に同じ率を適用します。BNは推論モードで統計値を固定しています。",
        "Questの水増し前test全件を使い、各局面につき1回だけ推論しています。",
        "率0.8での半精度のオーバーフローを避けるため、全条件の推論計算をfloat32に統一しています。重みの値は元モデルと同じです。",
        "合法手のTop-1を選び、盤面を保つ対称変換で等価な着手も一致に含めます。",
        "高いdropout率でsoftmaxが0へ丸められる影響を避けるため、合法手の順位はsoftmax前の値から求めます。", "",
        f"乱数seedは{identity['seed']}＋レート帯の番号（0～6）、バッチサイズは{identity['batch_size']}です。",
        f"元モデルは{identity['base_model']['dataset']}で学習済みの{identity['base_model']['adopted_epoch']}エポック目をそのまま使用しています。", "",
        "| レート帯 | test局面数 | " + " | ".join(f"dropout {rate:.1f}" for rate in rates) + " |",
        "| --- | ---: | " + " | ".join("---:" for _ in rates) + " |"]
    for info in identity["datasets"]:
        band = info["rating_band"]
        entry = dict(rating_band=band, positions=info["positions"])
        cells = []
        for rate in rates:
            matches = [row for row in rows if row["rating_band"] == band and row["dropout_rate"] == rate]
            if len(matches) > 1:
                raise ValueError("Duplicate rate/band results")
            entry[f"dropout_{rate:.1f}_accuracy_percent"] = matches[0]["top1_accuracy_percent"] if matches else None
            cells.append(f"{matches[0]['top1_accuracy_percent']:.3f}%" if matches else "計算待ち")
        matrix.append(entry)
        lines.append(f"| {band} | {info['positions']:,} | " + " | ".join(cells) + " |")
    write_csv(output / "accuracy_matrix.csv", matrix)
    lines += ["", "率0.0は同じ重みをfloat32で計算するDropoutなしモデルと一致することをtest全局面で検証しています。",
              "各帯の評価後に全重みとBNの統計値が元モデルと同じであることを確認しています。",
              "率0より大きい結果は、上記seedでのDropoutマスク1回分の測定値です。",
              "状態: " + ("全条件の評価が完了しました。" if complete else "実行中です。"), ""]
    temporary = output / "RESULTS.tmp.md"
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(output / "RESULTS.md")
    write(output / "summary.json", dict(status="complete" if complete else "running", results=rows,
        dropout_rates=rates, rating_bands=list(BANDS), updated_at_unix=time.time()))


class LogitPredictor:
    """Rank logits instead of rounded softmax probabilities at high dropout rates."""
    def __init__(self, model):
        import tensorflow as tf
        self.view = tf.keras.Model(model.inputs, model.get_layer("policy_logits60").output)

    def predict(self, features, batch_size=512, verbose=0):
        import numpy as np
        from paper_policy_models import PAPER_POLICIES
        logits = self.view.predict(features, batch_size=batch_size, verbose=verbose)
        scores = np.zeros((len(features), 64), dtype=np.float32)
        scores[:, PAPER_POLICIES] = logits
        return scores


def run(args):
    identity = build_identity(args)
    if args.dry_run:
        print(f"Inference only: {identity['base_model']['path']}; rates={args.dropout_rates}", flush=True)
        for info in identity["datasets"]:
            print(f"Quest {info['rating_band']}: all {info['positions']:,} test positions", flush=True)
        return
    import numpy as np
    import tensorflow as tf
    from dropout_model import build_inference_dropout_model
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    pretrained = tf.keras.models.load_model(args.base_model, compile=False)
    original_weights = pretrained.get_weights()
    reference_float32 = build_inference_dropout_model(pretrained, 0, args.seed)
    output = args.output_dir.resolve()
    started, total = time.time(), len(args.dropout_rates) * len(BANDS)
    with run_lock(output):
        path = output / "experiment_identity.json"
        if path.exists():
            if not args.resume or read(path) != identity:
                raise ValueError("Use --resume with identical pretrained model/data/code/settings")
        else:
            if any(item.name != "run.lock" for item in output.iterdir()):
                raise FileExistsError("Choose an empty output directory")
            write(path, identity)
        rows = []
        def status(state, **extra):
            elapsed = time.time()-started
            eta = elapsed/len(rows)*(total-len(rows)) if rows else None
            write(output / "run_status.json", dict(state=state, pid=os.getpid(), updated_at_unix=time.time(),
                started_at_unix=started, elapsed_sec=elapsed, eta_sec=eta,
                completed_evaluations=len(rows), total_evaluations=total, **extra))
            print(f"[sweep] {state}: {len(rows)}/{total} | elapsed {elapsed:.1f}s | "
                  f"ETA {eta:.1f}s" if eta is not None else f"[sweep] {state}: 0/{total} | warming up", flush=True)
        try:
            write_results(output, rows, identity)
            for rate in args.dropout_rates:
                name = f"dropout_{int(round(rate*100)):03d}"
                for index, band in enumerate(BANDS):
                    tf.keras.backend.clear_session()
                    tf.keras.utils.set_random_seed(args.seed + index)
                    model = build_inference_dropout_model(pretrained, rate, args.seed)
                    model_file = output / "models" / (name + ".h5")
                    if not model_file.exists():
                        model_file.parent.mkdir(parents=True, exist_ok=True)
                        model.save(model_file, include_optimizer=False)
                    paths, info = inspect_band(args.quest_root, band)
                    if info != identity["datasets"][index]:
                        raise ValueError("Quest dataset changed during evaluation")
                    evidence = dict(dropout_rate=rate, adopted_epoch=identity["base_model"]["adopted_epoch"],
                        model_sha256=sha256(model_file), pretrained_sha256=identity["base_model"]["sha256"],
                        inference_seed=args.seed+index, experiment=sha256(path))
                    status("running", dropout_rate=rate, rating_band=band)
                    rows.append(evaluate_band(LogitPredictor(model), paths, info, output / "quest" / name / band,
                        args.batch_size, args.progress_interval_sec, evidence,
                        reference_model=reference_float32 if rate == 0 else None))
                    if any(not np.array_equal(a, b) for a, b in zip(original_weights, model.get_weights())):
                        raise ValueError("Weights or BN statistics changed during inference")
                    write_results(output, rows, identity)
                    status("running", dropout_rate=rate, rating_band=band)
            if sha256(args.base_model) != identity["base_model"]["sha256"]:
                raise ValueError("Original model file changed")
            write_results(output, rows, identity, complete=True)
            status("complete", all_weights_and_bn_statistics_unchanged=True, zero_rate_matches_original=True)
        except BaseException as error:
            status("interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=repr(error))
            raise


def main():
    run(make_argparser().parse_args())


if __name__ == "__main__":
    main()
