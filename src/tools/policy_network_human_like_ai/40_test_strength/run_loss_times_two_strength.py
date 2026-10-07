#!/usr/bin/env python3
"""Play the completed 2 * disc_loss network against all 18 reference opponents."""
from pathlib import Path

import run_loss_plus_one_strength as base

HERE = Path(__file__).resolve().parent
TRAINING_RUN = HERE.parent / "10_train_policy_network/trained/conv8_bn_loss_times_two_e10_20261006"


class LossTimesTwoExtension(base.LossPlusOneExtension):
    def __init__(self, reference_dir, normal_reference_dir, model):
        super().__init__(reference_dir, normal_reference_dir, model, label="loss_times_two")

    def update_configuration(self, configuration):
        super().update_configuration(configuration)
        configuration["inputs"]["implementation_sources"]["loss_times_two_runner"] = base.battle.file_identity(Path(__file__))


def make_argparser():
    parser = base.make_argparser()
    parser.description = __doc__
    parser.set_defaults(model=TRAINING_RUN / "train/selected_model.h5",
                        output_dir=HERE / "output/conv8_bn_loss_times_two_single_20261006")
    return parser


def prepare(args):
    return base.prepare(args, extension_class=LossTimesTwoExtension)


def verify_completed_training(model):
    run_dir = model.resolve().parent.parent
    record = base.read_json(run_dir / "run.json")
    if (record.get("status") != "completed" or record.get("returncode") != 0
            or record.get("sample_weight_formula") != "2 * disc_loss"
            or Path(record.get("model", "")).resolve() != model.resolve()):
        raise ValueError("2 * disc_loss training must have finished successfully before matches start")
    dataset = base.read_json(model.parent / "prepared_dataset.json")
    if dataset["sample_weights"]["formula"] != "2 * disc_loss":
        raise ValueError("training data uses a different sample weight formula")


def main(argv=None):
    args = make_argparser().parse_args(argv)
    verify_completed_training(args.model)
    runtime_args, extension = prepare(args)
    if args.dry_run:
        print("Dry-run passed. No engines started; no results written.", flush=True)
        return
    with base.battle.OutputRunLock(runtime_args.output_dir):
        base.prepare_extension_output(runtime_args, extension)
        base.battle.run_tournament(runtime_args, extension=extension)


if __name__ == "__main__":
    main()
