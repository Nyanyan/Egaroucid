/* Model initialization, custom resource directories and failure/override tests. */
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include "../../engine/engine_all.hpp"

void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

int main(int argc, char **argv) {
    try {
        require(argc == 3, "usage: test_move_ordering_model_init <resource_dir> <scratch_dir>");
        namespace fs = std::filesystem;
        const fs::path resources = fs::absolute(argv[1]);
        const fs::path scratch = fs::absolute(argv[2]);
        require(!fs::exists(scratch), "scratch directory must be new");
        fs::create_directories(scratch);
        const auto end_model = (resources / "eval_move_ordering_end.egev").string();
        const auto main_model = (resources / "eval.egev2").string();
#if USE_SEPARATE_MID_MOVE_ORDERING_EVALUATION && !defined(USE_NNUE_EVALUATION)
        eval_move_ordering_auto_requested = true;
        require(evaluate_init(main_model, end_model, false), "default initialization");
        require(eval_move_ordering_auto_enabled, "default automatic activation");

        // Missing models must fail with all banks disabled, even after a
        // successful initialization. The main filename need not exist here.
        const auto custom_main = (scratch / "eval.egev2").string();
        require(!initialize_auto_move_ordering_models(custom_main.c_str(), false), "missing Offset fails");
        require(!eval_move_ordering_auto_enabled && !eval_move_ordering_mid_enabled && !eval_move_ordering_aux_enabled, "failure clears previous models");
        fs::copy_file(resources / "offset0_16.egev2", scratch / "offset0_16.egev2");
        require(!initialize_auto_move_ordering_models(custom_main.c_str(), false), "missing Decay fails");
        require(!eval_move_ordering_mid_enabled && !eval_move_ordering_aux_enabled, "partially loaded models stay disabled");
        { std::ofstream invalid(scratch / "decay_half16.egev2", std::ios::binary); invalid << "invalid"; }
        require(!initialize_auto_move_ordering_models(custom_main.c_str(), false), "invalid Decay fails");

        eval_move_ordering_auto_requested = false;
        require(initialize_auto_move_ordering_models(custom_main.c_str(), false), "override needs no automatic models");
        require(!eval_move_ordering_auto_enabled && !eval_move_ordering_mid_enabled && !eval_move_ordering_aux_enabled, "override stays ordinary");

        fs::copy_file(resources / "decay_half16.egev2", scratch / "decay_half16.egev2", fs::copy_options::overwrite_existing);
        fs::copy_file(resources / "eval.egev2", scratch / "eval.egev2");
        fs::current_path(scratch);
        eval_move_ordering_auto_requested = true;
        require(evaluate_init("eval.egev2", end_model.c_str(), false), "bare relative main filename loads adjacent models");
        require(eval_move_ordering_auto_enabled && eval_move_ordering_mid_enabled && eval_move_ordering_aux_enabled, "custom models activated");
        eval_move_ordering_auto_requested = false;
        require(evaluate_init("eval.egev2", end_model.c_str(), false), "reinitialization without automatic policy");
        require(!eval_move_ordering_auto_enabled && !eval_move_ordering_mid_enabled && !eval_move_ordering_aux_enabled, "reinitialization clears model activation");
        std::cout << "PASS model_init default/custom/missing/invalid/override/reinit\n";
#else
        require(initialize_auto_move_ordering_models("missing/eval.egev2", false), "unsupported builds require no extra models");
        require(!eval_move_ordering_auto_enabled, "unsupported builds stay ordinary");
        std::cout << "PASS disabled_model_init\n";
#endif
    } catch (const std::exception &error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
