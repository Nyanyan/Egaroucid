#pragma once
#include "setting.hpp"
#if USE_PRE_CALCULATED_EVALUATION
    #include "evaluation_pre_calc.hpp"
#else
    #include "evaluate_nn.hpp"
#endif