#pragma once
#include <iostream>
#include "setting.hpp"
#if USE_INT_EVAL
    #include "evaluate_int.hpp"
#else
    #include "evaluate_float.hpp"
#endif