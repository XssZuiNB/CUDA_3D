#pragma once

#include "util/math.cuh"

namespace gca
{
mat4x4 solve_JTJ_JTr(const mat6x6 &JTJ, const mat6x1 &JTr);
}