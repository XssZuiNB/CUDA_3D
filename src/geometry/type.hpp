#pragma once

namespace gca
{
struct point_t
{
    float x, y, z;
    uint8_t b, g, r;
    uint8_t __align_not_use;
};
} // namespace gca