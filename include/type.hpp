#pragma once

namespace gca
{
struct intrinsics
{
    float fx;
    float fy;
    float cx;
    float cy;
};

struct extrinsics
{

    float rotation[9];
    float translation[3];
};

enum frame_format
{
    BGR8,
    RGB8,
    Z16 // For depth
};

enum device_type
{
    REALSENSE,
};

struct point_t
{
    float x, y, z;
    uint8_t b, g, r;
};
} // namespace gca
