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

enum param_type
{
    depth_intrinsics,
    color_intrinsics,
    depth2color_extrinsics,
    color2depth_extrinsics,
    width,
    height,
    depth_scale
};
} // namespace gca