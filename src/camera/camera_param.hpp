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
} // namespace gca