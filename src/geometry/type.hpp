#pragma once

#include "util/math.cuh"

#include <iostream>

#include <vector_types.h>

namespace gca
{
typedef int32_t index_t;
typedef int32_t counter_t;

enum point_property : uint8_t
{
    invalid = 0,
    active = 1,
    inactive = 2
};

struct color3
{
    float r;
    float g;
    float b;

    __forceinline__ __host__ __device__ operator float3() const
    {
        return make_float3(r, g, b);
    }

    __forceinline__ __host__ __device__ color3 operator+(const color3 &other) const
    {
        return color3{r + other.r, g + other.g, b + other.b};
    }

    __forceinline__ __host__ __device__ color3 &operator+=(const color3 &other)
    {
        r += other.r;
        g += other.g;
        b += other.b;
        return *this;
    }

    __forceinline__ __host__ __device__ float to_intensity() const
    {
        return 0.2126 * r + 0.7152 * g + 0.0722 * b;
    }

    __forceinline__ __host__ __device__ float get_average() const
    {
        return (r + g + b) / 3;
    }

    template <typename T> __forceinline__ __host__ __device__ color3 operator/(T n) const
    {
        auto _n = (float)n;
        return color3{r / _n, g / _n, b / _n};
    }
};

struct point_t
{
    float3 coordinates;
    color3 color;
    point_property property;
};

__forceinline__ std::ostream &operator<<(std::ostream &os, const point_t &point)
{
    os << "Coordinates: x = " << point.coordinates.x << "\n"
       << "             y = " << point.coordinates.y << "\n"
       << "             z = " << point.coordinates.z << "\n"
       << "R: " << point.color.r << " G: " << point.color.g << " B: " << point.color.b << "\n"
       << "Property: " << (point.property == gca::point_property::active ? "Active" : "Inactive")
       << "\n";
    return os;
}
} // namespace gca
