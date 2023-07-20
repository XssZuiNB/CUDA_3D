#pragma once

#include <vector_types.h>

namespace gca
{
enum point_property
{
    invalid = 0,
    active = 1,
    inactive = 2
};

struct coordinates3
{
    float x;
    float y;
    float z;

    __host__ __device__ coordinates3()
        : x(0.0)
        , y(0.0)
        , z(0.0)
    {
    }

    __host__ __device__ coordinates3(float x_, float y_, float z_)
        : x(x_)
        , y(y_)
        , z(z_)
    {
    }

    __host__ __device__ coordinates3(const float3 &xyz)
        : x(xyz.x)
        , y(xyz.y)
        , z(xyz.z)
    {
    }

    __host__ __device__ coordinates3(const coordinates3 &other)
        : x(other.x)
        , y(other.y)
        , z(other.z)
    {
    }

    __host__ __device__ coordinates3 &operator=(const float3 &xyz)
    {
        x = xyz.x;
        y = xyz.y;
        z = xyz.z;
        return *this;
    }

    __host__ __device__ coordinates3 &operator=(const coordinates3 &other)
    {
        x = other.x;
        y = other.y;
        z = other.z;
        return *this;
    }

    __host__ __device__ coordinates3 operator+(const coordinates3 &other) const
    {
        return coordinates3(x + other.x, y + other.y, z + other.z);
    }

    __host__ __device__ coordinates3 operator+(const float3 &xyz) const
    {
        return coordinates3(x + xyz.x, y + xyz.y, z + xyz.z);
    }

    __host__ __device__ coordinates3 &operator+=(const coordinates3 &other)
    {
        x += other.x;
        y += other.y;
        z += other.z;
        return *this;
    }

    __host__ __device__ coordinates3 &operator+=(const float3 &xyz)
    {
        x += xyz.x;
        y += xyz.y;
        z += xyz.z;
        return *this;
    }

    __host__ __device__ coordinates3 operator-(const coordinates3 &other) const
    {
        return coordinates3(x - other.x, y - other.y, z - other.z);
    }

    __host__ __device__ coordinates3 operator-(const float3 &xyz) const
    {
        return coordinates3(x - xyz.x, y - xyz.y, z - xyz.z);
    }

    __host__ __device__ coordinates3 &operator-=(const coordinates3 &other)
    {
        x -= other.x;
        y -= other.y;
        z -= other.z;
        return *this;
    }

    __host__ __device__ coordinates3 &operator-=(const float3 &xyz)
    {
        x -= xyz.x;
        y -= xyz.y;
        z -= xyz.z;
        return *this;
    }

    template <typename T> __host__ __device__ coordinates3 operator*(T scalar) const
    {
        auto scalar_ = static_cast<float>(scalar);
        return coordinates3(x * scalar_, y * scalar_, z * scalar_);
    }

    template <typename T> __host__ __device__ coordinates3 operator/(T scalar) const
    {
        auto scalar_ = static_cast<float>(scalar);
        return coordinates3(x / scalar_, y / scalar_, z / scalar_);
    }

    __host__ __device__ bool operator==(const coordinates3 &other) const
    {
        return x == other.x && y == other.y && z == other.z;
    }

    __host__ __device__ bool operator==(const float3 &xyz) const
    {
        return x == xyz.x && y == xyz.y && z == xyz.z;
    }

    __host__ __device__ bool operator<(const coordinates3 &other) const
    {
        if (x != other.x)
            return x < other.x;

        else if (y != other.y)
            return y < other.y;

        else if (z != other.z)
            return z < other.z;

        return false;
    }

    __host__ __device__ bool operator!=(const coordinates3 &other) const
    {
        return !(*this == other);
    }

    __host__ __device__ bool operator!=(const float3 &xyz) const
    {
        return !(*this == xyz);
    }

    __host__ __device__ operator float3() const
    {
        return float3{.x = this->x, .y = this->y, .z = this->z};
    }

    // __host__ __device__ Eigen::Vector3f to_eigen_vector3f() for later if need to be used :)
};

struct color4
{
    uint32_t r;
    uint32_t g;
    uint32_t b;
    uint32_t a = 255;
};

struct __align__(16) point_t
{
    float3 coordinates;
    uint32_t r;
    uint32_t g;
    uint32_t b;
    point_property property;
};
} // namespace gca
