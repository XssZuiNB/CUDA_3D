#pragma once

// #include <Eigen/Core>
#include <vector_types.h>

namespace gca
{
enum class point_property // make the size of enum 1 byte
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
        return coordinates3(x * scalar, y * scalar, z * scalar);
    }

    template <typename T> __host__ __device__ coordinates3 operator/(T scalar) const
    {
        return coordinates3(x / scalar, y / scalar, z / scalar);
    }

    __host__ __device__ bool operator==(const coordinates3 &other) const
    {
        return x == other.x && y == other.y && z == other.z;
    }

    __host__ __device__ bool operator==(const float3 &xyz) const
    {
        return x == xyz.x && y == xyz.y && z == xyz.z;
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

struct __align__(16) point_t
{
    float3 coordinates;
    uint32_t r, g, b;
    point_property property;
};
} // namespace gca