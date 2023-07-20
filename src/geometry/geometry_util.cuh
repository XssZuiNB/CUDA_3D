#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <thrust/extrema.h>

#include "geometry/type.hpp"

namespace gca
{
struct min_bound_functor
{
    __forceinline__ __device__ gca::point_t operator()(const gca::point_t &first,
                                                       const gca::point_t &second)
    {
        gca::point_t temp;
        temp.coordinates.x = thrust::min(first.coordinates.x, second.coordinates.x);
        temp.coordinates.y = thrust::min(first.coordinates.y, second.coordinates.y);
        temp.coordinates.z = thrust::min(first.coordinates.z, second.coordinates.z);
        return temp;
    }
};

struct max_bound_functor
{
    __forceinline__ __device__ gca::point_t operator()(const gca::point_t &first,
                                                       const gca::point_t &second)
    {
        gca::point_t temp;
        temp.coordinates.x = thrust::max(first.coordinates.x, second.coordinates.x);
        temp.coordinates.y = thrust::max(first.coordinates.y, second.coordinates.y);
        temp.coordinates.z = thrust::max(first.coordinates.z, second.coordinates.z);
        return temp;
    }
};

struct compute_voxel_key_functor
{
    compute_voxel_key_functor(const float3 &voxel_grid_min_bound, const float voxel_size)
        : m_voxel_grid_min_bound(voxel_grid_min_bound)
        , m_voxel_size(voxel_size)
    {
    }

    const float3 m_voxel_grid_min_bound;
    const float m_voxel_size;

    __forceinline__ __device__ int3 operator()(const gca::point_t &point)
    {
        int3 ref_coord;
        ref_coord.x =
            __float2int_rd((point.coordinates.x - m_voxel_grid_min_bound.x) / m_voxel_size);
        ref_coord.y =
            __float2int_rd((point.coordinates.y - m_voxel_grid_min_bound.y) / m_voxel_size);
        ref_coord.z =
            __float2int_rd((point.coordinates.z - m_voxel_grid_min_bound.z) / m_voxel_size);
        return ref_coord;
    }
};

struct compare_voxel_key_functor : public thrust::binary_function<int3, int3, bool>
{
    __forceinline__ __host__ __device__ bool operator()(const int3 &lhs, const int3 &rhs) const
    {
        if (lhs.x != rhs.x)
            return lhs.x < rhs.x;

        else if (lhs.y != rhs.y)
            return lhs.y < rhs.y;

        else if (lhs.z != rhs.z)
            return lhs.z < rhs.z;

        return false;
    }
};

struct voxel_key_equal_functor : public thrust::binary_function<int3, int3, bool>
{
    __forceinline__ __host__ __device__ bool operator()(const int3 &lhs, const int3 &rhs) const
    {
        return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z;
    }
};

__forceinline__ float3 cuda_compute_min_bound(const thrust::device_vector<gca::point_t> &points)
{
    gca::point_t init{.coordinates{.x = FLT_MAX, .y = FLT_MAX, .z = FLT_MAX}};
    return thrust::reduce(points.begin(), points.end(), init, min_bound_functor()).coordinates;
}

__forceinline__ float3 cuda_compute_max_bound(const thrust::device_vector<gca::point_t> &points)
{
    gca::point_t init{.coordinates{.x = FLT_MIN, .y = FLT_MIN, .z = FLT_MIN}};
    return thrust::reduce(points.begin(), points.end(), init, max_bound_functor()).coordinates;
}

__forceinline__ ::cudaError_t cuda_compute_voxel_keys(
    thrust::device_vector<int3> &keys, const thrust::device_vector<gca::point_t> &points,
    const float3 &voxel_grid_min_bound, const float voxel_size)
{
    if (keys.size() != points.size())
    {
        return ::cudaErrorInvalidValue;
    }

    thrust::transform(points.begin(), points.end(), keys.begin(),
                      compute_voxel_key_functor(voxel_grid_min_bound, voxel_size));

    return cudaGetLastError();
}
} // namespace gca