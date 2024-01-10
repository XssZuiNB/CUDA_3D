#pragma once

#include "geometry/type.hpp"
#include "util/math.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/remove.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

namespace gca
{
/************************** Compute min and max bound of a point cloud **************************/
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

struct min_max_bound_functor
{
    __forceinline__ __device__ thrust::tuple<gca::point_t, gca::point_t> operator()(
        const thrust::tuple<gca::point_t, gca::point_t> &first,
        const thrust::tuple<gca::point_t, gca::point_t> &second)
    {
        auto min = min_bound_functor()(thrust::get<0>(first), thrust::get<0>(second));
        auto max = max_bound_functor()(thrust::get<1>(first), thrust::get<1>(second));

        return thrust::make_tuple(min, max);
    }
};

struct add_points_coordinates_functor
{
    __forceinline__ __device__ gca::point_t operator()(const gca::point_t &first,
                                                       const gca::point_t &second)
    {
        gca::point_t temp;
        temp.coordinates = make_float3(first.coordinates.x + second.coordinates.x,
                                       first.coordinates.y + second.coordinates.y,
                                       first.coordinates.z + second.coordinates.z);
        return temp;
    }
};

__forceinline__ float3 cuda_compute_min_bound(const thrust::device_vector<gca::point_t> &points)
{
    gca::point_t init{.coordinates{.x = FLT_MAX, .y = FLT_MAX, .z = FLT_MAX}};
    auto min_bound =
        thrust::reduce(points.begin(), points.end(), init, min_bound_functor()).coordinates;

    return min_bound;
}

__forceinline__ float3 cuda_compute_max_bound(const thrust::device_vector<gca::point_t> &points)
{
    gca::point_t init{.coordinates{.x = FLT_MIN, .y = FLT_MIN, .z = FLT_MIN}};
    auto max_bound =
        thrust::reduce(points.begin(), points.end(), init, max_bound_functor()).coordinates;

    return max_bound;
}

__forceinline__ thrust::pair<float3, float3> cuda_compute_min_max_bound(
    const thrust::device_vector<gca::point_t> &points)
{
    gca::point_t init_min{.coordinates{.x = FLT_MAX, .y = FLT_MAX, .z = FLT_MAX}};
    gca::point_t init_max{.coordinates{.x = FLT_MIN, .y = FLT_MIN, .z = FLT_MIN}};
    auto zipped_init = thrust::make_tuple(init_min, init_max);

    auto zipped_iter_begin =
        thrust::make_zip_iterator(thrust::make_tuple(points.begin(), points.begin()));
    auto zipped_iter_end =
        thrust::make_zip_iterator(thrust::make_tuple(points.end(), points.end()));

    auto result =
        thrust::reduce(zipped_iter_begin, zipped_iter_end, zipped_init, min_max_bound_functor());

    return thrust::make_pair(thrust::get<0>(result).coordinates,
                             thrust::get<1>(result).coordinates);
}

__forceinline__ float3 cuda_compute_centroid(const thrust::device_vector<gca::point_t> &points)
{
    gca::point_t init{.coordinates{.x = 0.0f, .y = 0.0f, .z = 0.0f}};
    return thrust::reduce(points.begin(), points.end(), init, add_points_coordinates_functor())
               .coordinates /
           points.size();
}

/******************************* Functor check if a point is valid ******************************/
struct check_is_invalid_point_functor
{
    __forceinline__ __device__ bool operator()(gca::point_t p)
    {
        return p.property == gca::point_property::invalid;
    }
};

__forceinline__ void remove_invalid_points(thrust::device_vector<gca::point_t> &points)
{
    auto new_size =
        thrust::remove_if(points.begin(), points.end(), check_is_invalid_point_functor());

    points.resize(new_size - points.begin());
}

/************************************* Transform point cloud ************************************/
struct transform_point_functor
{
    transform_point_functor(const mat4x4 &trans_mat)
    {
        trans_mat.get_block<3, 3>(0, 0, m_rotation_mat);
        trans_mat.get_block<3, 1>(0, 3, m_translation_vec);
    }

    mat3x3 m_rotation_mat;
    mat3x1 m_translation_vec;

    __forceinline__ __device__ void operator()(gca::point_t &p) const
    {
        mat3x1 coordinate_mat(p.coordinates);
        p.coordinates = m_rotation_mat * coordinate_mat + m_translation_vec;
    }
};

struct transform_normal_functor
{
    transform_normal_functor(const mat4x4 &trans_mat)
    {
        trans_mat.get_block<3, 3>(0, 0, m_rotation_mat);
    }

    mat3x3 m_rotation_mat;

    __forceinline__ __device__ void operator()(float3 &n) const
    {
        mat3x1 coordinate_mat(n);
        n = m_rotation_mat * coordinate_mat;
    }
};

__forceinline__ void cuda_transform_point(thrust::device_vector<gca::point_t> &src,
                                          const mat4x4 &trans_matrix)
{
    thrust::for_each(src.begin(), src.end(), transform_point_functor(trans_matrix));
}

__forceinline__ void cuda_transform_normals(thrust::device_vector<float3> &src,
                                            const mat4x4 &trans_matrix)
{
    thrust::for_each(src.begin(), src.end(), transform_normal_functor(trans_matrix));
}
} // namespace gca
