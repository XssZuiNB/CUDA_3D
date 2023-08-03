#pragma once

#include "geometry/type.hpp"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <thrust/extrema.h>
#include <thrust/remove.h>
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

__forceinline__ float3 cuda_compute_min_bound(const thrust::device_vector<gca::point_t> &points,
                                              ::cudaStream_t stream = cudaStreamDefault)
{
    auto exec_policy = thrust::cuda::par.on(stream);

    gca::point_t init{.coordinates{.x = FLT_MAX, .y = FLT_MAX, .z = FLT_MAX}};
    auto min_bound =
        thrust::reduce(exec_policy, points.begin(), points.end(), init, min_bound_functor())
            .coordinates;
    cudaStreamSynchronize(stream);

    return min_bound;
}

__forceinline__ float3 cuda_compute_max_bound(const thrust::device_vector<gca::point_t> &points,
                                              ::cudaStream_t stream = cudaStreamDefault)
{
    auto exec_policy = thrust::cuda::par.on(stream);

    gca::point_t init{.coordinates{.x = FLT_MIN, .y = FLT_MIN, .z = FLT_MIN}};
    auto max_bound =
        thrust::reduce(exec_policy, points.begin(), points.end(), init, max_bound_functor())
            .coordinates;
    cudaStreamSynchronize(stream);

    return max_bound;
}

__forceinline__ thrust::pair<float3, float3> cuda_compute_min_max_bound(
    const thrust::device_vector<gca::point_t> &points, ::cudaStream_t stream = cudaStreamDefault)
{
    auto exec_policy = thrust::cuda::par.on(stream);

    gca::point_t init_min{.coordinates{.x = FLT_MAX, .y = FLT_MAX, .z = FLT_MAX}};
    gca::point_t init_max{.coordinates{.x = FLT_MIN, .y = FLT_MIN, .z = FLT_MIN}};
    auto zipped_init = thrust::make_tuple(init_min, init_max);

    auto zipped_iter_begin =
        thrust::make_zip_iterator(thrust::make_tuple(points.begin(), points.begin()));
    auto zipped_iter_end =
        thrust::make_zip_iterator(thrust::make_tuple(points.end(), points.end()));

    auto result = thrust::reduce(exec_policy, zipped_iter_begin, zipped_iter_end, zipped_init,
                                 min_max_bound_functor());
    cudaStreamSynchronize(stream);

    return thrust::make_pair(thrust::get<0>(result).coordinates,
                             thrust::get<1>(result).coordinates);
}

/******************************* Functor check if a point is valid ******************************/
struct check_is_invalid_point_functor
{
    __forceinline__ __device__ bool operator()(gca::point_t p)
    {
        return p.property == gca::point_property::invalid;
    }
};

__forceinline__ void remove_invalid_points(thrust::device_vector<gca::point_t> &points,
                                           ::cudaStream_t stream = cudaStreamDefault)
{
    auto exec_policy = thrust::cuda::par.on(stream);
    auto new_size = thrust::remove_if(exec_policy, points.begin(), points.end(),
                                      check_is_invalid_point_functor());
    cudaStreamSynchronize(stream);

    points.resize(new_size - points.begin());
}
} // namespace gca
