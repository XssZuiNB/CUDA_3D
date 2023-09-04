#include "geometry/cuda_nn_search.cuh"
#include "geometry/type.hpp"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

namespace gca
{
__forceinline__ __device__ float color_intensity(gca::color3 color)
{
    return 0.2126 * color.r + 0.7152 * color.g + 0.0722 * color.b;
}

__forceinline__ __device__ float color_intensity(float r, float g, float b)
{
    return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

struct check_if_moving_point_functor
{
    check_if_moving_point_functor(
        const thrust::device_vector<gca::point_t> &pts_last_frame,
        const thrust::device_vector<gca::index_t> &all_neighbors_in_last_frame,
        const thrust::device_vector<thrust::pair<gca::index_t, gca::counter_t>>
            &pair_neighbors_begin_idx_and_count,
        const float color_constraint)
        : m_pts_last_frame_ptr(thrust::raw_pointer_cast(pts_last_frame.data()))
        , m_found_neighbors_ptr(thrust::raw_pointer_cast(all_neighbors_in_last_frame.data()))
        , m_pair_neighbors_begin_idx_and_count_ptr(
              thrust::raw_pointer_cast(pair_neighbors_begin_idx_and_count.data()))
        , m_color_constraint(color_constraint)
    {
    }

    const gca::point_t *m_pts_last_frame_ptr;
    const gca::index_t *m_found_neighbors_ptr;
    const thrust::pair<gca::index_t, gca::counter_t> *m_pair_neighbors_begin_idx_and_count_ptr;
    const float m_color_constraint;

    __forceinline__ __device__ gca::point_t operator()(const gca::point_t &point,
                                                       gca::index_t point_idx)
    {
        auto neighbors_begin_idx =
            __ldg(&(m_pair_neighbors_begin_idx_and_count_ptr[point_idx].first));
        auto neighbors_num = __ldg(&(m_pair_neighbors_begin_idx_and_count_ptr[point_idx].second));

        if (!neighbors_num)
            return point;

        for (gca::index_t i = 0; i < neighbors_num; i++)
        {
            auto neighbor_idx = __ldg(&m_found_neighbors_ptr[neighbors_begin_idx + i]);
            auto neighbor_point_x = __ldg(&(m_pts_last_frame_ptr[neighbor_idx].coordinates.x));
            auto neighbor_point_y = __ldg(&(m_pts_last_frame_ptr[neighbor_idx].coordinates.y));
            auto neighbor_point_z = __ldg(&(m_pts_last_frame_ptr[neighbor_idx].coordinates.z));
            auto neighbor_point_color_r = __ldg(&(m_pts_last_frame_ptr[neighbor_idx].color.r));
            auto neighbor_point_color_g = __ldg(&(m_pts_last_frame_ptr[neighbor_idx].color.g));
            auto neighbor_point_color_b = __ldg(&(m_pts_last_frame_ptr[neighbor_idx].color.b));

            auto coordinates_diff_x =
                abs((neighbor_point_x - point.coordinates.x) / point.coordinates.x);
            auto coordinates_diff_y =
                abs((neighbor_point_y - point.coordinates.y) / point.coordinates.y);
            auto coordinates_diff_z =
                abs((neighbor_point_z - point.coordinates.z) / point.coordinates.z);
            auto color_diff = abs(color_intensity(point.color) -
                                  color_intensity(neighbor_point_color_r, neighbor_point_color_g,
                                                  neighbor_point_color_b));
            if (coordinates_diff_x < 0.015 * abs(point.coordinates.z) &&
                coordinates_diff_y < 0.015 * abs(point.coordinates.z) &&
                coordinates_diff_z < 0.015 * abs(point.coordinates.z) &&
                color_diff < m_color_constraint)
            {
                return point;
            }
        }
        auto temp_p = point;
        temp_p.color.r = 0;
        temp_p.color.g = 1;
        temp_p.color.b = 0;
        temp_p.property = gca::point_property::active;
        return temp_p;
    }
};

/* This moving objects detection algorithm assume that camera is not moving */
/* This version is a easy implement without normals of points considered */
::cudaError_t cuda_movement_detection(thrust::device_vector<gca::point_t> &result,
                                      const thrust::device_vector<gca::point_t> &pts_this_frame,
                                      const thrust::device_vector<gca::point_t> &pts_last_frame,
                                      const float3 min_bound, const float3 max_bound,
                                      const float geometry_constraint, const float color_constraint)
{
    thrust::device_vector<gca::index_t> all_neighbors_in_last_frame;
    thrust::device_vector<thrust::pair<gca::index_t, gca::counter_t>>
        pair_neighbors_begin_idx_and_count(pts_this_frame.size());

    auto err = cuda_search_radius_neighbors(
        all_neighbors_in_last_frame, pair_neighbors_begin_idx_and_count, pts_this_frame,
        pts_last_frame, min_bound, max_bound, geometry_constraint);
    if (err != ::cudaSuccess)
        return err;

    // thrust::device_vector<bool> if_moving(pts_this_frame.size());
    result.resize(pts_this_frame.size());
    thrust::transform(pts_this_frame.begin(), pts_this_frame.end(),
                      thrust::make_counting_iterator<gca::index_t>(0), result.begin(),
                      check_if_moving_point_functor(pts_last_frame, all_neighbors_in_last_frame,
                                                    pair_neighbors_begin_idx_and_count,
                                                    color_constraint));
    err = cudaGetLastError();
    if (err != ::cudaSuccess)
        return err;

    return cudaSuccess;
}
} // namespace gca