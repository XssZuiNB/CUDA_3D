#include "cuda_movement_detection.cuh"

#include "geometry/cuda_nn_search.cuh"
#include "geometry/type.hpp"
#include "util/cuda_util.cuh"
#include "util/math.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

namespace gca
{
struct assign_data_functor
{
    assign_data_functor(float k1, float k2)
        : m_k1(k1)
        , m_k2(k2)
    {
    }

    float m_k1;
    float m_k2;

    __forceinline__ __device__ uint8_t operator()(float data)
    {
        return (data - m_k1) < (data - m_k2) ? 0 : 1;
    }
};

struct reduce_and_count_functor
{
    __forceinline__ __device__ thrust::tuple<float, gca::counter_t> operator()(
        const thrust::tuple<float, gca::counter_t> &a,
        const thrust::tuple<float, gca::counter_t> &b) const
    {
        return thrust::make_tuple(thrust::get<0>(a) + thrust::get<0>(b),
                                  thrust::get<1>(a) + thrust::get<1>(b));
    }
};

void assign_data_and_update(thrust::device_vector<uint8_t> &result, float &k1, float &k2,
                            const thrust::device_vector<float> &data)
{
    thrust::device_vector<uint8_t> cluster_of_pts(data.size());
    thrust::device_vector<float> data_copy(data);

    thrust::transform(data_copy.begin(), data_copy.end(), cluster_of_pts.begin(),
                      assign_data_functor(k1, k2));

    result = cluster_of_pts;

    /** compute new centroid **/
    thrust::sort_by_key(cluster_of_pts.begin(), cluster_of_pts.end(), data_copy.begin());

    thrust::device_vector<thrust::tuple<float, gca::counter_t>> new_centroid(2);
    thrust::constant_iterator<gca::counter_t> counts(1);
    auto zipped_data_begin =
        thrust::make_zip_iterator(thrust::make_tuple(data_copy.begin(), counts));

    thrust::reduce_by_key(cluster_of_pts.begin(), cluster_of_pts.end(), zipped_data_begin,
                          thrust::make_discard_iterator(), new_centroid.begin(),
                          thrust::equal_to<int>(), reduce_and_count_functor());

    auto data1 = new_centroid[0];
    //  k1 = thrust::get<0>(new_centroid[0]);
}
} // namespace gca
