#pragma once

#include "camera/camera_param.hpp"
#include "util/cuda_util.cuh"

#include <cuda_runtime_api.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>

namespace gca
{
template <typename T, size_t N_CHANNEL> class cuda_frame
{
public:
    cuda_frame<T, N_CHANNEL>()
        : m_width(0)
        , m_height(0)
        , m_data_vec(0)
    {
    }

    cuda_frame<T, N_CHANNEL>(uint32_t width, uint32_t height)
        : m_width(width)
        , m_height(height)
        , m_data_vec(width * height * N_CHANNEL)
    {
    }

    cuda_frame<T, N_CHANNEL>(const T *frame, uint32_t width, uint32_t height)
        : m_width(width)
        , m_height(height)
        , m_data_vec(width * height * N_CHANNEL)
    {
        if (frame)
        {
            thrust::copy(frame, frame + m_data_vec.size(), m_data_vec.begin());
            auto err = cudaGetLastError();
            check_cuda_error(err, __FILE__, __LINE__);
        }
    }

    cuda_frame<T, N_CHANNEL>(const cuda_frame<T, N_CHANNEL> &other)
        : m_width(other.m_width)
        , m_height(other.m_height)
        , m_data_vec(other.m_data_vec.size())
    {
        thrust::copy(other.m_data_vec.begin(), other.m_data_vec.end(), m_data_vec.begin());
        auto err = cudaGetLastError();
        check_cuda_error(err, __FILE__, __LINE__);
    }

    cuda_frame<T, N_CHANNEL>(cuda_frame<T, N_CHANNEL> &&other) noexcept
        : m_width(other.m_width)
        , m_height(other.m_height)
        , m_data_vec(0)
    {
        m_data_vec.swap(other.m_data_vec);
    }

    cuda_frame<T, N_CHANNEL> &operator=(const cuda_frame<T, N_CHANNEL> &other)
    {
        if (this != &other)
        {
            auto other_size = other.m_data_vec.size();
            if (other_size != m_data_vec.size())
                m_data_vec.resize(other_size);

            thrust::copy(other.m_data_vec.begin(), other.m_data_vec.end(), m_data_vec.begin());
            auto err = cudaGetLastError();
            check_cuda_error(err, __FILE__, __LINE__);

            m_width = other.m_width;
            m_height = other.m_height;
        }

        return *this;
    }

    cuda_frame<T, N_CHANNEL> &operator=(cuda_frame<T, N_CHANNEL> &&other) noexcept
    {
        if (this != &other)
        {
            m_width = other.m_width;
            m_height = other.m_height;
            m_data_vec.swap(other.m_data_vec);
        }

        return *this;
    }

    uint32_t get_frame_width() const
    {
        return m_width;
    }

    uint32_t get_frame_height() const
    {
        return m_height;
    }

    void upload(const T *frame, uint32_t width, uint32_t height)
    {
        if (!frame)
        {
            return;
        }

        auto new_size = width * height * N_CHANNEL;
        if (new_size != m_data_vec.size())
        {
            m_data_vec.resize(new_size);
        }

        thrust::copy(frame, frame + new_size, m_data_vec.begin());
        auto err = cudaGetLastError();
        check_cuda_error(err, __FILE__, __LINE__);

        m_width = width;
        m_height = height;
    }

    const thrust::device_vector<T> &get_frame_vec() const
    {
        return m_data_vec;
    }

    void clear()
    {
        m_data_vec.clear();
    }

    ~cuda_frame<T, N_CHANNEL>()
    {
    }

private:
    uint32_t m_width;
    uint32_t m_height;
    thrust::device_vector<T> m_data_vec;
};
} // namespace gca
