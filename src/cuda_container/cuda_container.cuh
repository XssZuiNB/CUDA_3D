#pragma once

#include <cuda_runtime_api.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>

#include "camera/camera_param.hpp"
#include "util/cuda_util.cuh"

namespace gca
{
template <typename T, size_t N_CHANNEL> class cuda_frame
{
public:
    cuda_frame<T, N_CHANNEL>()
        : m_width(0)
        , m_height(0)
        , m_size(0)
        , m_frame_ptr(nullptr)
    {
    }

    cuda_frame<T, N_CHANNEL>(uint32_t width, uint32_t height)
        : m_width(width)
        , m_height(height)
        , m_size(width * height * N_CHANNEL)
    {
        auto err = cudaMalloc(&m_frame_ptr, sizeof(T) * m_size);
        check_cuda_error(err, __FILE__, __LINE__);
    }

    cuda_frame<T, N_CHANNEL>(const T *frame, uint32_t width, uint32_t height)
        : m_width(width)
        , m_height(height)
        , m_size(width * height * N_CHANNEL)
    {
        auto err = cudaMalloc(&m_frame_ptr, sizeof(T) * m_size);
        check_cuda_error(err, __FILE__, __LINE__);
        cudaMemcpy(m_frame_ptr, frame, sizeof(T) * m_size, cudaMemcpyDefault);
        // cudaDeviceSynchronize();
    }

    cuda_frame<T, N_CHANNEL>(const cuda_frame<T, N_CHANNEL> &other)
        : m_width(other.m_width)
        , m_height(other.m_height)
        , m_size(other.m_size)
    {
        auto err = cudaMalloc(&m_frame_ptr, sizeof(T) * m_size);
        check_cuda_error(err, __FILE__, __LINE__);
        cudaMemcpy(m_frame_ptr, other.m_frame_ptr, sizeof(T) * m_size, cudaMemcpyDefault);
        // cudaDeviceSynchronize();
    }

    cuda_frame<T, N_CHANNEL>(cuda_frame<T, N_CHANNEL> &&other) noexcept
        : m_width(other.m_width)
        , m_height(other.m_height)
        , m_size(other.m_size)
        , m_frame_ptr(other.m_frame_ptr)
    {
        other.m_frame_ptr = nullptr;
    }

    cuda_frame<T, N_CHANNEL> &operator=(const cuda_frame<T, N_CHANNEL> &other)
    {
        if (this != &other)
        {
            auto other_size = other.m_size;
            if (other_size != m_size)
            {
                m_size = other_size;
                if (m_frame_ptr)
                {
                    cudaFree(m_frame_ptr);
                }
                auto err = cudaMalloc(&m_frame_ptr, sizeof(T) * m_size);
                check_cuda_error(err, __FILE__, __LINE__);
            }
            m_width = other.m_width;
            m_height = other.m_height;
            cudaMemcpy(m_frame_ptr, other.m_frame_ptr, sizeof(T) * m_size, cudaMemcpyDefault);
            // cudaDeviceSynchronize();
        }

        return *this;
    }

    cuda_frame<T, N_CHANNEL> &operator=(cuda_frame<T, N_CHANNEL> &&other) noexcept
    {
        if (this != &other)
        {
            if (m_frame_ptr)
            {
                cudaFree(m_frame_ptr);
            }

            m_width = other.m_width;
            m_height = other.m_height;
            m_size = other.m_size;
            m_frame_ptr = other.m_frame_ptr;
            other.m_frame_ptr = nullptr;
        }
        return *this;
    }

    void upload(const T *frame, uint32_t width, uint32_t height)
    {

        auto new_size = width * height * N_CHANNEL;
        if (new_size != m_size)
        {
            m_size = new_size;
            if (m_frame_ptr)
            {
                cudaFree(m_frame_ptr);
            }
            auto err = cudaMalloc(&m_frame_ptr, sizeof(T) * m_size);
            check_cuda_error(err, __FILE__, __LINE__);
        }
        m_width = width;
        m_height = height;
        cudaMemcpy(m_frame_ptr, frame, sizeof(T) * m_size, cudaMemcpyDefault);
    }

    const T *data() const
    {
        return m_frame_ptr;
    }

    void clear()
    {
        if (m_frame_ptr)
        {
            cudaFree(m_frame_ptr);
            m_frame_ptr = nullptr;
        }
    }

    ~cuda_frame<T, N_CHANNEL>()
    {
        clear();
    }

private:
    uint32_t m_width;
    uint32_t m_height;
    size_t m_size;
    T *m_frame_ptr;
};
} // namespace gca
