#pragma once

#include <cuda_runtime_api.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>

#include "camera/camera_param.hpp"
#include "util/cuda_util.cuh"

namespace gca
{
template <typename T> class cuda_object
{
public:
    cuda_object<T>()
        : m_cuda_ptr(nullptr)
    {
    }

    cuda_object<T>(const T &obj)
    {
        auto err = cudaMalloc(&m_cuda_ptr, sizeof(T));
        check_cuda_error(err, __FILE__, __LINE__);
        *m_cuda_ptr = obj;
    }

    cuda_object<T>(const T &&obj)
    {
        auto err = cudaMalloc(&m_cuda_ptr, sizeof(T));
        check_cuda_error(err, __FILE__, __LINE__);
        *m_cuda_ptr = obj;
    }

    cuda_object<T>(const cuda_object<T> &other)
    {
        auto err = cudaMalloc(&m_cuda_ptr, sizeof(T));
        check_cuda_error(err, __FILE__, __LINE__);
        cudaMemcpy(m_cuda_ptr, other.m_cuda_ptr, sizeof(T), cudaMemcpyDefault);
    }

    cuda_object<T>(cuda_object<T> &&other) noexcept
        : m_cuda_ptr(other.m_cuda_ptr)
    {
        other.m_cuda_ptr = nullptr;
    }

    cuda_object<T> &operator=(const cuda_object<T> &other)
    {
        if (this != &other)
        {
            upload(*other.m_cuda_ptr);
        }
    }

    cuda_object<T> &operator=(cuda_object<T> &&other) noexcept
    {
        if (this != &other)
        {
            if (m_cuda_ptr)
            {
                cudaFree(m_cuda_ptr);
            }
            m_cuda_ptr = other.m_cuda_ptr;
            other.m_cuda_ptr = nullptr;
        }
    }

    cuda_object<T> &operator=(const T &obj)
    {
        upload(obj);
    }

    void upload(const T &obj)
    {
        if (!m_cuda_ptr)
        {
            auto err = cudaMalloc(&m_cuda_ptr, sizeof(T));
            check_cuda_error(err, __FILE__, __LINE__);
        }

        *m_cuda_ptr = obj;
    }

    void download(T *dst)
    {
        *dst = *m_cuda_ptr;
    }

    const T *data() const
    {
        if (m_cuda_ptr)
        {
            return m_cuda_ptr;
        }

        return nullptr;
    }

    void clear()
    {
        if (m_cuda_ptr)
        {
            cudaFree(m_cuda_ptr);
            m_cuda_ptr = nullptr;
        }
    }

    ~cuda_object<T>()
    {
        clear();
    }

private:
    T *m_cuda_ptr;
};

template <typename T, size_t N_CHANNEL> class cuda_frame
{
public:
    cuda_frame<T, N_CHANNEL>()
        : m_width(0)
        , m_height(0)
        , m_size(0)
        , m_frame(nullptr)
    {
    }

    cuda_frame<T, N_CHANNEL>(uint32_t width, uint32_t height)
        : m_width(width)
        , m_height(height)
        , m_size(width * height * N_CHANNEL)
    {
        auto err = cudaMalloc(&m_frame, sizeof(T) * m_size);
        check_cuda_error(err, __FILE__, __LINE__);
    }

    cuda_frame<T, N_CHANNEL>(const T *frame, uint32_t width, uint32_t height)
        : m_width(width)
        , m_height(height)
        , m_size(width * height * N_CHANNEL)
    {
        auto err = cudaMalloc(&m_frame, sizeof(T) * m_size);
        check_cuda_error(err, __FILE__, __LINE__);
        cudaMemcpy(m_frame, frame, m_size, cudaMemcpyDefault);
    }

    cuda_frame<T, N_CHANNEL>(const cuda_frame<T, N_CHANNEL> &other)
        : m_width(other.m_width)
        , m_height(other.m_height)
        , m_size(other.m_size)
    {
        auto err = cudaMalloc(&m_frame, sizeof(T) * m_size);
        check_cuda_error(err, __FILE__, __LINE__);
        cudaMemcpy(m_frame, other.m_frame, m_size, cudaMemcpyDefault);
    }

    cuda_frame<T, N_CHANNEL>(cuda_frame<T, N_CHANNEL> &&other) noexcept
        : m_width(other.m_width)
        , m_height(other.m_height)
        , m_size(other.m_size)
        , m_frame(other.m_frame)
    {
        other.m_frame = nullptr;
    }

    cuda_frame<T, N_CHANNEL> &operator=(const cuda_frame<T, N_CHANNEL> &other)
    {
        if (this != &other)
        {
            auto other_size = other.m_size;
            if (other_size != m_size)
            {
                m_size = other.m_size;
                if (m_frame)
                {
                    cudaFree(m_frame);
                }
                auto err = cudaMalloc(&m_frame, sizeof(T) * m_size);
                check_cuda_error(err, __FILE__, __LINE__);
            }
            m_width = other.m_width;
            m_height = other.m_height;
            cudaMemcpy(m_frame, other.m_frame, m_size, cudaMemcpyDefault);
        }

        return *this;
    }

    cuda_frame<T, N_CHANNEL> &operator=(cuda_frame<T, N_CHANNEL> &&other) noexcept
    {
        if (this != &other)
        {
            m_width = other.m_width;
            m_height = other.m_height;
            m_size = other.m_size;
            m_frame = other.m_frame;
            other.m_frame = nullptr;
        }
        return *this;
    }

    void upload(const T *frame, uint32_t width, uint32_t height)
    {
        auto new_size = width * height * N_CHANNEL;
        if (new_size != m_size)
        {
            m_size = new_size;
            if (m_frame)
            {
                cudaFree(m_frame);
            }
            auto err = cudaMalloc(&m_frame, sizeof(T) * m_size);
            check_cuda_error(err, __FILE__, __LINE__);
        }
        m_width = width;
        m_height = height;
        cudaMemcpy(m_frame, frame, m_size, cudaMemcpyDefault);
    }

    const T *data() const
    {
        return m_frame;
    }

    void clear()
    {
        if (m_frame)
        {
            cudaFree(m_frame);
            m_frame = nullptr;
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
    T *m_frame;
};
} // namespace gca
