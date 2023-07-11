#include <cuda_runtime_api.h>

#include "cuda_container.cuh"
#include "cuda_container.hpp"
#include "util/cuda_util.cuh"

namespace gca
{
class cuda_depth_frame_impl
{
private:
    cuda_frame<uint16_t, 1> m_depth_frame;

public:
    cuda_depth_frame_impl(uint32_t width, uint32_t height)
        : m_depth_frame(width, height)
    {
    }

    cuda_depth_frame_impl(const uint16_t *frame, uint32_t width, uint32_t height)
        : m_depth_frame(frame, width, height)
    {
    }

    cuda_depth_frame_impl(const cuda_depth_frame_impl &other)
        : m_depth_frame(other.m_depth_frame)
    {
    }

    const uint16_t *data() const
    {
        return m_depth_frame.data();
    }

    void upload(const uint16_t *src, uint32_t width, uint32_t height)
    {
        m_depth_frame.upload(src, width, height);
    }

    void clear()
    {
        m_depth_frame.clear();
    }

    ~cuda_depth_frame_impl() = default;
};

cuda_depth_frame::cuda_depth_frame(uint32_t width, uint32_t height)
    : __m_impl(new cuda_depth_frame_impl(width, height))
    , m_width(width)
    , m_height(height)
{
}

cuda_depth_frame::cuda_depth_frame(const uint16_t *frame, uint32_t width, uint32_t height)
    : __m_impl(new cuda_depth_frame_impl(frame, width, height))
    , m_width(width)
    , m_height(height)
{
}

cuda_depth_frame::cuda_depth_frame(const cuda_depth_frame &other)
    : __m_impl(new cuda_depth_frame_impl(*other.__m_impl))
    , m_width(other.m_width)
    , m_height(other.m_height)
{
}

cuda_depth_frame::cuda_depth_frame(cuda_depth_frame &&other) noexcept
    : __m_impl(other.__m_impl)
{
    other.__m_impl = nullptr;
}

cuda_depth_frame &cuda_depth_frame::operator=(const cuda_depth_frame &other)
{
    if (this != &other)
    {
        upload(other.data(), other.m_width, other.m_height);
    }
    return *this;
}

cuda_depth_frame &cuda_depth_frame::operator=(cuda_depth_frame &&other) noexcept
{
    if (this != &other)
    {
        if (__m_impl)
        {
            delete __m_impl;
        }

        __m_impl = other.__m_impl;
        other.__m_impl = nullptr;
    }
    return *this;
}

const uint16_t *cuda_depth_frame::data() const
{
    return __m_impl->data();
}

void cuda_depth_frame::upload(const uint16_t *src, uint32_t width, uint32_t height)
{
    __m_impl->upload(src, width, height);
}

void cuda_depth_frame::clear()
{
    __m_impl->clear();
}

cuda_depth_frame::~cuda_depth_frame()
{
    delete __m_impl;
    __m_impl = nullptr;
}
} // namespace gca