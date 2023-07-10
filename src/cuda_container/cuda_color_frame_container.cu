#include <cuda_runtime_api.h>

#include "cuda_container.cuh"
#include "cuda_container.hpp"
#include "util/cuda_util.cuh"

namespace gca
{
class cuda_color_frame_impl
{
private:
    cuda_frame<uint8_t, 3> m_color_frame;

public:
    cuda_color_frame_impl(uint32_t width, uint32_t height)
        : m_color_frame(width, height)
    {
    }

    cuda_color_frame_impl(const uint8_t *frame, uint32_t width, uint32_t height)
        : m_color_frame(frame, width, height)
    {
    }

    cuda_color_frame_impl(const cuda_color_frame_impl &other)
        : m_color_frame(other.m_color_frame)
    {
    }

    const uint8_t *data() const
    {
        return m_color_frame.data();
    }

    void upload(const uint8_t *src, uint32_t width, uint32_t height)
    {
        m_color_frame.upload(src, width, height);
    }

    void clear()
    {
        m_color_frame.clear();
    }

    ~cuda_color_frame_impl() = default;
};

cuda_color_frame::cuda_color_frame(uint32_t width, uint32_t height)
    : __m_impl(new cuda_color_frame_impl(width, height))
    , m_width(width)
    , m_height(height)
{
}

cuda_color_frame::cuda_color_frame(const uint8_t *frame, uint32_t width, uint32_t height)
    : __m_impl(new cuda_color_frame_impl(frame, width, height))
    , m_width(width)
    , m_height(height)
{
}

cuda_color_frame::cuda_color_frame(const cuda_color_frame &other)
    : __m_impl(new cuda_color_frame_impl(*other.__m_impl))
    , m_width(other.m_width)
    , m_height(other.m_height)
{
}

cuda_color_frame::cuda_color_frame(cuda_color_frame &&other) noexcept
    : __m_impl(other.__m_impl)
{
    other.__m_impl = nullptr;
}

cuda_color_frame &cuda_color_frame::operator=(const cuda_color_frame &other)
{
    if (this != &other)
    {
        upload(other.data(), other.m_width, other.m_height);
    }
    return *this;
}

cuda_color_frame &cuda_color_frame::operator=(cuda_color_frame &&other) noexcept
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

const uint8_t *cuda_color_frame::data() const
{
    return __m_impl->data();
}

void cuda_color_frame::upload(const uint8_t *src, uint32_t width, uint32_t height)
{
    __m_impl->upload(src, width, height);
}

void cuda_color_frame::clear()
{
    __m_impl->clear();
}

cuda_color_frame::~cuda_color_frame()
{
    delete __m_impl;
    __m_impl = nullptr;
}
} // namespace gca
