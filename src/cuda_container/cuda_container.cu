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

namespace gca
{
cuda_camera_param::cuda_camera_param()
    : m_depth_intrin_ptr(nullptr)
    , m_color_intrin_ptr(nullptr)
    , m_depth2color_extrin_ptr(nullptr)
    , m_color2depth_extrin_ptr(nullptr)
    , m_width(0)
    , m_height(0)
    , m_depth_scale(0.0)
{
}

cuda_camera_param::cuda_camera_param(const device *device)
    : cuda_camera_param(*device)
{
}

cuda_camera_param::cuda_camera_param(const device &device)
    : cuda_camera_param(device.get_depth_intrinsics(), device.get_color_intrinsics(),
                        device.get_depth_to_color_extrinsics(),
                        device.get_color_to_depth_extrinsics(), device.get_width(),
                        device.get_height(), device.get_depth_scale())
{
}

cuda_camera_param::cuda_camera_param(const gca::intrinsics *depth_intrin_ptr,
                                     const gca::intrinsics *color_intrin_ptr,
                                     const gca::extrinsics *depth2color_extrin_ptr,
                                     const gca::extrinsics *color2depth_extrin_ptr,
                                     const uint32_t *width_ptr, const uint32_t *height_ptr,
                                     const float *depth_scale_ptr)
    : cuda_camera_param(*depth_intrin_ptr, *color_intrin_ptr, *depth2color_extrin_ptr,
                        *color2depth_extrin_ptr, *width_ptr, *height_ptr, *depth_scale_ptr)
{
}

cuda_camera_param::cuda_camera_param(const gca::intrinsics &depth_intrin,
                                     const gca::intrinsics &color_intrin,
                                     const gca::extrinsics &depth2color_extrin,
                                     const gca::extrinsics &color2depth_extrin,
                                     const uint32_t width, const uint32_t height,
                                     const float depth_scale)
    : m_width(width)
    , m_height(height)
    , m_depth_scale(depth_scale)
{
    auto err = cudaMalloc(&m_depth_intrin_ptr, sizeof(gca::intrinsics));
    check_cuda_error(err, __FILE__, __LINE__);
    err = cudaMalloc(&m_color_intrin_ptr, sizeof(gca::intrinsics));
    check_cuda_error(err, __FILE__, __LINE__);
    err = cudaMalloc(&m_depth2color_extrin_ptr, sizeof(gca::extrinsics));
    check_cuda_error(err, __FILE__, __LINE__);
    err = cudaMalloc(&m_color2depth_extrin_ptr, sizeof(gca::extrinsics));
    check_cuda_error(err, __FILE__, __LINE__);

    cudaMemcpy(m_depth_intrin_ptr, &depth_intrin, sizeof(gca::intrinsics), cudaMemcpyDefault);
    cudaMemcpy(m_color_intrin_ptr, &color_intrin, sizeof(gca::intrinsics), cudaMemcpyDefault);
    cudaMemcpy(m_depth2color_extrin_ptr, &depth2color_extrin, sizeof(gca::extrinsics),
               cudaMemcpyDefault);
    cudaMemcpy(m_color2depth_extrin_ptr, &color2depth_extrin, sizeof(gca::extrinsics),
               cudaMemcpyDefault);
    cudaDeviceSynchronize();
}

cuda_camera_param::cuda_camera_param(const cuda_camera_param &other)
    : m_width(other.m_width)
    , m_height(other.m_height)
    , m_depth_scale(other.m_depth_scale)
{
    auto err = cudaMalloc(&m_depth_intrin_ptr, sizeof(gca::intrinsics));
    check_cuda_error(err, __FILE__, __LINE__);
    err = cudaMalloc(&m_color_intrin_ptr, sizeof(gca::intrinsics));
    check_cuda_error(err, __FILE__, __LINE__);
    err = cudaMalloc(&m_depth2color_extrin_ptr, sizeof(gca::extrinsics));
    check_cuda_error(err, __FILE__, __LINE__);
    err = cudaMalloc(&m_color2depth_extrin_ptr, sizeof(gca::extrinsics));
    check_cuda_error(err, __FILE__, __LINE__);

    cudaMemcpy(m_depth_intrin_ptr, other.m_depth_intrin_ptr, sizeof(gca::intrinsics),
               cudaMemcpyDefault);
    cudaMemcpy(m_color_intrin_ptr, other.m_color_intrin_ptr, sizeof(gca::intrinsics),
               cudaMemcpyDefault);
    cudaMemcpy(m_depth2color_extrin_ptr, other.m_depth2color_extrin_ptr, sizeof(gca::extrinsics),
               cudaMemcpyDefault);
    cudaMemcpy(m_color2depth_extrin_ptr, other.m_color2depth_extrin_ptr, sizeof(gca::extrinsics),
               cudaMemcpyDefault);
    cudaDeviceSynchronize();
}
cuda_camera_param::cuda_camera_param(cuda_camera_param &&other) noexcept
    : m_depth_intrin_ptr(other.m_depth_intrin_ptr)
    , m_color_intrin_ptr(other.m_color_intrin_ptr)
    , m_depth2color_extrin_ptr(other.m_depth2color_extrin_ptr)
    , m_color2depth_extrin_ptr(other.m_color2depth_extrin_ptr)
    , m_width(other.m_width)
    , m_height(other.m_height)
    , m_depth_scale(other.m_depth_scale)
{
    other.m_depth_intrin_ptr = nullptr;
    other.m_color_intrin_ptr = nullptr;
    other.m_depth2color_extrin_ptr = nullptr;
    other.m_color2depth_extrin_ptr = nullptr;
}

cuda_camera_param &cuda_camera_param::operator=(const cuda_camera_param &other)
{
    if (!m_depth_intrin_ptr)
    {
        auto err = cudaMalloc(&m_depth_intrin_ptr, sizeof(gca::intrinsics));
        check_cuda_error(err, __FILE__, __LINE__);
    }
    if (!m_color_intrin_ptr)
    {
        auto err = cudaMalloc(&m_color_intrin_ptr, sizeof(gca::intrinsics));
        check_cuda_error(err, __FILE__, __LINE__);
    }
    if (!m_depth2color_extrin_ptr)
    {
        auto err = cudaMalloc(&m_depth2color_extrin_ptr, sizeof(gca::extrinsics));
        check_cuda_error(err, __FILE__, __LINE__);
    }
    if (!m_color2depth_extrin_ptr)
    {
        auto err = cudaMalloc(&m_color2depth_extrin_ptr, sizeof(gca::extrinsics));
        check_cuda_error(err, __FILE__, __LINE__);
    }

    cudaMemcpy(m_depth_intrin_ptr, other.m_depth_intrin_ptr, sizeof(gca::intrinsics),
               cudaMemcpyDefault);
    cudaMemcpy(m_color_intrin_ptr, other.m_color_intrin_ptr, sizeof(gca::intrinsics),
               cudaMemcpyDefault);
    cudaMemcpy(m_depth2color_extrin_ptr, other.m_depth2color_extrin_ptr, sizeof(gca::extrinsics),
               cudaMemcpyDefault);
    cudaMemcpy(m_color2depth_extrin_ptr, other.m_color2depth_extrin_ptr, sizeof(gca::extrinsics),
               cudaMemcpyDefault);
    cudaDeviceSynchronize();

    return *this;
}
cuda_camera_param &cuda_camera_param::operator=(cuda_camera_param &&other) noexcept
{
    m_depth_intrin_ptr = other.m_depth_intrin_ptr;
    m_color_intrin_ptr = other.m_color_intrin_ptr;
    m_depth2color_extrin_ptr = other.m_depth2color_extrin_ptr;
    m_color2depth_extrin_ptr = other.m_color2depth_extrin_ptr;
    m_width = other.m_width;
    m_height = other.m_height;
    m_depth_scale = other.m_depth_scale;

    other.m_depth_intrin_ptr = nullptr;
    other.m_color_intrin_ptr = nullptr;
    other.m_depth2color_extrin_ptr = nullptr;
    other.m_color2depth_extrin_ptr = nullptr;

    return *this;
}

bool cuda_camera_param::set(const gca::intrinsics &intrin, gca::param_type type)
{
    switch (type)
    {
    case depth_intrinsics:
        if (!m_depth_intrin_ptr)
        {
            auto err = cudaMalloc(&m_depth_intrin_ptr, sizeof(gca::intrinsics));
            check_cuda_error(err, __FILE__, __LINE__);
        }
        cudaMemcpy(m_depth_intrin_ptr, &intrin, sizeof(gca::intrinsics), cudaMemcpyDefault);
        return true;

    case color_intrinsics:
        if (!m_color_intrin_ptr)
        {
            auto err = cudaMalloc(&m_color_intrin_ptr, sizeof(gca::intrinsics));
            check_cuda_error(err, __FILE__, __LINE__);
        }
        cudaMemcpy(m_color_intrin_ptr, &intrin, sizeof(gca::intrinsics), cudaMemcpyDefault);
        return true;

    default:
        std::cout << "Wrong camera parameter type!" << std::endl;
        return false;
    }
}
bool cuda_camera_param::set(const gca::extrinsics &extrin, gca::param_type type)
{
    switch (type)
    {
    case depth2color_extrinsics:
        if (!m_depth2color_extrin_ptr)
        {
            auto err = cudaMalloc(&m_depth2color_extrin_ptr, sizeof(gca::extrinsics));
            check_cuda_error(err, __FILE__, __LINE__);
        }
        cudaMemcpy(m_depth2color_extrin_ptr, &extrin, sizeof(gca::extrinsics), cudaMemcpyDefault);
        return true;

    case color2depth_extrinsics:
        if (!m_color2depth_extrin_ptr)
        {
            auto err = cudaMalloc(&m_color2depth_extrin_ptr, sizeof(gca::extrinsics));
            check_cuda_error(err, __FILE__, __LINE__);
        }
        cudaMemcpy(m_color2depth_extrin_ptr, &extrin, sizeof(gca::extrinsics), cudaMemcpyDefault);
        return true;

    default:
        std::cout << "Wrong camera parameter type!" << std::endl;
        return false;
    }
}
bool cuda_camera_param::set(uint32_t length, param_type type)
{
    switch (type)
    {
    case width:
        m_width = length;
        return true;

    case height:
        m_height = length;
        return true;

    default:
        std::cout << "Wrong camera parameter type!" << std::endl;
        return false;
    }
}
bool cuda_camera_param::set(float depth_scale)
{
    m_depth_scale = depth_scale;
    return true;
}

const gca::intrinsics *cuda_camera_param::get_depth_intrinsics_ptr() const
{
    return m_depth_intrin_ptr;
}
const gca::intrinsics *cuda_camera_param::get_color_intrinsics_ptr() const
{
    return m_color_intrin_ptr;
}
const gca::extrinsics *cuda_camera_param::get_depth2color_extrinsics_ptr() const
{
    return m_depth2color_extrin_ptr;
}
const gca::extrinsics *cuda_camera_param::get_color2depth_extrinsics_ptr() const
{
    return m_color2depth_extrin_ptr;
}
uint32_t cuda_camera_param::get_width() const
{
    return m_width;
}
uint32_t cuda_camera_param::get_height() const
{
    return m_height;
}
float cuda_camera_param::get_depth_scale() const
{
    return m_depth_scale;
}

void cuda_camera_param::clear()
{
    if (m_depth_intrin_ptr)
    {
        cudaFree(m_depth_intrin_ptr);
    }
    m_depth_intrin_ptr = nullptr;

    if (m_color_intrin_ptr)
    {
        cudaFree(m_color_intrin_ptr);
    }
    m_color_intrin_ptr = nullptr;

    if (m_depth2color_extrin_ptr)
    {
        cudaFree(m_depth2color_extrin_ptr);
    }
    m_depth2color_extrin_ptr = nullptr;

    if (m_color2depth_extrin_ptr)
    {
        cudaFree(m_color2depth_extrin_ptr);
    }
    m_color2depth_extrin_ptr = nullptr;
}

cuda_camera_param::~cuda_camera_param()
{
    clear();
}
} // namespace gca
