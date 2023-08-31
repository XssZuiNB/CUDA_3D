#pragma once

#include "camera/camera_device.hpp"
#include "camera/camera_param.hpp"
#include "cuda_container/cuda_frame.cuh"

#include <thrust/device_vector.h>

namespace gca
{
class cuda_color_frame
{
public:
    cuda_color_frame(uint32_t width, uint32_t height);
    cuda_color_frame(const uint8_t *frame, uint32_t width, uint32_t height);
    cuda_color_frame(const cuda_color_frame &other);
    cuda_color_frame(cuda_color_frame &&other) noexcept;

    cuda_color_frame &operator=(const cuda_color_frame &other);
    cuda_color_frame &operator=(cuda_color_frame &&other) noexcept;

    uint32_t get_color_frame_width() const;
    uint32_t get_color_frame_height() const;

    void upload(const uint8_t *src, uint32_t width, uint32_t height);

    const thrust::device_vector<uint8_t> &get_color_frame_vec() const;

    void clear();

    ~cuda_color_frame();

private:
    using cuda_color_frame_impl = gca::cuda_frame<uint8_t, 3>;
    cuda_color_frame_impl *__m_impl;
};

class cuda_depth_frame
{
public:
    cuda_depth_frame(uint32_t width, uint32_t height);
    cuda_depth_frame(const uint16_t *frame, uint32_t width, uint32_t height);
    cuda_depth_frame(const cuda_depth_frame &other);
    cuda_depth_frame(cuda_depth_frame &&other) noexcept;

    cuda_depth_frame &operator=(const cuda_depth_frame &other);
    cuda_depth_frame &operator=(cuda_depth_frame &&other) noexcept;

    uint32_t get_depth_frame_width() const;
    uint32_t get_depth_frame_height() const;

    void upload(const uint16_t *src, uint32_t width, uint32_t height);

    const thrust::device_vector<uint16_t> &get_depth_frame_vec() const;

    void clear();

    ~cuda_depth_frame();

private:
    using cuda_depth_frame_impl = gca::cuda_frame<uint16_t, 1>;
    cuda_depth_frame_impl *__m_impl;
};

class cuda_camera_param
{
public:
    cuda_camera_param();

    cuda_camera_param(const device &device);
    cuda_camera_param(const device *device_ptr);

    cuda_camera_param(const gca::intrinsics &depth_intrin, const gca::intrinsics &color_intrin,
                      const gca::extrinsics &depth2color_extrin,
                      const gca::extrinsics &color2depth_extrin, const uint32_t width,
                      const uint32_t height, const float depth_scale,
                      const gca::frame_format depth_format = gca::Z16,
                      const gca::frame_format color_format = gca::BGR8);

    cuda_camera_param(const cuda_camera_param &other);
    cuda_camera_param(cuda_camera_param &&other) noexcept;

    cuda_camera_param &operator=(const cuda_camera_param &other);
    cuda_camera_param &operator=(cuda_camera_param &&other) noexcept;

    bool set(const gca::intrinsics &intrin, param_type type);
    bool set(const gca::extrinsics &extrin, param_type type);
    bool set(uint32_t length, param_type type);
    bool set(float depth_scale);

    const gca::intrinsics *get_depth_intrinsics_ptr() const;
    const gca::intrinsics *get_color_intrinsics_ptr() const;
    const gca::extrinsics *get_depth2color_extrinsics_ptr() const;
    const gca::extrinsics *get_color2depth_extrinsics_ptr() const;
    uint32_t get_width() const;
    uint32_t get_height() const;
    float get_depth_scale() const;
    gca::frame_format get_depth_frame_format() const;
    gca::frame_format get_color_frame_format() const;

    ~cuda_camera_param();

private:
    gca::intrinsics *m_depth_intrin_ptr;
    gca::intrinsics *m_color_intrin_ptr;
    gca::extrinsics *m_depth2color_extrin_ptr;
    gca::extrinsics *m_color2depth_extrin_ptr;
    uint32_t m_width;
    uint32_t m_height;
    float m_depth_scale;
    gca::frame_format m_depth_frame_format;
    gca::frame_format m_color_frame_format;
};
} // namespace gca
