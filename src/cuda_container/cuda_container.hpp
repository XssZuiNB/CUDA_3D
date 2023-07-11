#pragma once

#include <memory>

#include "camera/camera_device.hpp"
#include "camera/camera_param.hpp"

namespace gca
{
class cuda_color_frame_impl;
class cuda_color_frame
{
public:
    cuda_color_frame(uint32_t width, uint32_t height);
    cuda_color_frame(const uint8_t *frame, uint32_t width, uint32_t height);
    cuda_color_frame(const cuda_color_frame &other);
    cuda_color_frame(cuda_color_frame &&other) noexcept;

    cuda_color_frame &operator=(const cuda_color_frame &other);
    cuda_color_frame &operator=(cuda_color_frame &&other) noexcept;

    void upload(const uint8_t *src, uint32_t width, uint32_t height);

    const uint8_t *data() const;

    void clear();

    ~cuda_color_frame();

private:
    cuda_color_frame_impl *__m_impl;
    uint32_t m_width;
    uint32_t m_height;
};

class cuda_depth_frame_impl;
class cuda_depth_frame
{
public:
    cuda_depth_frame(uint32_t width, uint32_t height);
    cuda_depth_frame(const uint16_t *frame, uint32_t width, uint32_t height);
    cuda_depth_frame(const cuda_depth_frame &other);
    cuda_depth_frame(cuda_depth_frame &&other) noexcept;

    cuda_depth_frame &operator=(const cuda_depth_frame &other);
    cuda_depth_frame &operator=(cuda_depth_frame &&other) noexcept;

    void upload(const uint16_t *src, uint32_t width, uint32_t height);

    const uint16_t *data() const;

    void clear();

    ~cuda_depth_frame();

private:
    cuda_depth_frame_impl *__m_impl;
    uint32_t m_width;
    uint32_t m_height;
};

class cuda_camera_param
{
public:
    cuda_camera_param();

    cuda_camera_param(const device &device);
    cuda_camera_param(const device *device_ptr);

    cuda_camera_param(const gca::intrinsics *depth_intrin_ptr,
                      const gca::intrinsics *color_intrin_ptr,
                      const gca::extrinsics *depth2color_extrin_ptr,
                      const gca::extrinsics *color2depth_extrin_ptr, const uint32_t *width_ptr,
                      const uint32_t *height_ptr, const float *depth_scale_ptr);
    cuda_camera_param(const gca::intrinsics &depth_intrin, const gca::intrinsics &color_intrin,
                      const gca::extrinsics &depth2color_extrin,
                      const gca::extrinsics &color2depth_extrin, const uint32_t width,
                      const uint32_t height, const float depth_scale);

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

    void clear();

    ~cuda_camera_param();

private:
    gca::intrinsics *m_depth_intrin_ptr;
    gca::intrinsics *m_color_intrin_ptr;
    gca::extrinsics *m_depth2color_extrin_ptr;
    gca::extrinsics *m_color2depth_extrin_ptr;
    uint32_t m_width;
    uint32_t m_height;
    float m_depth_scale;
};
} // namespace gca
