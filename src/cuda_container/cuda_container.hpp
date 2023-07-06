#pragma once

#include <memory>

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
} // namespace gca
