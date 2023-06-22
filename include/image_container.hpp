#pragma once

#include <memory>

#include "device.hpp"
#include "type.hpp"

namespace gca
{
class image_container
{
public:
    image_container(/* args */);
    ~image_container();

private:
    std::unique_ptr<gca::device> camera;
};
} // namespace gca
