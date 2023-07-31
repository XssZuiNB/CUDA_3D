#pragma once

#include <vector_types.h>

namespace gca
{
typedef int32_t index_t;
typedef int32_t counter_t;

enum point_property
{
    invalid = 0,
    active = 1,
    inactive = 2
};

struct color3
{
    float r;
    float g;
    float b;
};

struct __align__(16) point_t
{
    float3 coordinates;
    color3 color;
    point_property property;
    friend std::ostream &operator<<(std::ostream &os, const point_t &point);
};

inline std::ostream &operator<<(std::ostream &os, const point_t &point)
{
    os << "Coordinates: x = " << point.coordinates.x << "\n"
       << "             y = " << point.coordinates.y << "\n"
       << "             z = " << point.coordinates.z << "\n"
       << "R: " << point.color.r << " G: " << point.color.g << " B: " << point.color.b << "\n"
       << "Property: " << (point.property == gca::point_property::active ? "Active" : "Inactive")
       << "\n";
    return os;
}

struct __align__(16) grid_cell_t
{
    index_t start_index;
    counter_t points_number = 0;
    friend std::ostream &operator<<(std::ostream &os, const grid_cell_t &cell);
};

inline std::ostream &operator<<(std::ostream &os, const grid_cell_t &cell)
{
    os << "Start Index: " << cell.start_index << ", Points Number: " << cell.points_number << "\n";
    return os;
}
} // namespace gca
