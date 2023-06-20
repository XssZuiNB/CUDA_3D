#pragma once

#include <memory>

/*
template <typename T>
std::shared_ptr<T> make_device_copy(T obj);
*/

template <typename T>
bool alloc_dev(std::shared_ptr<T>& cuda_ptr, int elements);
