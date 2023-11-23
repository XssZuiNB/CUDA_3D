#pragma once

#include "util/math.cuh"

// If a ldlt on host needed. USE EIGEN PLS!!!
class ldlt_3x3
{
public:
    __forceinline__ __device__ __host__ ldlt_3x3() = delete;

    __forceinline__ __device__ ldlt_3x3(const mat3x3 &A)
    {
        L.set_identity();
        D.set_zero();

        D(0, 0) = A(0, 0);
        L(1, 0) = A(1, 0) / D(0, 0);
        L(2, 0) = A(2, 0) / D(0, 0);
        D(1, 1) = A(1, 1) - L(1, 0) * L(1, 0) * D(0, 0);
        L(2, 1) = (A(2, 1) - L(1, 0) * L(2, 0) * D(0, 0)) / D(1, 1);
        D(2, 2) = A(2, 2) - L(2, 0) * L(2, 0) * D(0, 0) - L(2, 1) * L(2, 1) * D(1, 1);
    }

    __forceinline__ __device__ mat3x1 solve(const mat3x1 &b)
    {
        // Ly = b
        /*
        mat3x1 y;
        y(0) = b(0);
        y(1) = b(1) - L(1, 0) * y(0);
        y(2) = b(2) - L(2, 0) * y(0) - L(2, 1) * y(1);

        // Dz = y
        mat3x1 z;
        z(0) = y(0) / D(0, 0);
        z(1) = y(1) / D(1, 1);
        z(2) = y(2) / D(2, 2);

        // L^Tx = z
        mat3x1 x;
        x(2) = z(2);
        x(1) = z(1) - L(2, 1) * x(2);
        x(0) = z(0) - L(1, 0) * x(1) - L(2, 0) * x(2);
        */

        // This version uses less register, good for cuda!
        mat3x1 x;
        x(0) = b(0);
        x(1) = b(1) - L(1, 0) * x(0);
        x(2) = b(2) - L(2, 0) * x(0) - L(2, 1) * x(1);

        x(0) /= D(0, 0);
        x(1) /= D(1, 1);
        x(2) /= D(2, 2);

        x(1) -= L(2, 1) * x(2);
        x(0) -= L(1, 0) * x(1) + L(2, 0) * x(2);

        return x;
    }

    __forceinline__ __device__ __host__ ~ldlt_3x3() = default;

private:
    mat3x3 L;
    mat3x3 D;
};
