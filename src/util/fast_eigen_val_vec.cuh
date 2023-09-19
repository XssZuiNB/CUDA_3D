#pragma once

#include "util/math.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <thrust/pair.h>
#include <thrust/tuple.h>

// The document
// https://www.geometrictools.com/Documentation/RobustEigenSymmetric3x3.pdf
// describes algorithms for solving the eigensystem associated with a 3x3
// symmetric！！！ real-valued matrix.

__forceinline__ __device__ static float3 __compute_eigenvector_1(const float3x3 &A, float eval1)
{
    float3 row1 = make_float3(A.m11 - eval1, A.m12, A.m13);
    float3 row2 = make_float3(A.m12, A.m22 - eval1, A.m23);
    float3 row3 = make_float3(A.m13, A.m23, A.m33 - eval1);

    float3 rxr[3];
    rxr[0] = cross(row1, row2);
    rxr[1] = cross(row1, row3);
    rxr[2] = cross(row2, row3);

    float3 d;
    d.x = dot(rxr[0], rxr[0]);
    d.y = dot(rxr[1], rxr[1]);
    d.z = dot(rxr[2], rxr[2]);

    int imax;
    auto max_d = max_coeff(d, imax);
    return rxr[imax] / sqrtf(max_d);
}

__forceinline__ __device__ static float3 __compute_eigenvector_2(const float3x3 &A,
                                                                 const float3 &evec1, float eval2)
{
    float max_evec1_abs = fmaxf(fabsf(evec1.x), fabsf(evec1.y));
    float inv_length = 1 / sqrtf(max_evec1_abs * max_evec1_abs + evec1.z * evec1.z);

    float3 U = (fabsf(evec1.x) > fabsf(evec1.y)) ? make_float3(-evec1.z, 0, evec1.x)
                                                 : make_float3(0, evec1.z, -evec1.y);

    U *= inv_length;

    float3 V = cross(evec1, U);

    float3 AU = A * U;

    float3 AV = A * V;

    float m00 = dot(U, AU) - eval2;
    float m01 = dot(U, AV);
    float m11 = dot(V, AV) - eval2;

    float abs_m00 = fabsf(m00);
    float abs_m01 = fabsf(m01);
    float abs_m11 = fabsf(m11);

    float max_abs_comp;
    if (abs_m00 >= abs_m11)
    {
        max_abs_comp = fmaxf(abs_m00, abs_m01);
        if (max_abs_comp > 0.0f)
        {
            if (abs_m00 >= abs_m01)
            {
                m01 /= m00;
                m00 = 1.0f / sqrtf(1.0f + m01 * m01);
                m01 *= m00;
            }
            else
            {
                m00 /= m01;
                m01 = 1.0f / sqrtf(1.0f + m00 * m00);
                m00 *= m01;
            }
            return U * m01 - V * m00;
        }
        else
        {
            return U;
        }
    }
    else
    {
        max_abs_comp = fmaxf(abs_m11, abs_m01);
        if (max_abs_comp > 0.0f)
        {
            if (abs_m11 >= abs_m01)
            {
                m01 /= m11;
                m11 = 1.0f / sqrtf(1.0f + m01 * m01);
                m01 *= m11;
            }
            else
            {
                m11 /= m01;
                m01 = 1.0f / sqrtf(1.0f + m11 * m11);
                m11 *= m01;
            }
            return U * m11 - V * m01;
        }
        else
        {
            return U;
        }
    }
}

using eval_evec_pair = thrust::pair<float, float3>;
// The input matrix must be symmetric ！！！
__forceinline__ __device__ auto fast_eigen_compute_3x3_symm(const float3x3 &A)
    -> thrust::tuple<eval_evec_pair, eval_evec_pair, eval_evec_pair>
{
    auto row1 = A.get_row(0);
    auto row2 = A.get_row(1);
    auto row3 = A.get_row(2);

    auto abs_max_1 = fmaxf(fmaxf(fabsf(row1.x), fabsf(row1.y)), fabsf(row1.z));
    auto abs_max_2 = fmaxf(abs_max_1, fmaxf(fabsf(row2.y), fabsf(row2.z)));
    auto abs_max_coeff = fmaxf(abs_max_2, fabsf(row3.z));
    if (abs_max_coeff == 0.0f)
    {
        // A is the zero matrix.
        auto eval_evec_1 = thrust::make_pair<float, float3>(0.0f, {1.0f, 0.0f, 0.0f});
        auto eval_evec_2 = thrust::make_pair<float, float3>(0.0f, {0.0f, 1.0f, 0.0f});
        auto eval_evec_3 = thrust::make_pair<float, float3>(0.0f, {0.0f, 0.0f, 1.0f});

        return thrust::make_tuple(eval_evec_1, eval_evec_2, eval_evec_3);
    }

    float inv_abs_max_coeff = 1.0f / abs_max_coeff;
    row1 *= inv_abs_max_coeff;
    row2 *= inv_abs_max_coeff;
    row3 *= inv_abs_max_coeff;

    float norm = row1.y * row1.y + row1.z * row1.z + row2.z * row2.z;
    if (norm > 0)
    {
        float q = (row1.x + row2.y + row3.z) / 3.0f;

        float b00 = row1.x - q;
        float b11 = row2.y - q;
        float b22 = row3.z - q;

        float p = sqrtf((b00 * b00 + b11 * b11 + b22 * b22 + norm * 2) / 6);

        float c00 = b11 * b22 - row2.z * row2.z;
        float c01 = row1.y * b22 - row2.z * row1.z;
        float c02 = row1.y * row2.z - b11 * row1.z;
        float det = (b00 * c00 - row1.y * c01 + row1.z * c02) / (p * p * p);

        float half_det = det * 0.5;
        half_det = fminf(fmaxf(half_det, -1.0), 1.0);

        float angle = acos(half_det) / 3.0f;
        float constexpr two_thirds_pi = 2.09439510239319549;
        float beta2 = cos(angle) * 2;
        float beta0 = cos(angle + two_thirds_pi) * 2;
        float beta1 = -(beta0 + beta2);

        auto eval_1 = q + p * beta0;
        auto eval_2 = q + p * beta1;
        auto eval_3 = q + p * beta2;

        if (half_det >= 0)
        {
            float3x3 A_(row1, row2, row3);
            auto evec_3 = __compute_eigenvector_1(A_, eval_3);
            auto evec_2 = __compute_eigenvector_2(A_, evec_3, eval_2);
            auto evec_1 = cross(evec_2, evec_3);

            auto eval_evec_1 = thrust::make_pair<float, float3>(eval_1 * abs_max_coeff, evec_1);
            auto eval_evec_2 = thrust::make_pair<float, float3>(eval_2 * abs_max_coeff, evec_2);
            auto eval_evec_3 = thrust::make_pair<float, float3>(eval_3 * abs_max_coeff, evec_3);

            return thrust::make_tuple(eval_evec_1, eval_evec_2, eval_evec_3);
        }
        else
        {
            float3x3 A_(row1, row2, row3);
            auto evec_1 = __compute_eigenvector_1(A_, eval_1);
            auto evec_2 = __compute_eigenvector_2(A_, evec_1, eval_2);
            auto evec_3 = cross(evec_1, evec_2);

            auto eval_evec_1 = thrust::make_pair<float, float3>(eval_1 * abs_max_coeff, evec_1);
            auto eval_evec_2 = thrust::make_pair<float, float3>(eval_2 * abs_max_coeff, evec_2);
            auto eval_evec_3 = thrust::make_pair<float, float3>(eval_3 * abs_max_coeff, evec_3);

            return thrust::make_tuple(eval_evec_1, eval_evec_2, eval_evec_3);
        }
    }
    else
    {
        // A is diagonal.
        auto eval_evec_1 = thrust::make_pair<float, float3>(A.m11, {1.0f, 0.0f, 0.0f});
        auto eval_evec_2 = thrust::make_pair<float, float3>(A.m22, {0.0f, 1.0f, 0.0f});
        auto eval_evec_3 = thrust::make_pair<float, float3>(A.m33, {0.0f, 0.0f, 1.0f});

        return thrust::make_tuple(eval_evec_1, eval_evec_2, eval_evec_3);
    }
}
