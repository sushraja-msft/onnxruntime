
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda_runtime.h>

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
void QOrderQuantize(
    cudaStream_t stream, const cudaDeviceProp& device_prop,
    const T* src, int8_t* dst, size_t N, T scale);

template <typename T>
void QOrderDequantize(
    cudaStream_t stream, const cudaDeviceProp& device_prop,
    const int8_t* src, T* dst, size_t N, T scale);

void QOrderQuantizeRowToCol32(
    cudaStream_t stream, const cudaDeviceProp& device_prop,
    const __half* src, int8_t* dst, float scale, unsigned batch, unsigned rows, unsigned cols);

void QOrderQuantizeRowToCol32(
    cudaStream_t stream, const cudaDeviceProp& device_prop,
    const float* src, int8_t* dst, float scale, unsigned batch, unsigned rows, unsigned cols);

void QOrderDequantizeCol32ToRow(
    cudaStream_t stream, const cudaDeviceProp& device_prop,
    const int8_t* src, __half* dst, float scale,
    unsigned batch, unsigned rows, unsigned cols);

void QOrderDequantizeCol32ToRow(
    cudaStream_t stream, const cudaDeviceProp& device_prop,
    const int8_t* src, float* dst, float scale,
    unsigned batch, unsigned rows, unsigned cols);

void QOrderAddBiasResidualLayerNorm(
    cudaStream_t stream, const cudaDeviceProp& device_prop, cublasLtOrder_t order,
    const int8_t* src, const float src_scale,
    const int8_t* residual, const float residual_scale,
    const __half* bias,
    int8_t* dst, const float dst_scale,
    const __half* gamma, const __half* beta, const float epsilon,
    const unsigned batch, const unsigned rows, const unsigned cols);

void ReorderS8RowToCol32(
    cudaStream_t stream, const cudaDeviceProp& device_prop,
    const int8_t* src, int8_t* dst,
    unsigned batch, unsigned rows, unsigned cols);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
