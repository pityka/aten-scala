#pragma once
// @generated by torchgen/gen.py from DispatchKeyFunction.h

// NB: The implementing C++ file is RegisterDispatchKey.cpp

// The only #includes we need are for custom classes that have defaults in the C++ API
#include <c10/core/MemoryFormat.h>
#include <c10/core/Scalar.h>
#include <ATen/core/Reduction.h>

// Forward declarations of any types needed in the operator signatures.
// We can't directly include these classes because it will cause circular include dependencies.
// This file is included by TensorBody.h, which defines the Tensor class.
#include <ATen/core/ATen_fwd.h>

namespace at {

namespace cuda {

TORCH_API at::Tensor _jagged_to_padded_dense_forward(const at::Tensor & values, at::TensorList offsets, at::IntArrayRef max_lengths, double padding_value=0.0);
TORCH_API at::Tensor _jagged_to_padded_dense_forward_symint(const at::Tensor & values, at::TensorList offsets, c10::SymIntArrayRef max_lengths, double padding_value=0.0);

} // namespace cuda
} // namespace at