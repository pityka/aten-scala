#pragma once

// @generated by torchgen/gen.py from Operator.h

#include <tuple>
#include <vector>

// Forward declarations of any types needed in the operator signatures.
// We can't directly include these classes because it will cause circular include dependencies.
// This file is included by TensorBody.h, which defines the Tensor class.
#include <ATen/core/ATen_fwd.h>

namespace at {
namespace _ops {


struct TORCH_API _index_put_impl_ {
  using schema = at::Tensor & (at::Tensor &, const c10::List<c10::optional<at::Tensor>> &, const at::Tensor &, bool, bool);
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "aten::_index_put_impl_")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, "_index_put_impl_(Tensor(a!) self, Tensor?[] indices, Tensor values, bool accumulate=False, bool unsafe=False) -> Tensor(a!)")
  static at::Tensor & call(at::Tensor & self, const c10::List<c10::optional<at::Tensor>> & indices, const at::Tensor & values, bool accumulate, bool unsafe);
  static at::Tensor & redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const c10::List<c10::optional<at::Tensor>> & indices, const at::Tensor & values, bool accumulate, bool unsafe);
};

struct TORCH_API _index_put_impl_out {
  using schema = at::Tensor & (const at::Tensor &, const c10::List<c10::optional<at::Tensor>> &, const at::Tensor &, bool, bool, at::Tensor &);
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "aten::_index_put_impl")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "out")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, "_index_put_impl.out(Tensor self, Tensor?[] indices, Tensor values, bool accumulate=False, bool unsafe=False, *, Tensor(a!) out) -> Tensor(a!)")
  static at::Tensor & call(const at::Tensor & self, const c10::List<c10::optional<at::Tensor>> & indices, const at::Tensor & values, bool accumulate, bool unsafe, at::Tensor & out);
  static at::Tensor & redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const c10::List<c10::optional<at::Tensor>> & indices, const at::Tensor & values, bool accumulate, bool unsafe, at::Tensor & out);
};

struct TORCH_API _index_put_impl_functional {
  using schema = at::Tensor (const at::Tensor &, const c10::List<c10::optional<at::Tensor>> &, const at::Tensor &, bool, bool);
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "aten::_index_put_impl")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "functional")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, "_index_put_impl.functional(Tensor self, Tensor?[] indices, Tensor values, bool accumulate=False, bool unsafe=False) -> Tensor")
  static at::Tensor call(const at::Tensor & self, const c10::List<c10::optional<at::Tensor>> & indices, const at::Tensor & values, bool accumulate, bool unsafe);
  static at::Tensor redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const c10::List<c10::optional<at::Tensor>> & indices, const at::Tensor & values, bool accumulate, bool unsafe);
};

}} // namespace at::_ops
