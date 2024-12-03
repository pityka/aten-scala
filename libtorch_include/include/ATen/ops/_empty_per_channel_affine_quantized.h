#pragma once

// @generated by torchgen/gen.py from Function.h

#include <ATen/Context.h>
#include <ATen/DeviceGuard.h>
#include <ATen/TensorUtils.h>
#include <ATen/TracerMode.h>
#include <ATen/core/Generator.h>
#include <ATen/core/Reduction.h>
#include <ATen/core/Tensor.h>
#include <c10/core/Scalar.h>
#include <c10/core/Storage.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/Deprecated.h>
#include <optional>



#include <ATen/ops/_empty_per_channel_affine_quantized_ops.h>

namespace at {


// aten::_empty_per_channel_affine_quantized(SymInt[] size, *, Tensor scales, Tensor zero_points, int axis, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=contiguous_format) -> Tensor
inline at::Tensor _empty_per_channel_affine_quantized(at::IntArrayRef size, const at::Tensor & scales, const at::Tensor & zero_points, int64_t axis, at::TensorOptions options={}, ::std::optional<at::MemoryFormat> memory_format=c10::MemoryFormat::Contiguous) {
    return at::_ops::_empty_per_channel_affine_quantized::call(c10::fromIntArrayRefSlow(size), scales, zero_points, axis, c10::optTypeMetaToScalarType(options.dtype_opt()), options.layout_opt(), options.device_opt(), options.pinned_memory_opt(), c10::impl::check_tensor_options_and_extract_memory_format(options, memory_format));
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same<T, int64_t>::value>>
  at::Tensor _empty_per_channel_affine_quantized(at::IntArrayRef size, const at::Tensor & scales, const at::Tensor & zero_points, int64_t axis, at::TensorOptions options={}, ::std::optional<at::MemoryFormat> memory_format=c10::MemoryFormat::Contiguous) {
    return at::_ops::_empty_per_channel_affine_quantized::call(c10::fromIntArrayRefSlow(size), scales, zero_points, axis, c10::optTypeMetaToScalarType(options.dtype_opt()), options.layout_opt(), options.device_opt(), options.pinned_memory_opt(), c10::impl::check_tensor_options_and_extract_memory_format(options, memory_format));
  }
}

// aten::_empty_per_channel_affine_quantized(SymInt[] size, *, Tensor scales, Tensor zero_points, int axis, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=contiguous_format) -> Tensor
inline at::Tensor _empty_per_channel_affine_quantized(at::IntArrayRef size, const at::Tensor & scales, const at::Tensor & zero_points, int64_t axis, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory, ::std::optional<at::MemoryFormat> memory_format) {
    return at::_ops::_empty_per_channel_affine_quantized::call(c10::fromIntArrayRefSlow(size), scales, zero_points, axis, dtype, layout, device, pin_memory, memory_format);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same<T, int64_t>::value>>
  at::Tensor _empty_per_channel_affine_quantized(at::IntArrayRef size, const at::Tensor & scales, const at::Tensor & zero_points, int64_t axis, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory, ::std::optional<at::MemoryFormat> memory_format) {
    return at::_ops::_empty_per_channel_affine_quantized::call(c10::fromIntArrayRefSlow(size), scales, zero_points, axis, dtype, layout, device, pin_memory, memory_format);
  }
}

// aten::_empty_per_channel_affine_quantized(SymInt[] size, *, Tensor scales, Tensor zero_points, int axis, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=contiguous_format) -> Tensor
inline at::Tensor _empty_per_channel_affine_quantized_symint(c10::SymIntArrayRef size, const at::Tensor & scales, const at::Tensor & zero_points, int64_t axis, at::TensorOptions options={}, ::std::optional<at::MemoryFormat> memory_format=c10::MemoryFormat::Contiguous) {
    return at::_ops::_empty_per_channel_affine_quantized::call(size, scales, zero_points, axis, c10::optTypeMetaToScalarType(options.dtype_opt()), options.layout_opt(), options.device_opt(), options.pinned_memory_opt(), c10::impl::check_tensor_options_and_extract_memory_format(options, memory_format));
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same<T, c10::SymInt>::value>>
  at::Tensor _empty_per_channel_affine_quantized(c10::SymIntArrayRef size, const at::Tensor & scales, const at::Tensor & zero_points, int64_t axis, at::TensorOptions options={}, ::std::optional<at::MemoryFormat> memory_format=c10::MemoryFormat::Contiguous) {
    return at::_ops::_empty_per_channel_affine_quantized::call(size, scales, zero_points, axis, c10::optTypeMetaToScalarType(options.dtype_opt()), options.layout_opt(), options.device_opt(), options.pinned_memory_opt(), c10::impl::check_tensor_options_and_extract_memory_format(options, memory_format));
  }
}

// aten::_empty_per_channel_affine_quantized(SymInt[] size, *, Tensor scales, Tensor zero_points, int axis, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=contiguous_format) -> Tensor
inline at::Tensor _empty_per_channel_affine_quantized_symint(c10::SymIntArrayRef size, const at::Tensor & scales, const at::Tensor & zero_points, int64_t axis, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory, ::std::optional<at::MemoryFormat> memory_format) {
    return at::_ops::_empty_per_channel_affine_quantized::call(size, scales, zero_points, axis, dtype, layout, device, pin_memory, memory_format);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same<T, c10::SymInt>::value>>
  at::Tensor _empty_per_channel_affine_quantized(c10::SymIntArrayRef size, const at::Tensor & scales, const at::Tensor & zero_points, int64_t axis, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory, ::std::optional<at::MemoryFormat> memory_format) {
    return at::_ops::_empty_per_channel_affine_quantized::call(size, scales, zero_points, axis, dtype, layout, device, pin_memory, memory_format);
  }
}

// aten::_empty_per_channel_affine_quantized.out(SymInt[] size, *, Tensor scales, Tensor zero_points, int axis, MemoryFormat? memory_format=contiguous_format, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & _empty_per_channel_affine_quantized_out(at::Tensor & out, at::IntArrayRef size, const at::Tensor & scales, const at::Tensor & zero_points, int64_t axis, ::std::optional<at::MemoryFormat> memory_format=c10::MemoryFormat::Contiguous) {
    return at::_ops::_empty_per_channel_affine_quantized_out::call(c10::fromIntArrayRefSlow(size), scales, zero_points, axis, memory_format, out);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same<T, int64_t>::value>>
  at::Tensor & _empty_per_channel_affine_quantized_out(at::Tensor & out, at::IntArrayRef size, const at::Tensor & scales, const at::Tensor & zero_points, int64_t axis, ::std::optional<at::MemoryFormat> memory_format=c10::MemoryFormat::Contiguous) {
    return at::_ops::_empty_per_channel_affine_quantized_out::call(c10::fromIntArrayRefSlow(size), scales, zero_points, axis, memory_format, out);
  }
}

// aten::_empty_per_channel_affine_quantized.out(SymInt[] size, *, Tensor scales, Tensor zero_points, int axis, MemoryFormat? memory_format=contiguous_format, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & _empty_per_channel_affine_quantized_outf(at::IntArrayRef size, const at::Tensor & scales, const at::Tensor & zero_points, int64_t axis, ::std::optional<at::MemoryFormat> memory_format, at::Tensor & out) {
    return at::_ops::_empty_per_channel_affine_quantized_out::call(c10::fromIntArrayRefSlow(size), scales, zero_points, axis, memory_format, out);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same<T, int64_t>::value>>
  at::Tensor & _empty_per_channel_affine_quantized_outf(at::IntArrayRef size, const at::Tensor & scales, const at::Tensor & zero_points, int64_t axis, ::std::optional<at::MemoryFormat> memory_format, at::Tensor & out) {
    return at::_ops::_empty_per_channel_affine_quantized_out::call(c10::fromIntArrayRefSlow(size), scales, zero_points, axis, memory_format, out);
  }
}

// aten::_empty_per_channel_affine_quantized.out(SymInt[] size, *, Tensor scales, Tensor zero_points, int axis, MemoryFormat? memory_format=contiguous_format, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & _empty_per_channel_affine_quantized_symint_out(at::Tensor & out, c10::SymIntArrayRef size, const at::Tensor & scales, const at::Tensor & zero_points, int64_t axis, ::std::optional<at::MemoryFormat> memory_format=c10::MemoryFormat::Contiguous) {
    return at::_ops::_empty_per_channel_affine_quantized_out::call(size, scales, zero_points, axis, memory_format, out);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same<T, c10::SymInt>::value>>
  at::Tensor & _empty_per_channel_affine_quantized_out(at::Tensor & out, c10::SymIntArrayRef size, const at::Tensor & scales, const at::Tensor & zero_points, int64_t axis, ::std::optional<at::MemoryFormat> memory_format=c10::MemoryFormat::Contiguous) {
    return at::_ops::_empty_per_channel_affine_quantized_out::call(size, scales, zero_points, axis, memory_format, out);
  }
}

// aten::_empty_per_channel_affine_quantized.out(SymInt[] size, *, Tensor scales, Tensor zero_points, int axis, MemoryFormat? memory_format=contiguous_format, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & _empty_per_channel_affine_quantized_symint_outf(c10::SymIntArrayRef size, const at::Tensor & scales, const at::Tensor & zero_points, int64_t axis, ::std::optional<at::MemoryFormat> memory_format, at::Tensor & out) {
    return at::_ops::_empty_per_channel_affine_quantized_out::call(size, scales, zero_points, axis, memory_format, out);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same<T, c10::SymInt>::value>>
  at::Tensor & _empty_per_channel_affine_quantized_outf(c10::SymIntArrayRef size, const at::Tensor & scales, const at::Tensor & zero_points, int64_t axis, ::std::optional<at::MemoryFormat> memory_format, at::Tensor & out) {
    return at::_ops::_empty_per_channel_affine_quantized_out::call(size, scales, zero_points, axis, memory_format, out);
  }
}

}
