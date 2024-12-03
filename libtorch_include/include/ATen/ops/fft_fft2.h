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



#include <ATen/ops/fft_fft2_ops.h>

namespace at {


// aten::fft_fft2(Tensor self, SymInt[1]? s=None, int[1] dim=[-2,-1], str? norm=None) -> Tensor
inline at::Tensor fft_fft2(const at::Tensor & self, at::OptionalIntArrayRef s=::std::nullopt, at::IntArrayRef dim={-2,-1}, ::std::optional<c10::string_view> norm=::std::nullopt) {
    return at::_ops::fft_fft2::call(self, s.has_value() ? ::std::make_optional(c10::fromIntArrayRefSlow(*s)) : ::std::nullopt, dim, norm);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same<T, int64_t>::value>>
  at::Tensor fft_fft2(const at::Tensor & self, at::OptionalIntArrayRef s=::std::nullopt, at::IntArrayRef dim={-2,-1}, ::std::optional<c10::string_view> norm=::std::nullopt) {
    return at::_ops::fft_fft2::call(self, s.has_value() ? ::std::make_optional(c10::fromIntArrayRefSlow(*s)) : ::std::nullopt, dim, norm);
  }
}

// aten::fft_fft2(Tensor self, SymInt[1]? s=None, int[1] dim=[-2,-1], str? norm=None) -> Tensor
inline at::Tensor fft_fft2_symint(const at::Tensor & self, at::OptionalSymIntArrayRef s=::std::nullopt, at::IntArrayRef dim={-2,-1}, ::std::optional<c10::string_view> norm=::std::nullopt) {
    return at::_ops::fft_fft2::call(self, s, dim, norm);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same<T, c10::SymInt>::value>>
  at::Tensor fft_fft2(const at::Tensor & self, at::OptionalSymIntArrayRef s=::std::nullopt, at::IntArrayRef dim={-2,-1}, ::std::optional<c10::string_view> norm=::std::nullopt) {
    return at::_ops::fft_fft2::call(self, s, dim, norm);
  }
}

// aten::fft_fft2.out(Tensor self, SymInt[1]? s=None, int[1] dim=[-2,-1], str? norm=None, *, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & fft_fft2_out(at::Tensor & out, const at::Tensor & self, at::OptionalIntArrayRef s=::std::nullopt, at::IntArrayRef dim={-2,-1}, ::std::optional<c10::string_view> norm=::std::nullopt) {
    return at::_ops::fft_fft2_out::call(self, s.has_value() ? ::std::make_optional(c10::fromIntArrayRefSlow(*s)) : ::std::nullopt, dim, norm, out);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same<T, int64_t>::value>>
  at::Tensor & fft_fft2_out(at::Tensor & out, const at::Tensor & self, at::OptionalIntArrayRef s=::std::nullopt, at::IntArrayRef dim={-2,-1}, ::std::optional<c10::string_view> norm=::std::nullopt) {
    return at::_ops::fft_fft2_out::call(self, s.has_value() ? ::std::make_optional(c10::fromIntArrayRefSlow(*s)) : ::std::nullopt, dim, norm, out);
  }
}

// aten::fft_fft2.out(Tensor self, SymInt[1]? s=None, int[1] dim=[-2,-1], str? norm=None, *, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & fft_fft2_outf(const at::Tensor & self, at::OptionalIntArrayRef s, at::IntArrayRef dim, ::std::optional<c10::string_view> norm, at::Tensor & out) {
    return at::_ops::fft_fft2_out::call(self, s.has_value() ? ::std::make_optional(c10::fromIntArrayRefSlow(*s)) : ::std::nullopt, dim, norm, out);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same<T, int64_t>::value>>
  at::Tensor & fft_fft2_outf(const at::Tensor & self, at::OptionalIntArrayRef s, at::IntArrayRef dim, ::std::optional<c10::string_view> norm, at::Tensor & out) {
    return at::_ops::fft_fft2_out::call(self, s.has_value() ? ::std::make_optional(c10::fromIntArrayRefSlow(*s)) : ::std::nullopt, dim, norm, out);
  }
}

// aten::fft_fft2.out(Tensor self, SymInt[1]? s=None, int[1] dim=[-2,-1], str? norm=None, *, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & fft_fft2_symint_out(at::Tensor & out, const at::Tensor & self, at::OptionalSymIntArrayRef s=::std::nullopt, at::IntArrayRef dim={-2,-1}, ::std::optional<c10::string_view> norm=::std::nullopt) {
    return at::_ops::fft_fft2_out::call(self, s, dim, norm, out);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same<T, c10::SymInt>::value>>
  at::Tensor & fft_fft2_out(at::Tensor & out, const at::Tensor & self, at::OptionalSymIntArrayRef s=::std::nullopt, at::IntArrayRef dim={-2,-1}, ::std::optional<c10::string_view> norm=::std::nullopt) {
    return at::_ops::fft_fft2_out::call(self, s, dim, norm, out);
  }
}

// aten::fft_fft2.out(Tensor self, SymInt[1]? s=None, int[1] dim=[-2,-1], str? norm=None, *, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & fft_fft2_symint_outf(const at::Tensor & self, at::OptionalSymIntArrayRef s, at::IntArrayRef dim, ::std::optional<c10::string_view> norm, at::Tensor & out) {
    return at::_ops::fft_fft2_out::call(self, s, dim, norm, out);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same<T, c10::SymInt>::value>>
  at::Tensor & fft_fft2_outf(const at::Tensor & self, at::OptionalSymIntArrayRef s, at::IntArrayRef dim, ::std::optional<c10::string_view> norm, at::Tensor & out) {
    return at::_ops::fft_fft2_out::call(self, s, dim, norm, out);
  }
}

}
