#pragma once

// @generated by torchgen/gen.py from NativeMetaFunction.h

#include <c10/core/Scalar.h>
#include <c10/core/Storage.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/Deprecated.h>
#include <optional>
#include <c10/core/QScheme.h>
#include <ATen/core/Reduction.h>
#include <ATen/TensorIterator.h>
#include <ATen/TensorMeta.h>
#include <tuple>
#include <vector>

namespace at {
namespace meta {

struct TORCH_API structured_lcm : public TensorIteratorBase {
    
    
    void meta(const at::Tensor & self, const at::Tensor & other);
};

} // namespace native
} // namespace at
