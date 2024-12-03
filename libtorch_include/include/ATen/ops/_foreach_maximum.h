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



#include <ATen/ops/_foreach_maximum_ops.h>

namespace at {


// aten::_foreach_maximum.Scalar(Tensor[] self, Scalar scalar) -> Tensor[]
inline ::std::vector<at::Tensor> _foreach_maximum(at::TensorList self, const at::Scalar & scalar) {
    return at::_ops::_foreach_maximum_Scalar::call(self, scalar);
}

// aten::_foreach_maximum_.Scalar(Tensor(a!)[] self, Scalar scalar) -> ()
inline void _foreach_maximum_(at::TensorList self, const at::Scalar & scalar) {
    return at::_ops::_foreach_maximum__Scalar::call(self, scalar);
}

// aten::_foreach_maximum.List(Tensor[] self, Tensor[] other) -> Tensor[]
inline ::std::vector<at::Tensor> _foreach_maximum(at::TensorList self, at::TensorList other) {
    return at::_ops::_foreach_maximum_List::call(self, other);
}

// aten::_foreach_maximum_.List(Tensor(a!)[] self, Tensor[] other) -> ()
inline void _foreach_maximum_(at::TensorList self, at::TensorList other) {
    return at::_ops::_foreach_maximum__List::call(self, other);
}

// aten::_foreach_maximum.ScalarList(Tensor[] self, Scalar[] scalars) -> Tensor[]
inline ::std::vector<at::Tensor> _foreach_maximum(at::TensorList self, at::ArrayRef<at::Scalar> scalars) {
    return at::_ops::_foreach_maximum_ScalarList::call(self, scalars);
}

// aten::_foreach_maximum_.ScalarList(Tensor(a!)[] self, Scalar[] scalars) -> ()
inline void _foreach_maximum_(at::TensorList self, at::ArrayRef<at::Scalar> scalars) {
    return at::_ops::_foreach_maximum__ScalarList::call(self, scalars);
}

// aten::_foreach_maximum.Scalar_out(Tensor[] self, Scalar scalar, *, Tensor(a!)[] out) -> ()
inline void _foreach_maximum_out(at::TensorList out, at::TensorList self, const at::Scalar & scalar) {
    return at::_ops::_foreach_maximum_Scalar_out::call(self, scalar, out);
}
// aten::_foreach_maximum.Scalar_out(Tensor[] self, Scalar scalar, *, Tensor(a!)[] out) -> ()
inline void _foreach_maximum_outf(at::TensorList self, const at::Scalar & scalar, at::TensorList out) {
    return at::_ops::_foreach_maximum_Scalar_out::call(self, scalar, out);
}

// aten::_foreach_maximum.List_out(Tensor[] self, Tensor[] other, *, Tensor(a!)[] out) -> ()
inline void _foreach_maximum_out(at::TensorList out, at::TensorList self, at::TensorList other) {
    return at::_ops::_foreach_maximum_List_out::call(self, other, out);
}
// aten::_foreach_maximum.List_out(Tensor[] self, Tensor[] other, *, Tensor(a!)[] out) -> ()
inline void _foreach_maximum_outf(at::TensorList self, at::TensorList other, at::TensorList out) {
    return at::_ops::_foreach_maximum_List_out::call(self, other, out);
}

// aten::_foreach_maximum.ScalarList_out(Tensor[] self, Scalar[] scalars, *, Tensor(a!)[] out) -> ()
inline void _foreach_maximum_out(at::TensorList out, at::TensorList self, at::ArrayRef<at::Scalar> scalars) {
    return at::_ops::_foreach_maximum_ScalarList_out::call(self, scalars, out);
}
// aten::_foreach_maximum.ScalarList_out(Tensor[] self, Scalar[] scalars, *, Tensor(a!)[] out) -> ()
inline void _foreach_maximum_outf(at::TensorList self, at::ArrayRef<at::Scalar> scalars, at::TensorList out) {
    return at::_ops::_foreach_maximum_ScalarList_out::call(self, scalars, out);
}

}
