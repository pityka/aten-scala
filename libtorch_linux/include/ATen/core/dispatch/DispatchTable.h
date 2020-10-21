#pragma once

#include <ATen/core/function_schema.h>
#include <c10/util/Metaprogramming.h>
#include <c10/util/flat_hash_map.h>
#include <c10/util/either.h>
#include <c10/core/DispatchKey.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/boxing/KernelFunction.h>
#include <ATen/core/dispatch/DispatchKeyExtractor.h>

#include <array>
#include <atomic>
#include <iostream>
#include <mutex>
#include <type_traits>
#include <sstream>
#include <unordered_map>
#include <functional>

namespace c10 {

namespace impl {
/**
 * A KernelFunctionTable is a map from DispatchKey to a KernelFunction.
 * It can store zero or one KernelFunctions for each DispatchKey.
 */
class KernelFunctionTable final {
public:
  explicit KernelFunctionTable()
  : kernels_()
  , kernelCount_(0) {}

  void setKernel(DispatchKey dispatchKey, KernelFunction kernel) {
    TORCH_INTERNAL_ASSERT(dispatchKey != DispatchKey::Undefined);
    auto& slot = kernels_[static_cast<uint8_t>(dispatchKey)];
    if (!slot.isValid()) {
      ++kernelCount_;
    }
    slot = std::move(kernel);
  }

  void removeKernelIfExists(DispatchKey dispatchKey) {
    auto& slot = kernels_[static_cast<uint8_t>(dispatchKey)];
    if (slot.isValid()) {
      --kernelCount_;
      slot = {};
    } else {
    }
  }

  const KernelFunction& operator[](DispatchKey dispatchKey) const {
    return kernels_[static_cast<uint8_t>(dispatchKey)];
  }

  KernelFunction& operator[](DispatchKey dispatchKey) {
    return kernels_[static_cast<uint8_t>(dispatchKey)];
  }

  size_t size() const {
    return kernelCount_;
  }

  std::string dumpState() const;

private:
  std::array<KernelFunction, static_cast<uint8_t>(DispatchKey::NumDispatchKeys)> kernels_;
  size_t kernelCount_;
};
}

/**
 * Per-operator dispatch table.
 *
 * Given an operator specified by a FunctionSchema, this class records a dispatch
 * table for various kernels provided for this operator.  For example, if we
 * consider the operator add(Tensor, Tensor), the dispatch table for this
 * operator may contain implementations for various dynamic tensor types, such
 * as CPU, CUDA, etc.
 */
class DispatchTable final {
 public:
  explicit DispatchTable(const FunctionSchema& schema)
  : kernels_()
  , catchallKernel_()
  , dispatchKeyExtractor_(DispatchKeyExtractor::make(schema))
  , operatorName_(schema.operator_name()) {}

  // a dispatch table may be default constructed with only an
  // operator name.  Such a dispatch table is not callable until
  // the schema is provided
  DispatchTable(OperatorName op_name)
  : kernels_()
  , catchallKernel_()
  , dispatchKeyExtractor_(DispatchKeyExtractor::makeUninitialized())
  , operatorName_(std::move(op_name)) {}

  /**
   * Register a kernel in the table at some dispatch key.
   * @param dispatch_key Dispatch key to define when this kernel is selected.
   * @param kernel Concrete kernel function implementation to register
   */
  void setKernel(DispatchKey dispatchKey, KernelFunction kernel) {
    if (manuallyBoxedKernel_.has_value()) {
      kernel.setManuallyBoxedKernel_(*manuallyBoxedKernel_);
    }
    kernels_.setKernel(dispatchKey, std::move(kernel));
    dispatchKeyExtractor_.setOperatorHasKernelForBackend(dispatchKey, true);
    if (kernel.isFallthrough()) {
      dispatchKeyExtractor_.setOperatorHasFallthroughForBackend(dispatchKey, true);
    }
  }

  /**
   * Deregister the kernel for some dispatch key.
   *
   * @param dispatch_key Dispatch key to unregister.
   */
  void removeKernelIfExists(DispatchKey dispatchKey) {
    kernels_.removeKernelIfExists(dispatchKey);
    dispatchKeyExtractor_.setOperatorHasKernelForBackend(dispatchKey, false);
    dispatchKeyExtractor_.setOperatorHasFallthroughForBackend(dispatchKey, false); // may be no op
  }

  /**
   * Register a catch-all kernel that is called for this operator
   * independent of the inputs. An operator can have either
   * a catch-all kernel or a set of kernels with concrete
   * dispatch keys, not both.
   */
  void setCatchallKernel(KernelFunction kernel) {
    if (manuallyBoxedKernel_.has_value()) {
      kernel.setManuallyBoxedKernel_(*manuallyBoxedKernel_);
    }
    catchallKernel_ = std::move(kernel);
  }

  /**
   * Remove the catch-all kernel.
   */
  void removeCatchallKernel() {
    catchallKernel_ = {};
  }

  bool isEmpty() const {
    return !catchallKernel_.isValid() && kernels_.size() == 0;
  }

  std::string listAllDispatchKeys() const {
    std::ostringstream str;
    str << "[";

    bool has_kernels = false;
    for (uint8_t iter = 0; iter != static_cast<uint8_t>(DispatchKey::NumDispatchKeys); ++iter) {
      if (!kernels_[static_cast<DispatchKey>(iter)].isValid()) {
        continue;
      }
      if (has_kernels) {
        str << ", ";
      }
      str << static_cast<DispatchKey>(iter);
      has_kernels = true;
    }

    if (catchallKernel_.isValid()) {
      if (has_kernels) {
        str << ", ";
      }
      str << "CATCH-ALL";
    }
    str << "]";
    return str.str();
  }

  const KernelFunction* lookup(DispatchKey dispatchKey) const {
    auto& slot = kernels_[dispatchKey];
    // TODO: this condition shouldn't be necessary
    if (slot.isValid()) {
      return &slot;
    } else {
      return nullptr;
    }
  }

  const KernelFunction* lookupCatchallKernel() const {
    // TODO: this condition shouldn't be necessary
    if (!catchallKernel_.isValid()) {
      return nullptr;
    }

    return &catchallKernel_;
  }

  const DispatchKeyExtractor& dispatchKeyExtractor() const {
    return dispatchKeyExtractor_;
  }

  const OperatorName& operatorName() const {
    return operatorName_;
  }

  void registerSchema(const FunctionSchema& schema) {
    dispatchKeyExtractor_.registerSchema(schema);
  }

  void deregisterSchema() {
    dispatchKeyExtractor_.deregisterSchema();
  }

  std::string dumpState() const;

  // This function is a temporary hack, see comment at manuallyBoxedKernel_ member
  void setManuallyBoxedKernel_(KernelFunction::InternalBoxedKernelFunction* func) {
    TORCH_INTERNAL_ASSERT(!manuallyBoxedKernel_.has_value(), "Cannot set multiple manually boxed kernels for the same operator ", operatorName_);
    manuallyBoxedKernel_ = func;

    // make sure that all previously registered kernels get this manually boxed kernel
    for (uint8_t iter = 0; iter != static_cast<uint8_t>(DispatchKey::NumDispatchKeys); ++iter) {
      auto& kernel = kernels_[static_cast<DispatchKey>(iter)];
      if (kernel.isValid()) {
        kernel.setManuallyBoxedKernel_(func);
      }
    }
    if (catchallKernel_.isValid()) {
      catchallKernel_.setManuallyBoxedKernel_(func);
    }
  }

  c10::optional<KernelFunction::InternalBoxedKernelFunction*> manuallyBoxedKernel() const {
    return manuallyBoxedKernel_;
  }

private:

  impl::KernelFunctionTable kernels_;
  KernelFunction catchallKernel_;
  DispatchKeyExtractor dispatchKeyExtractor_;
  OperatorName operatorName_;

  // This manuallyBoxedKernel_ member is a temporary hack that allows generated_unboxing_wrappers.cpp to register its codegen'ed
  // unboxing wrapper for aten operators. We still need those for some operators because not all work
  // with the templated unboxing logic yet.
  // TODO Delete manuallyBoxedKernel_ once all operators work with the templated boxing logic
  c10::optional<KernelFunction::InternalBoxedKernelFunction*> manuallyBoxedKernel_;
};

} // namespace c10
