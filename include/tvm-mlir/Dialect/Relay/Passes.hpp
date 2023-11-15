#pragma once
#include <mlir/Pass/Pass.h>

#include <memory>

namespace relay {

/// Shape inference pass
std::unique_ptr<mlir::Pass> createShapeInference();
/// Operator fusion pass
std::unique_ptr<mlir::Pass> createOpFusion();

#define GEN_PASS_REGISTRATION
#include "tvm-mlir/Dialect/Relay/Passes.h.inc"

} // namespace relay
