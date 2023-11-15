#pragma once
#include <mlir/Pass/Pass.h>

#include <memory>

namespace relay {

std::unique_ptr<mlir::Pass> createShapeInference();

#define GEN_PASS_REGISTRATION
#include "tvm-mlir/Dialect/Relay/Passes.h.inc"

} // namespace relay
