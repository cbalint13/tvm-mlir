#pragma once
#include <mlir/Pass/Pass.h>

#include <memory>

// Optimze Affine Pass
std::unique_ptr<mlir::Pass> createOptimizeAffine();

#define GEN_PASS_REGISTRATION
#include "tvm-mlir/Transforms/Passes.h.inc"
