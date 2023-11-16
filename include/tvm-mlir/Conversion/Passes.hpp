#pragma once
#include <mlir/Pass/Pass.h>

#include <memory>

/// TVM relay to affine pass
std::unique_ptr<mlir::Pass> createRelayToAffine();

/// MLIR affine to LLVM pass
std::unique_ptr<mlir::Pass> createAffineToLLVM();

#define GEN_PASS_REGISTRATION
#include "tvm-mlir/Conversion/Passes.h.inc"
