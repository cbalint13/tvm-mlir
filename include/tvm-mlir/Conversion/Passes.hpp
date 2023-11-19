#pragma once
#include <mlir/Pass/Pass.h>

#include <memory>

/// TVM relay to affine pass
std::unique_ptr<mlir::Pass> createRelayToAffine();

/// MLIR affine to SCF pass
std::unique_ptr<mlir::Pass> createAffineToSCF();

/// MLIR SCF to LLVM pass
std::unique_ptr<mlir::Pass> createSCFToLLVM();

#define GEN_PASS_REGISTRATION
#include "tvm-mlir/Conversion/Passes.h.inc"
