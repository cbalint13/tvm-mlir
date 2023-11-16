#ifndef CONVERSION_PASSES_TD
#define CONVERSION_PASSES_TD

include "mlir/Pass/PassBase.td"

def RelayToAffine : Pass<"relay-to-affine", "mlir::ModuleOp"> {
  let summary = "Lower Relay to Affine.";
  let description = [{Lower TVM Relay to MLIR Affine dialect.}];

  let constructor = "createRelayToAffine()";
}

def AffineToLLVM : Pass<"affine-to-llvm", "mlir::ModuleOp"> {
  let summary = "Lower Affine to LLVM.";
  let description = [{Lower MLIR Affine dialect to LLVM IR.}];

  let constructor = "createAffineToLLVM()";
}

#endif
