#ifndef CONVERSION_PASSES_TD
#define CONVERSION_PASSES_TD

include "mlir/Pass/PassBase.td"

def RelayToAffine : Pass<"relay-to-affine", "mlir::ModuleOp"> {
  let summary = "Lower Relay to Affine.";
  let description = [{Lower TVM Relay to MLIR Affine dialect.}];

  let constructor = "createRelayToAffine()";
}

def AffineToSCF : Pass<"affine-to-scf", "mlir::ModuleOp"> {
  let summary = "Lower Affine to SCF.";
  let description = [{Lower MLIR Affine dialect to SCF dialect.}];

  let constructor = "createAffineToSCF()";
}

def SCFToLLVM : Pass<"scf-to-llvm", "mlir::ModuleOp"> {
  let summary = "Lower SCF to LLVM.";
  let description = [{Lower SCF dialect to LLVM IR.}];

  let constructor = "createSCFToLLVM()";
}

#endif
