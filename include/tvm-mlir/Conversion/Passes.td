#ifndef CONVERSION_PASSES_TD
#define CONVERSION_PASSES_TD

include "mlir/Pass/PassBase.td"

def RelayToAffine : Pass<"relay-to-affine", "mlir::ModuleOp"> {
  let summary = "Lower Relay to Affine.";
  let description = [{
    Pass that converts Relay operations to the implementation
    with the low-level operations in Affine dialect.}];

  let constructor = "createRelayToAffine()";
}

#endif
