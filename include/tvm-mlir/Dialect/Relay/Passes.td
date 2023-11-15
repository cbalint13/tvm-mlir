#ifndef RELAY_PASSES_TD
#define RELAY_PASSES_TD

include "mlir/Pass/PassBase.td"

def ShapeInference : Pass<"relay-shape-inference", "mlir::func::FuncOp"> {
  let summary = "Shape inference.";
  let description = [{
    Check whether operator attributes and input tensor shapes satisfy
    constraints of operators, and infer shapes of output tensors.}];

  let constructor = "createShapeInference()";
}

def OpFusion : Pass<"relay-op-fusion", "mlir::func::FuncOp"> {
  let summary = "Operator fusion.";
  let description = [{
    Fuse operators in the main function to a larger function according
    to some rules.}];

  let constructor = "createOpFusion()";
}

#endif
