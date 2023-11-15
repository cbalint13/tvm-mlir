#ifndef PASSES_TD
#define PASSES_TD

include "mlir/Pass/PassBase.td"

def ShapeInference : Pass<"relay-shape-inference", "mlir::func::FuncOp"> {
  let summary = "Shape inference.";
  let description = [{
    Check whether operator attributes and input tensor shapes satisfy
    constraints of operators, and infer shapes of output tensors.}];

  let constructor = "createShapeInference()";
}

#endif
