#include "tvm-mlir/Dialect/Relay/Passes.hpp"
#include "tvm-mlir/Relay/PassDetail.hpp"
#include "tvm-mlir/Support/Common.hpp"

namespace relay {

class ShapeInference : public ShapeInferenceBase<ShapeInference> {
  void runOnOperation() override;
};

void ShapeInference::runOnOperation() {
  // Infer types of Relay operators
  auto func = getOperation();
  mlir::ValueRange retValues;
  func.walk([&](mlir::Operation *op) {
    // Get return value of the function
    if (mlir::func::ReturnOp::classof(op)) {
      retValues = llvm::cast<mlir::func::ReturnOp>(op).getOperands();
      return;
    }

    // Skip operators not defined in Relay dialect
    if (op->getDialect()->getNamespace() != RelayDialect::getDialectNamespace())
      return;

    // Infer type with operator interface
    auto opInterface = mlir::dyn_cast<mlir::InferShapedTypeOpInterface>(op);
    if (!opInterface) {
      return;
    }
    llvm::SmallVector<mlir::ShapedTypeComponents> inferredShapes;
    auto result = opInterface.inferReturnTypeComponents(
        op->getContext(), op->getLoc(), op->getOperands(),
        op->getAttrDictionary(), op->getPropertiesStorage(), op->getRegions(),
        inferredShapes);
    if (result.failed()) {
      Fatal("Cannot infer type for operator `{0}`.\n",
            op->getName().getStringRef().str().c_str());
    }

    // Assign inferred types to output tensors
    for (auto [value, type] : zip(op->getResults(), inferredShapes)) {
      value.setType(mlir::RankedTensorType::get(
          type.getDims(),
          value.getType().cast<mlir::TensorType>().getElementType()));
    }
  });

  // Update return type of function
  auto funcType = mlir::FunctionType::get(
      &getContext(), func.getArgumentTypes(), mlir::TypeRange(retValues));
  func.setType(funcType);
}

std::unique_ptr<mlir::Pass> createShapeInference() {
  return std::make_unique<ShapeInference>();
}

} // namespace relay
