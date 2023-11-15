#include <mlir/IR/OpImplementation.h>

#include "tvm-mlir/Dialect/Relay/RelayDialect.hpp"
#include "tvm-mlir/Dialect/Relay/RelayOps.hpp"
#include "tvm-mlir/Support/Common.hpp"

#define GET_OP_CLASSES
#include "tvm-mlir/Dialect/Relay/RelayOps.cpp.inc"

namespace relay {

mlir::LogicalResult ConstantOp::inferReturnTypeComponents(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    mlir::ValueShapeRange operands, mlir::DictionaryAttr attributes,
    mlir::OpaqueProperties properties, mlir::RegionRange regions,
    llvm::SmallVectorImpl<mlir::ShapedTypeComponents> &inferredReturnShapes) {
  mlir::DictionaryAttr dict = mlir::dyn_cast<mlir::DictionaryAttr>(attributes);
  auto attr = dict.getAs<mlir::StringAttr>("value");
  auto type = attr.getType().cast<mlir::TensorType>();
  inferredReturnShapes.push_back({type.getShape()});
  return mlir::success();
}

mlir::LogicalResult ReLUOp::inferReturnTypeComponents(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    mlir::ValueShapeRange operands, mlir::DictionaryAttr attributes,
    mlir::OpaqueProperties properties, mlir::RegionRange regions,
    llvm::SmallVectorImpl<mlir::ShapedTypeComponents> &inferredReturnShapes) {
  inferredReturnShapes.push_back({operands.getShape(0)});
  return mlir::success();
}

mlir::LogicalResult DenseOp::inferReturnTypeComponents(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    mlir::ValueShapeRange operands, mlir::DictionaryAttr attributes,
    mlir::OpaqueProperties properties, mlir::RegionRange regions,
    llvm::SmallVectorImpl<mlir::ShapedTypeComponents> &inferredReturnShapes) {
  llvm::SmallVector<int64_t> dataShape, weightShape;

  operands.getShape(0).getDims(dataShape);
  operands.getShape(1).getDims(weightShape);

  if (dataShape.size() != 2) {
    Error("Expect rank 2 for data tensor, got {0}.", dataShape.size());
    return mlir::failure();
  }
  if (weightShape.size() != 2) {
    Error("Expect rank 2 for weight tensor, got {0}.", weightShape.size());
    return mlir::failure();
  }
  if (dataShape[1] != weightShape[1]) {
    Error("Expect data.shape[1] == weight.shape[1], got {0} != {1}.",
          dataShape[1], weightShape[1]);
    return mlir::failure();
  }

  inferredReturnShapes.push_back(
      llvm::ArrayRef({dataShape[0], weightShape[0]}));

  return mlir::success();
}

mlir::LogicalResult BiasAddOp::inferReturnTypeComponents(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    mlir::ValueShapeRange operands, mlir::DictionaryAttr attributes,
    mlir::OpaqueProperties properties, mlir::RegionRange regions,
    llvm::SmallVectorImpl<mlir::ShapedTypeComponents> &inferredReturnShapes) {
  llvm::SmallVector<int64_t> dataShape, biasShape;
  operands.getShape(0).getDims(dataShape);
  operands.getShape(1).getDims(biasShape);
  if (biasShape.size() != 1) {
    Error("Expect rank 1 for bias tensor, got {0}.", biasShape.size());
    return mlir::failure();
  }
  auto dataRank = int64_t(dataShape.size());
  auto axis = attributes.get("axis").cast<mlir::IntegerAttr>().getSInt();
  if (axis < -dataRank || axis >= dataRank) {
    Error("Expect axis in range [{}, {}), got {0}.", -dataRank, dataRank, axis);
    return mlir::failure();
  }
  if (axis < 0)
    axis += dataRank;
  if (dataShape[axis] != biasShape[0]) {
    Error("Expect data.shape[axis] == bias.shape[0], got {0} != {1}.",
          dataShape[axis], biasShape[0]);
    return mlir::failure();
  }
  inferredReturnShapes.push_back({dataShape});
  return mlir::success();
}

} // namespace relay
