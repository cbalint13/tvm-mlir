#pragma once
#include <mlir/IR/OpDefinition.h>

#include <tvm/ir/attrs.h>

#include <vector>

namespace relay {

mlir::OpState ConvertRelayOp(const tvm::String &name,
                             const std::vector<mlir::Value> &args,
                             const tvm::Attrs &attrs, mlir::Location loc,
                             mlir::OpBuilder *builder);
}
