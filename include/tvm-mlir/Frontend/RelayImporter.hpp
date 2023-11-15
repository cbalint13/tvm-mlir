#pragma once
#include <mlir/IR/BuiltinOps.h>

#include <tvm/ir/module.h>

#include <string>

namespace relay {

mlir::ModuleOp ImportRelay(tvm::IRModule mod, const std::string &srcPath,
                           mlir::MLIRContext *ctx);

} // namespace relay
