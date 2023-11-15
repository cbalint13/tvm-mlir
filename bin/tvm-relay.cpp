#include <llvm/Support/Path.h>

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/ExecutionEngine/ExecutionEngine.h>
#include <mlir/ExecutionEngine/OptUtils.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

#include "tvm-mlir/Conversion/Passes.hpp"
#include "tvm-mlir/Dialect/Relay/Passes.hpp"
#include "tvm-mlir/Dialect/Relay/RelayDialect.hpp"
#include "tvm-mlir/Dialect/Relay/RelayOps.hpp"
#include "tvm-mlir/Frontend/RelayImporter.hpp"
#include "tvm-mlir/Support/Common.hpp"

int main(int argc, char const *argv[]) {
  auto inputPath = llvm::cl::opt<std::string>(llvm::cl::Positional);
  auto outputDir = llvm::cl::opt<std::string>(llvm::cl::Positional);

  /// Parse command line arguments
  llvm::cl::ParseCommandLineOptions(argc, argv, "tvm-relay MLIR Compiler");

  /// Initialize MLIR context
  mlir::MLIRContext mlirCtx;
  mlirCtx
      .loadDialect<relay::RelayDialect, mlir::func::FuncDialect,
                   mlir::memref::MemRefDialect, mlir::affine::AffineDialect>();
  mlirCtx.disableMultithreading();

  /// Configure pass manager
  mlir::PassManager pm(&mlirCtx, mlir::ModuleOp::getOperationName(),
                       mlir::PassManager::Nesting::Implicit);
  pm.addPass(relay::createShapeInference());
  pm.addPass(relay::createOpFusion());
  pm.addPass(createRelayToAffine());

  /// Parse Relay source
  auto fileOrErr = llvm::MemoryBuffer::getFile(inputPath, true);
  if (auto err = fileOrErr.getError()) {
    Fatal("Cannot open file {0}: {1}", inputPath, err.message());
  }
  auto buffer = fileOrErr->get()->getBuffer().str();
  auto irmod = tvm::IRModule::FromText(buffer, inputPath);

  std::cout << "TVM Relay:\n" << irmod << std::endl;

  /// Import and compile
  auto mod = relay::ImportRelay(irmod, inputPath, &mlirCtx);
  if (pm.run(mod).failed())
    Fatal("Failed to run passes on module.");
  mod.dump();
}
