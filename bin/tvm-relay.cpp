#include <llvm/ExecutionEngine/Orc/ExecutionUtils.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/TargetSelect.h>

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/ExecutionEngine/ExecutionEngine.h>
#include <mlir/ExecutionEngine/OptUtils.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Dialect/OpenMP/OpenMPToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Export.h>
#include <mlir/Transforms/Passes.h>

#include "tvm-mlir/Conversion/Passes.hpp"
#include "tvm-mlir/Dialect/Relay/Passes.hpp"
#include "tvm-mlir/Dialect/Relay/RelayDialect.hpp"
#include "tvm-mlir/Dialect/Relay/RelayOps.hpp"
#include "tvm-mlir/Frontend/RelayImporter.hpp"
#include "tvm-mlir/Support/Common.hpp"

namespace cl = llvm::cl;
namespace path = llvm::sys::path;
namespace fs = llvm::sys::fs;

int main(int argc, char const *argv[]) {
  auto inputPath = cl::opt<std::string>(cl::Positional);
  auto outputDir = cl::opt<std::string>(cl::Positional);

  /// Parse command line arguments
  llvm::cl::ParseCommandLineOptions(argc, argv, "tvm-relay MLIR Compiler");

  /// Initialize MLIR context
  mlir::MLIRContext mlirCtx;
  mlirCtx.loadDialect<relay::RelayDialect, mlir::func::FuncDialect,
                      mlir::cf::ControlFlowDialect, mlir::memref::MemRefDialect,
                      mlir::affine::AffineDialect, mlir::scf::SCFDialect,
                      mlir::LLVM::LLVMDialect>();
  mlirCtx.disableMultithreading();

  /// Configure pass manager
  mlir::PassManager pm(&mlirCtx, mlir::ModuleOp::getOperationName(),
                       mlir::PassManager::Nesting::Implicit);
  pm.addPass(relay::createShapeInference());
  pm.addPass(relay::createOpFusion());
  pm.addPass(createRelayToAffine());
  pm.addPass(createAffineToLLVM());

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

  // Create output filename
  auto inputFilename = path::filename(inputPath);
  llvm::SmallVector<char> outFilename(inputFilename.begin(),
                                      inputFilename.end());
  path::replace_extension(outFilename, "mlir");
  llvm::SmallVector<char> outPathBuf(outputDir.begin(), outputDir.end());
  path::append(outPathBuf, outFilename);
  // Write MLIR to output file
  {
    llvm::StringRef outputPath(outPathBuf.data(), outPathBuf.size());
    std::error_code err;
    llvm::raw_fd_ostream outStream(outputPath, err);
    if (err)
      Fatal("Cannot write to file {}: {}", outputPath, err.message());
    mod.print(outStream);
  }

  // Export to LLVM
  mlir::registerBuiltinDialectTranslation(mlirCtx);
  mlir::registerLLVMDialectTranslation(mlirCtx);
  llvm::LLVMContext llvmCtx;
  auto llvmMod = translateModuleToLLVMIR(mod, llvmCtx, inputPath);
  if (!llvmMod)
    Fatal("Failed to emit LLVM IR");
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  auto optPpl = mlir::makeOptimizingTransformer(3, 0, nullptr);
  if (auto err = optPpl(llvmMod.get()))
    Fatal("Failed to optimize LLVM IR");

  // Write LLVM IR to file
  path::replace_extension(outPathBuf, "ll");
  {
    llvm::StringRef outputPath(outPathBuf.data(), outPathBuf.size());
    std::error_code err;
    llvm::raw_fd_ostream outStream(outputPath, err);
    if (err)
      Fatal("Cannot write to file {}: {}", outputPath, err.message());
    llvmMod->print(outStream, nullptr);
  }

  // Set up LLVM JIT
  mlir::ExecutionEngineOptions engineOpts{.transformer = optPpl};
  auto expectEngine = mlir::ExecutionEngine::create(mod, engineOpts);
  if (!expectEngine)
    Fatal("Cannot create execution engine");
  auto &engine = expectEngine.get();
  auto expectPacked = engine->lookupPacked("main");
  if (!expectPacked)
    Fatal("Cannot find main function");

  return 0;
}
