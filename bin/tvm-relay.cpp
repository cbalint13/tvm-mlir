#include <llvm/IR/LLVMContext.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/TargetSelect.h>

#include <mlir/Conversion/SCFToOpenMP/SCFToOpenMP.h>
#include <mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h>
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
#include "tvm-mlir/Support/MemRef.hpp"
#include "tvm-mlir/Transforms/Passes.hpp"

namespace cl = llvm::cl;
namespace path = llvm::sys::path;
namespace fs = llvm::sys::fs;

static bool shouldPrint(mlir::Pass *pass, mlir::Operation *op) {
    return mlir::isa<mlir::ModuleOp>(op) || mlir::cast<mlir::func::FuncOp>(op).getSymName() == "main";
}

int main(int argc, char const *argv[]) {
  /// File paths
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
  pm.addPass(createOptimizeAffine());
  pm.addPass(createAffineToSCF());
  pm.addPass(mlir::createConvertSCFToOpenMPPass());
  pm.addPass(createSCFToLLVM());

  /// Parse Relay source
  auto fileOrErr = llvm::MemoryBuffer::getFile(inputPath, true);
  if (auto err = fileOrErr.getError()) {
    Fatal("Cannot open file {0}: {1}", inputPath, err.message());
  }
  auto buffer = fileOrErr->get()->getBuffer().str();
  auto irmod = tvm::IRModule::FromText(buffer, inputPath);

  std::cout << "TVM Relay:\n" << irmod << std::endl;

  // Create file to dump IR
  auto inputFilename = path::filename(inputPath);
  llvm::SmallVector<char> outFilename(inputFilename.begin(),
                                      inputFilename.end());
  path::replace_extension(outFilename, "mlir");
  llvm::SmallVector<char> outPathBuf(outputDir.begin(), outputDir.end());
  path::append(outPathBuf, outFilename);
    llvm::StringRef outputPath(outPathBuf.data(), outPathBuf.size());

    std::error_code err;
    llvm::raw_fd_ostream outStream(outputPath, err);
    if (err) Fatal("Cannot write to file {0}: {1}", outputPath, err.message());
    pm.enableIRPrinting([](mlir::Pass *, mlir::Operation *) { return false; }, shouldPrint,
                        true, false, false, outStream);

    // Import and compile to MLIR
    auto mod = relay::ImportRelay(irmod, inputPath, &mlirCtx);
    if (pm.run(mod).failed()) { Fatal("Failed to run passes."); }

    // End print stream
    outStream.close();

    // DEBUG
    mod.dump();
    exit(0);

  // Export to LLVM
  mlir::registerBuiltinDialectTranslation(mlirCtx);
  mlir::registerOpenMPDialectTranslation(mlirCtx);
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
      Fatal("Cannot write to file {0}: {1}", outputPath, err.message());
    llvm::outs() << "Write LLVM IR to: " << outputPath << "\n";
    llvmMod->print(outStream, nullptr);
    outStream.close();
  }

    // Set up LLVM JIT
    mlir::ExecutionEngineOptions engineOpts{.transformer = optPpl};
    auto expectEngine = mlir::ExecutionEngine::create(mod, engineOpts);
    if (!expectEngine) Fatal("Cannot create execution engine");
    auto &engine = expectEngine.get();
    auto expectPacked = engine->lookupPacked("main");
    if (!expectPacked) Fatal("Cannot find main function");

    exit(0);

    llvm::outs() << "Run LLVM JIT\n";

    // Run JIT
    float data[6]{1.f, 0.f, -1.f, 0.f, 2.f, -2.f};
    auto bufType = mlir::MemRefType::get({2, 3}, mlir::Float32Type::get(&mlirCtx));
    MemRef inBuf(bufType), outBuf(bufType);
    inBuf.LoadData(data);
    llvm::SmallVector<void *> jitArgs;
    inBuf.PopulateLLJITArgs(&jitArgs);
    outBuf.PopulateLLJITArgs(&jitArgs);
    if (auto err = engine->invokePacked("main", jitArgs))
        Fatal("Cannot invoke main function");
    auto result = outBuf.GetDataAs<float>();
    std::vector<float> resultVec(result, result + 6);

  return 0;
}
