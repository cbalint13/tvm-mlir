#include <mlir/Conversion/ArithToLLVM/ArithToLLVM.h>
#include <mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h>
#include <mlir/Conversion/LLVMCommon/ConversionTarget.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h>
#include <mlir/Conversion/OpenMPToLLVM/ConvertOpenMPToLLVM.h>
#include <mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Arith/Transforms/Passes.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/OpenMP/OpenMPDialect.h>
#include <mlir/Dialect/SCF/IR/SCF.h>

#include "tvm-mlir/Conversion/PassDetail.hpp"

class SCFToLLVM : public SCFToLLVMBase<SCFToLLVM> {
  void runOnOperation() override;
};

void SCFToLLVM::runOnOperation() {
  // Define conversion target
  mlir::LLVMConversionTarget target(getContext());
  target.addLegalOp<mlir::ModuleOp>();
  mlir::LLVMTypeConverter converter(&getContext());
  mlir::configureOpenMPToLLVMConversionLegality(target, converter);
  target.addLegalOp<mlir::scf::YieldOp, mlir::omp::YieldOp,
                    mlir::omp::TerminatorOp>();

  // Populate conversion patterns
  mlir::RewritePatternSet patterns(&getContext());
  mlir::populateSCFToControlFlowConversionPatterns(patterns);
  mlir::populateOpenMPToLLVMConversionPatterns(converter, patterns);
  mlir::arith::populateArithToLLVMConversionPatterns(converter, patterns);
  mlir::arith::populateArithExpandOpsPatterns(patterns);
  mlir::populateFinalizeMemRefToLLVMConversionPatterns(converter, patterns);

  mlir::cf::populateControlFlowToLLVMConversionPatterns(converter, patterns);
  mlir::populateFuncToLLVMConversionPatterns(converter, patterns);

  // Completely lower to LLVM dialect
  if (mlir::applyFullConversion(getOperation(), target, std::move(patterns))
          .failed())
    mlir::Pass::signalPassFailure();
}

std::unique_ptr<mlir::Pass> createSCFToLLVM() {
  return std::make_unique<SCFToLLVM>();
}
