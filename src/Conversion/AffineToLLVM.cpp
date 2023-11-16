#include <mlir/Conversion/AffineToStandard/AffineToStandard.h>
#include <mlir/Conversion/ArithToLLVM/ArithToLLVM.h>
#include <mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h>
#include <mlir/Conversion/LLVMCommon/ConversionTarget.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h>
#include <mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Arith/Transforms/Passes.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>

#include "tvm-mlir/Conversion/PassDetail.hpp"
#include "tvm-mlir/Conversion/Passes.hpp"

class AffineToLLVM : public AffineToLLVMBase<AffineToLLVM> {
  void runOnOperation() override;
};

void AffineToLLVM::runOnOperation() {
  // Define conversion target
  mlir::LLVMConversionTarget target(getContext());
  target.addLegalOp<mlir::ModuleOp>();
  mlir::LLVMTypeConverter converter(&getContext());

  // Populate conversion patterns
  mlir::RewritePatternSet patterns(&getContext());
  mlir::populateAffineToStdConversionPatterns(patterns);
  mlir::populateSCFToControlFlowConversionPatterns(patterns);
  mlir::arith::populateArithToLLVMConversionPatterns(converter, patterns);
  mlir::arith::populateArithExpandOpsPatterns(patterns);
  mlir::populateFinalizeMemRefToLLVMConversionPatterns(converter, patterns);

  mlir::cf::populateControlFlowToLLVMConversionPatterns(converter, patterns);
  mlir::populateFuncToLLVMConversionPatterns(converter, patterns);

  // Completely lower to LLVM IR
  if (mlir::applyFullConversion(getOperation(), target, std::move(patterns))
          .failed())
    mlir::Pass::signalPassFailure();
}

std::unique_ptr<mlir::Pass> createAffineToLLVM() {
  return std::make_unique<AffineToLLVM>();
}
