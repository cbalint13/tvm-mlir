#include <mlir/Conversion/AffineToStandard/AffineToStandard.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Transforms/DialectConversion.h>

#include "tvm-mlir/Conversion/PassDetail.hpp"

class AffineToSCF : public AffineToSCFBase<AffineToSCF> {
  void runOnOperation() override;
};

void AffineToSCF::runOnOperation() {
  mlir::ConversionTarget target(getContext());
  target.addLegalDialect<mlir::arith::ArithDialect, mlir::BuiltinDialect,
                         mlir::memref::MemRefDialect, mlir::scf::SCFDialect>();
  target.addIllegalDialect<mlir::affine::AffineDialect>();
  mlir::RewritePatternSet patterns(&getContext());
  mlir::populateAffineToStdConversionPatterns(patterns);
  if (applyPartialConversion(getOperation(), target, std::move(patterns))
          .failed())
    mlir::Pass::signalPassFailure();
}

std::unique_ptr<mlir::Pass> createAffineToSCF() {
  return std::make_unique<AffineToSCF>();
}
