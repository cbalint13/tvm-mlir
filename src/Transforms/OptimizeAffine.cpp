#include <mlir/Dialect/Affine/Analysis/Utils.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Affine/LoopFusionUtils.h>
#include <mlir/Dialect/Affine/LoopUtils.h>
#include <mlir/Dialect/Affine/Utils.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/Dominance.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include "tvm-mlir/Transforms/PassDetail.hpp"

namespace func = mlir::func;
namespace affine = mlir::affine;

class OptimizeAffine : public OptimizeAffineBase<OptimizeAffine> {
  void runOnOperation() override;
};

struct FuseLoop : public mlir::OpRewritePattern<func::FuncOp> {
  explicit FuseLoop(mlir::MLIRContext *ctx) : OpRewritePattern(ctx) {}

  mlir::LogicalResult
  matchAndRewrite(func::FuncOp fn,
                  mlir::PatternRewriter &rewriter) const override;
};

inline static uint32_t getPerfectlyNestedDepth(affine::AffineForOp root) {
  llvm::SmallVector<affine::AffineForOp> loops;
  getPerfectlyNestedLoops(loops, root);
  return loops.size();
}

mlir::LogicalResult
FuseLoop::matchAndRewrite(func::FuncOp fn,
                          mlir::PatternRewriter &rewriter) const {
  // Skip non-primitive function
  if (!fn->hasAttrOfType<mlir::BoolAttr>("primitive"))
    return mlir::failure();

  // Collect all allocation and deallocation operations
  llvm::SmallVector<mlir::memref::AllocOp> allocOps;
  llvm::SmallVector<mlir::memref::DeallocOp> deallocOps;
  for (auto &op : fn.getOps()) {
    if (mlir::isa<mlir::memref::AllocOp>(&op))
      allocOps.push_back(mlir::cast<mlir::memref::AllocOp>(&op));
    else if (mlir::isa<mlir::memref::DeallocOp>(&op))
      deallocOps.push_back(mlir::cast<mlir::memref::DeallocOp>(&op));
  }

  // Create a new function
  auto new_fn = rewriter.create<func::FuncOp>(
      fn.getLoc(), fn.getName(), fn.getFunctionType(), fn->getAttrs());
  rewriter.setInsertionPointToStart(new_fn.addEntryBlock());

  // Create value mapping
  mlir::IRMapping mapper;
  for (auto [prevArg, newArg] :
       llvm::zip(fn.getArguments(), new_fn.getArguments()))
    mapper.map(prevArg, newArg);

  // Reorder operations
  for (auto op : allocOps)
    rewriter.clone(*op, mapper); // allocation first
  llvm::SmallVector<affine::AffineForOp> forOps;
  for (auto &op : fn.getOps()) {
    if (mlir::isa<mlir::memref::AllocOp, mlir::memref::DeallocOp>(&op))
      continue;
    auto newOp = rewriter.clone(op, mapper);
    if (mlir::isa<affine::AffineForOp>(newOp))
      forOps.push_back(mlir::cast<affine::AffineForOp>(newOp));
  } // then other operations
  rewriter.setInsertionPoint(rewriter.getBlock()->getTerminator());
  for (auto op : deallocOps)
    rewriter.clone(*op, mapper); // deallocation finally

  // Erase previous function
  rewriter.eraseOp(fn);

  // Try fuse two consequent loops
  auto forIdx = 0u;
  while (forIdx + 1 < forOps.size()) {
    auto dstFor = forOps[forIdx], srcFor = forOps[forIdx + 1];
    affine::ComputationSliceState srcSlice;
    auto dstDepth = getPerfectlyNestedDepth(dstFor);
    auto fuseResult = canFuseLoops(srcFor, dstFor, dstDepth, &srcSlice);
    if (fuseResult.value != affine::FusionResult::Success) {
      forIdx++;
      continue;
    }
    fuseLoops(srcFor, dstFor, srcSlice, false);
    rewriter.eraseOp(srcFor);
    forOps.erase(forOps.begin() + forIdx + 1);
  }

  return mlir::success();
}

struct ParallelizeLoop : public mlir::OpRewritePattern<affine::AffineForOp> {
  explicit ParallelizeLoop(mlir::MLIRContext *ctx) : OpRewritePattern(ctx) {}

  mlir::LogicalResult
  matchAndRewrite(affine::AffineForOp forOp,
                  mlir::PatternRewriter &rewriter) const override;
};

mlir::LogicalResult
ParallelizeLoop::matchAndRewrite(affine::AffineForOp root,
                                 mlir::PatternRewriter &rewriter) const {
  // Skip if this loop is nested in another loop
  if (!mlir::isa<func::FuncOp>(root->getParentOp()))
    return mlir::failure();

  // Get all perfectly-nested, non-reduction loops
  mlir::SmallVector<affine::AffineForOp> nestedLoops;
  getPerfectlyNestedLoops(nestedLoops, root);
  if (llvm::any_of(nestedLoops, [](affine::AffineForOp op) {
        return op.getNumIterOperands() != 0;
      }))
    return mlir::failure();

  // Initialize parallel operation
  auto lbMaps = llvm::to_vector(llvm::map_range(
      nestedLoops, std::mem_fn(&affine::AffineForOp::getLowerBoundMap)));
  auto ubMaps = llvm::to_vector(llvm::map_range(
      nestedLoops, std::mem_fn(&affine::AffineForOp::getUpperBoundMap)));
  llvm::SmallVector<mlir::Value> lbArgs, ubArgs;
  for (auto forOp : nestedLoops) {
    auto lbs = forOp.getLowerBoundOperands(),
         ubs = forOp.getUpperBoundOperands();
    lbArgs.append(lbs.begin(), lbs.end());
    ubArgs.append(ubs.begin(), ubs.end());
  }
  auto steps = llvm::to_vector(
      llvm::map_range(nestedLoops, std::mem_fn(&affine::AffineForOp::getStep)));
  auto parOp = rewriter.create<affine::AffineParallelOp>(
      root.getLoc(), std::nullopt, std::nullopt, lbMaps, lbArgs, ubMaps, ubArgs,
      steps);

  // Clone body from innermost loop
  mlir::IRMapping mapper;
  auto forIvs = llvm::map_range(
      nestedLoops, std::mem_fn(&affine::AffineForOp::getInductionVar));
  for (auto [forIv, parIv] : llvm::zip(forIvs, parOp.getBody()->getArguments()))
    mapper.map(forIv, parIv);
  auto innerForOp = nestedLoops.back();
  rewriter.setInsertionPointToStart(parOp.getBody());
  for (auto &op : *innerForOp.getBody()) {
    if (&op != innerForOp.getBody()->getTerminator())
      rewriter.clone(op, mapper);
  }
  rewriter.replaceOp(root, parOp.getResults());

  return mlir::success();
}

void OptimizeAffine::runOnOperation() {
  // Get module and context
  auto mod = getOperation();
  auto ctx = &getContext();

  // Fuse consequent loops
  {
    mlir::RewritePatternSet patterns(ctx);
    patterns.add<FuseLoop>(ctx);

    auto funcs = llvm::to_vector(
        llvm::map_range(mod.getOps(), [](mlir::Operation &op) { return &op; }));

    bool changed = false;
    bool allErased = false;
    mlir::GreedyRewriteConfig config;
    config.strictMode = mlir::GreedyRewriteStrictness::ExistingOps;
    // config.maxIterations = 8;
    if (mlir::applyOpPatternsAndFold(funcs, std::move(patterns), config,
                                     &changed, &allErased)
            .failed())
      mlir::Pass::signalPassFailure();
  }

  // Eliminate redundant load/store
  auto &domInfo = getAnalysis<mlir::DominanceInfo>();
  auto &postDomInfo = getAnalysis<mlir::PostDominanceInfo>();
  for (auto &op : mod.getOps())
    affine::affineScalarReplace(mlir::cast<func::FuncOp>(&op), domInfo,
                                postDomInfo);

  // Parallelize loops
  {
    mlir::RewritePatternSet patterns(ctx);
    patterns.add<ParallelizeLoop>(ctx);
    mlir::GreedyRewriteConfig config{.useTopDownTraversal = true};
    if (mlir::applyPatternsAndFoldGreedily(mod, std::move(patterns),
                                           std::move(config))
            .failed())
      mlir::Pass::signalPassFailure();
  }
}

std::unique_ptr<mlir::Pass> createOptimizeAffine() {
  return std::make_unique<OptimizeAffine>();
}
