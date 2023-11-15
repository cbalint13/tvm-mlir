#include <mlir/IR/IRMapping.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Rewrite/FrozenRewritePatternSet.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <unordered_map>

#include "tvm-mlir/Dialect/Relay/Passes.hpp"
#include "tvm-mlir/Relay/PassDetail.hpp"
#include "tvm-mlir/Support/Common.hpp"

namespace relay {

static mlir::LogicalResult
fuseDenseBiasAddRelu(mlir::Operation *pivot,
                     llvm::SmallVector<mlir::Operation *> *group) {
  // Match dense
  if (pivot->getName().getStringRef() != "relay.nn.dense")
    return mlir::failure();

  // Match bias_add
  auto denseResult = pivot->getResult(0);
  if (!denseResult.hasOneUse())
    return mlir::failure();

  auto biasAdd = llvm::to_vector(denseResult.getUsers())[0];
  if (biasAdd->getName().getStringRef() != "relay.nn.bias_add")
    return mlir::failure();
  group->push_back(pivot);
  group->push_back(biasAdd);

  // Optionally match relu
  auto biasAddResult = biasAdd->getResult(0);
  if (!biasAddResult.hasOneUse())
    return mlir::success();
  auto relu = llvm::to_vector(biasAddResult.getUsers())[0];
  if (relu->getName().getStringRef() != "relay.nn.relu")
    return mlir::success();
  group->push_back(relu);

  return mlir::success();
}

using MatchFn = mlir::LogicalResult (*)(mlir::Operation *,
                                        llvm::SmallVector<mlir::Operation *> *);
static MatchFn matchFuncs[] = {fuseDenseBiasAddRelu};

struct FusionGroup {
  llvm::SmallVector<mlir::Operation *> ops;
  llvm::SmallVector<mlir::Operation *> outputs;
};

class OpFusionPattern : public mlir::RewritePattern {
public:
  OpFusionPattern(mlir::MLIRContext *ctx,
                  const std::vector<FusionGroup> &groups,
                  const std::unordered_map<mlir::Operation *, size_t> &opGrpIdx)
      : RewritePattern(Pattern::MatchAnyOpTypeTag(), 1, ctx), groups(groups),
        opGrpIdx(opGrpIdx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *root,
                  mlir::PatternRewriter &rewriter) const override;

private:
  mutable size_t nGroup = 0;
  const std::vector<FusionGroup> &groups;
  const std::unordered_map<mlir::Operation *, size_t> &opGrpIdx;
};

mlir::LogicalResult
OpFusionPattern::matchAndRewrite(mlir::Operation *root,
                                 mlir::PatternRewriter &rewriter) const {
  // Find out the group
  if (root->getDialect()->getNamespace() != RelayDialect::getDialectNamespace())
    return mlir::failure();
  if (mlir::cast<mlir::func::FuncOp>(root->getParentOp()).getName() != "main")
    return mlir::failure();
  auto &group = groups[opGrpIdx.at(root)];

  // Only rewrite at outputs
  if (!llvm::is_contained(group.outputs, root))
    return mlir::failure();

  // Find all arguments of the new function
  mlir::DenseSet<mlir::Operation *> opSet(group.ops.begin(), group.ops.end());
  llvm::SmallVector<mlir::Value> args;
  for (auto op : group.ops)
    for (auto arg : op->getOperands())
      if (!opSet.contains(arg.getDefiningOp()))
        args.push_back(arg);

  // Find all results of the new function
  llvm::SmallVector<mlir::Value> results;
  for (auto outOp : group.outputs) {
    auto opResults = outOp->getResults();
    results.append(opResults.begin(), opResults.end());
  }

  // Create prototype of the function
  auto inTypes = llvm::to_vector(
      llvm::map_range(args, [](mlir::Value in) { return in.getType(); }));
  auto outTypes = llvm::to_vector(
      llvm::map_range(results, [](mlir::Value out) { return out.getType(); }));
  auto funcType = rewriter.getFunctionType(inTypes, outTypes);
  llvm::StringRef funcName = "fused_" + std::to_string(nGroup++);
  rewriter.setInsertionPointToEnd(root->getParentOp()->getBlock());
  auto func =
      rewriter.create<mlir::func::FuncOp>(root->getLoc(), funcName, funcType);
  func->setAttr("primitive", rewriter.getBoolAttr(true));

  // Create function body
  auto block = func.addEntryBlock();
  mlir::IRMapping mapper;
  for (auto [arg, param] : llvm::zip(args, block->getArguments()))
    mapper.map(arg, param);
  rewriter.setInsertionPointToStart(block);
  llvm::SmallVector<mlir::Operation *> funcOps;
  for (auto op : group.ops) {
    auto clonedOp = rewriter.clone(*op, mapper);
    if (llvm::is_contained(group.outputs, op))
      funcOps.push_back(clonedOp);
  }
  llvm::SmallVector<mlir::Value> funcResults;
  for (auto outOp : funcOps) {
    auto opResults = outOp->getResults();
    funcResults.append(opResults.begin(), opResults.end());
  }
  rewriter.create<mlir::func::ReturnOp>(root->getLoc(), funcResults);

  // Replace group with function call
  rewriter.setInsertionPointAfter(root);
  auto funcCall = rewriter.create<mlir::func::CallOp>(
      root->getLoc(), mlir::FlatSymbolRefAttr::get(func), outTypes, args);

  // Replace uses of group outputs
  auto resultIter = funcCall.getResults().begin();
  for (auto op : group.outputs) {
    llvm::SmallVector<mlir::Value> newValues(resultIter,
                                             resultIter + op->getNumResults());
    rewriter.replaceOp(op, newValues);
    resultIter += op->getNumResults();
  }

  return mlir::success();
}

class OpFusion : public OpFusionBase<OpFusion> {
  void runOnOperation() override;
};

void OpFusion::runOnOperation() {
  // Get main function
  auto ctx = &getContext();
  auto mainFn = getOperation();
  if (mainFn.getName() != "main")
    return;

  // Match and fuse patterns
  llvm::SmallVector<mlir::Operation *> groupOps;
  std::vector<FusionGroup> groups;
  std::unordered_map<mlir::Operation *, size_t> opGrpIdx;

  mainFn.walk([&](mlir::Operation *pivot) {
    // Skip operations not interested
    if (pivot->getDialect()->getNamespace() != "relay")
      return;
    if (opGrpIdx.count(pivot))
      return;

    for (auto matcher : matchFuncs) {
      // Match with predefined functions
      if (matcher(pivot, &groupOps).failed())
        continue;
      assert(!groupOps.empty());

      // Create fusion group
      FusionGroup group;
      group.ops.swap(groupOps);
      mlir::DenseSet<mlir::Operation *> opSet(group.ops.begin(),
                                              group.ops.end());
      for (auto op : group.ops) {
        auto isOut = false;
        for (auto result : op->getResults())
          for (auto user : result.getUsers())
            if (!opSet.contains(user))
              isOut = true;
        if (isOut)
          group.outputs.push_back(op);
      }
      auto grpIdx = groups.size();
      for (auto op : group.ops)
        opGrpIdx.insert({op, grpIdx});
      groups.push_back(std::move(group));
      return;
    }

    // Create single-operation group
    opGrpIdx.insert({pivot, groups.size()});
    groups.push_back({.ops = {pivot}, .outputs = {pivot}});
  });

  // Create nested function for each group
  mlir::RewritePatternSet patterns(ctx);
  patterns.add<OpFusionPattern>(ctx, groups, opGrpIdx);
  mlir::FrozenRewritePatternSet frozenPat(std::move(patterns));
  mlir::GreedyRewriteConfig config{.useTopDownTraversal = true};
  if (mlir::applyPatternsAndFoldGreedily(mainFn, frozenPat, std::move(config))
          .failed())
    mlir::Pass::signalPassFailure();
}

std::unique_ptr<mlir::Pass> createOpFusion() {
  return std::make_unique<OpFusion>();
}

} // namespace relay
