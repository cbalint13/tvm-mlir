#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/Transforms/DialectConversion.h>

#include "tvm-mlir/Conversion/PassDetail.hpp"
#include "tvm-mlir/Dialect/Relay/RelayOps.hpp"
#include "tvm-mlir/Support/Common.hpp"

namespace func = mlir::func;
namespace affine = mlir::affine;

inline static mlir::Type cvtTensorToMemref(mlir::Type type) {
  auto tt = type.cast<mlir::TensorType>();
  return mlir::MemRefType::get(tt.getShape(), tt.getElementType());
}

template <class Op> struct LowerOp : public mlir::OpRewritePattern<Op> {
  explicit LowerOp(mlir::MLIRContext *ctx) : mlir::OpRewritePattern<Op>(ctx) {}

  mlir::LogicalResult
  matchAndRewrite(Op op, mlir::PatternRewriter &rewriter) const override;

  virtual mlir::LogicalResult lower(Op op, mlir::ValueRange buffers,
                                    mlir::PatternRewriter *rewriter) const = 0;
};

template <class Op>
mlir::LogicalResult
LowerOp<Op>::matchAndRewrite(Op op, mlir::PatternRewriter &rewriter) const {
  // Find buffers for function outputs
  llvm::SmallVector<mlir::Value> newResults;
  bool isFuncRet = false;

  for (auto result : op->getResults()) {
    // Check if the result of this operation is returned by the parent
    // function
    mlir::Value newResult;
    func::ReturnOp retOp;
    int64_t retIdx = -1;
    for (auto &use : result.getUses()) {
      auto owner = use.getOwner();
      if (mlir::isa<func::ReturnOp>(owner)) {
        retOp = mlir::cast<func::ReturnOp>(owner);
        retIdx = use.getOperandNumber();
        isFuncRet = true;
      }
    }

    // Collect result buffer or allocate a new one
    if (retOp) {
      auto func = mlir::cast<func::FuncOp>(op->getParentOp());
      auto numInputs =
          func->template getAttrOfType<mlir::IntegerAttr>("num_inputs")
              .getInt();
      newResult = func.getArgument(numInputs + retIdx);
    } else {
      auto alloc = rewriter.create<mlir::memref::AllocaOp>(
          op.getLoc(), result.getType().template cast<mlir::MemRefType>());
      newResult = alloc.getResult();
    }
    newResults.push_back(newResult);
  }

  auto buffers = llvm::to_vector(op->getOperands());
  buffers.append(newResults);

  // Lower operation with given buffers
  if (this->lower(op, buffers, &rewriter).failed())
    return mlir::failure();

  // Erase or replace previous operations
  if (!isFuncRet)
    rewriter.replaceOp(op, newResults);

  return mlir::success();
}

#define FOR(iv, low, high, body)                                               \
  auto iv##Loop =                                                              \
      rewriter->create<affine::AffineForOp>(op.getLoc(), low, high);           \
  rewriter->setInsertionPointToStart(iv##Loop.getBody());                      \
  {                                                                            \
    auto iv = iv##Loop.getInductionVar();                                      \
    body                                                                       \
  }                                                                            \
  rewriter->setInsertionPoint(iv##Loop.getBody()->getTerminator());

#define LOAD(buffer, indices)                                                  \
  rewriter->create<affine::AffineLoadOp>(op.getLoc(), buffer, indices)         \
      .getResult()

#define STORE(value, buffer, indices)                                          \
  rewriter->create<affine::AffineStoreOp>(op.getLoc(), value, buffer, indices)

#define F32_CONST(value)                                                       \
  rewriter                                                                     \
      ->create<mlir::arith::ConstantFloatOp>(                                  \
          op.getLoc(), llvm::APFloat(value), rewriter->getF32Type())           \
      .getResult()

#define BOP(Op, lhs, rhs)                                                      \
  rewriter->create<Op>(op.getLoc(), lhs, rhs).getResult()

#define ADDF(lhs, rhs) BOP(mlir::arith::AddFOp, lhs, rhs)
#define MULF(lhs, rhs) BOP(mlir::arith::MulFOp, lhs, rhs)
#define MAXF(lhs, rhs) BOP(mlir::arith::MaxFOp, lhs, rhs)

inline static void genNestedLoops(
    mlir::Value result, mlir::PatternRewriter *rewriter,
    mlir::function_ref<void(const llvm::SmallVector<mlir::Value> &)> body) {
  auto shape = result.getType().cast<mlir::MemRefType>().getShape();
  llvm::SmallVector<mlir::Value> ivs;
  for (auto dim : shape) {
    auto loop = rewriter->create<affine::AffineForOp>(result.getLoc(), 0, dim);
    ivs.push_back(loop.getInductionVar());
    rewriter->setInsertionPointToStart(loop.getBody());
  }
  body(ivs);
}

struct LowerReLU : public LowerOp<relay::ReLUOp> {
  explicit LowerReLU(mlir::MLIRContext *ctx) : LowerOp(ctx) {}

  mlir::LogicalResult lower(relay::ReLUOp op, mlir::ValueRange buffers,
                            mlir::PatternRewriter *rewriter) const override {
    auto data = buffers[0], result = buffers[1];
    genNestedLoops(op->getOpResult(0), rewriter,
                   [&](const llvm::SmallVector<mlir::Value> &ivs) {
                     auto x = LOAD(data, ivs);
                     auto y = MAXF(x, F32_CONST(0.f));
                     STORE(y, result, ivs);
                   });
    return mlir::success();
  }
};

struct LowerBiasAdd : public LowerOp<relay::BiasAddOp> {
  explicit LowerBiasAdd(mlir::MLIRContext *ctx) : LowerOp(ctx) {}

  mlir::LogicalResult lower(relay::BiasAddOp op, mlir::ValueRange buffers,
                            mlir::PatternRewriter *rewriter) const override {
    auto data = buffers[0], bias = buffers[1], result = buffers[2];
    auto axis = op.getAxis();
    genNestedLoops(op->getOpResult(0), rewriter,
                   [&](const llvm::SmallVector<mlir::Value> &ivs) {
                     auto x = LOAD(data, ivs);
                     auto b = LOAD(bias, (mlir::ValueRange{ivs[axis]}));
                     auto y = ADDF(x, b);
                     STORE(y, result, ivs);
                   });
    return mlir::success();
  }
};

struct LowerDense : public LowerOp<relay::DenseOp> {
  explicit LowerDense(mlir::MLIRContext *ctx) : LowerOp(ctx) {}

  mlir::LogicalResult lower(relay::DenseOp op, mlir::ValueRange buffers,
                            mlir::PatternRewriter *rewriter) const override {
    auto data = buffers[0], weight = buffers[1], result = buffers[2];
    auto dataShape = data.getType().cast<mlir::MemRefType>().getShape();
    auto weightShape = weight.getType().cast<mlir::MemRefType>().getShape();
    auto batchSize = dataShape[0], inDim = dataShape[1],
         outDim = weightShape[0];

    FOR(i, 0, batchSize,  // for (i, 0, data.shape[0])
        FOR(j, 0, outDim, // for (j, 0, weight.shape[0])
            auto init = F32_CONST(0.f);
            STORE(init, result, (mlir::ValueRange{i, j})); // result[i, j] = 0
            FOR(k, 0, inDim, // for (k, 0, data.shape[i])
                auto D_ik = LOAD(data, (mlir::ValueRange{i, k}));
                auto W_jk = LOAD(weight, (mlir::ValueRange{j, k}));
                auto mul = MULF(D_ik, W_jk);
                auto prev = LOAD(result, (mlir::ValueRange{i, j}));
                auto add = ADDF(prev, mul);
                // result[i, j] += data[i, k] * weight[j, k]
                STORE(add, result, (mlir::ValueRange{i, j}));) // end k
            )                                                  // end j
        )                                                      // end i

    return mlir::success();
  }
};

struct LowerCall : public LowerOp<func::CallOp> {
  explicit LowerCall(mlir::MLIRContext *ctx) : LowerOp(ctx) {}

  mlir::LogicalResult lower(func::CallOp op, mlir::ValueRange buffers,
                            mlir::PatternRewriter *rewriter) const override {
    rewriter->create<func::CallOp>(op.getLoc(), op.getCallee(), std::nullopt,
                                   buffers);
    return mlir::success();
  }
};

struct EraseReturnValue : public mlir::OpRewritePattern<func::ReturnOp> {
  explicit EraseReturnValue(mlir::MLIRContext *ctx) : OpRewritePattern(ctx) {}

  mlir::LogicalResult
  matchAndRewrite(func::ReturnOp op,
                  mlir::PatternRewriter &rewriter) const override {
    for (auto value : op.getOperands())
      rewriter.eraseOp(value.getDefiningOp());
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op, std::nullopt);
    return mlir::success();
  }
};

struct LowerFunc : public mlir::OpRewritePattern<func::FuncOp> {
  explicit LowerFunc(mlir::MLIRContext *ctx) : OpRewritePattern(ctx) {}

  mlir::LogicalResult
  matchAndRewrite(func::FuncOp func,
                  mlir::PatternRewriter &rewriter) const override;
};

mlir::LogicalResult
LowerFunc::matchAndRewrite(func::FuncOp func,
                           mlir::PatternRewriter &rewriter) const {
  // Convert function prototype
  auto inTypes = llvm::to_vector(
      llvm::map_range(func.getArgumentTypes(), cvtTensorToMemref));
  inTypes.append(llvm::to_vector(
      llvm::map_range(func.getResultTypes(), cvtTensorToMemref)));
  auto newFunc = rewriter.create<func::FuncOp>(
      func.getLoc(), func.getName(),
      rewriter.getFunctionType(inTypes, std::nullopt));
  if (func->hasAttrOfType<mlir::BoolAttr>("primitive"))
    newFunc->setAttr("primitive", rewriter.getBoolAttr(true));
  newFunc->setAttr("num_inputs",
                   rewriter.getI64IntegerAttr(func.getNumArguments()));

  // Find the last use of each intermediate value
  mlir::DenseMap<mlir::Value, mlir::Operation *> lastUse;
  for (auto &block : func.getRegion()) {
    for (auto &op : block) {
      for (auto arg : op.getOperands())
        if (lastUse.count(arg))
          lastUse[arg] = &op;
      for (auto result : op.getResults())
        lastUse.insert({result, nullptr});
    }
  }

  // Convert operations in the function
  rewriter.setInsertionPointToStart(newFunc.addEntryBlock());
  mlir::IRMapping mapper;
  for (auto [tValue, mValue] :
       llvm::zip(func.getArguments(), newFunc.getArguments()))
    mapper.map(tValue, mValue);
  for (auto &block : func.getRegion()) {
    for (auto &op : block) {
      // Clone operation and set result types
      auto newOp = rewriter.clone(op, mapper);
      for (auto result : newOp->getResults())
        result.setType(cvtTensorToMemref(result.getType()));

      // Deallocate arguments which are lastly used by this operation
      if (mlir::isa<func::ReturnOp>(op))
        continue;
      for (auto [prevArg, newArg] :
           llvm::zip(op.getOperands(), newOp->getOperands())) {
        if (!lastUse.count(prevArg))
          continue;
        if (lastUse[prevArg] != &op)
          continue;
        rewriter.create<mlir::memref::DeallocOp>(op.getLoc(), newArg);
      }
    }
  }
  rewriter.eraseOp(func);
  return mlir::success();
}

class RelayToAffine : public RelayToAffineBase<RelayToAffine> {
  void runOnOperation() override;
};

void RelayToAffine::runOnOperation() {
  // Define conversion target
  mlir::ConversionTarget target(getContext());
  target.addLegalDialect<affine::AffineDialect, mlir::arith::ArithDialect,
                         mlir::BuiltinDialect, func::FuncDialect,
                         mlir::memref::MemRefDialect>();
  target.addIllegalDialect<relay::RelayDialect>();
  target.addDynamicallyLegalOp<func::FuncOp>([](func::FuncOp func) {
    return llvm::none_of(func.getArgumentTypes(), [](mlir::Type type) {
      return type.isa<mlir::TensorType>();
    });
  });
  target.addDynamicallyLegalOp<func::CallOp>(
      [](func::CallOp op) { return op.getNumResults() == 0; });
  target.addDynamicallyLegalOp<func::ReturnOp>(
      [](func::ReturnOp op) { return op.getNumOperands() == 0; });

  // Add rewrite patterns
  auto ctx = &getContext();
  mlir::RewritePatternSet patterns(ctx);
  patterns.add<LowerFunc, LowerReLU, LowerBiasAdd, LowerDense, LowerCall,
               EraseReturnValue>(ctx);

  // Apply conversion
  if (mlir::applyPartialConversion(getOperation(), target, std::move(patterns))
          .failed())
    mlir::Pass::signalPassFailure();
}

std::unique_ptr<mlir::Pass> createRelayToAffine() {
  return std::make_unique<RelayToAffine>();
}
