#include <mlir/Dialect/Func/IR/FuncOps.h>

#include <tvm/relay/expr_functor.h>

#include "tvm-mlir/Dialect/Relay/RelayOps.hpp"
#include "tvm-mlir/Frontend/OpConverter.hpp"
#include "tvm-mlir/Frontend/RelayImporter.hpp"
#include "tvm-mlir/Support/Common.hpp"

namespace relay {

class RelayImporter
    : private tvm::relay::ExprFunctor<mlir::Value(const tvm::relay::Expr &)> {
public:
  RelayImporter(const std::string &srcPath, mlir::MLIRContext *ctx)
      : srcPath(srcPath), builder(ctx) {}

  mlir::ModuleOp Import(tvm::IRModule tvmMod);

private:
  using Base = tvm::relay::ExprFunctor<mlir::Value(const tvm::relay::Expr &)>;

  std::string srcPath;
  mlir::OpBuilder builder;
  std::unordered_map<tvm::relay::Expr, mlir::Value, tvm::ObjectHash,
                     tvm::ObjectEqual>
      exprValueMap;

  mlir::Value VisitExpr(const tvm::relay::Expr &n) override;
  mlir::Value VisitExpr_(const tvm::relay::ConstantNode *constant) override;
  mlir::Value VisitExpr_(const tvm::relay::VarNode *var) override;
  mlir::Value VisitExpr_(const tvm::relay::CallNode *call) override;

  mlir::Location cvtLoc(const tvm::Span &span) {
    return mlir::FileLineColLoc::get(builder.getStringAttr(srcPath), span->line,
                                     span->column);
  }
};

mlir::ModuleOp ImportRelay(tvm::IRModule mod, const std::string &srcPath,
                           mlir::MLIRContext *ctx) {
  return RelayImporter(srcPath, ctx).Import(mod);
}

inline static llvm::SmallVector<int64_t>
cvtTVMShape(const tvm::runtime::Array<tvm::PrimExpr> &relayShape) {
  llvm::SmallVector<int64_t> shape;
  for (const auto &dim : relayShape) {
    auto imm = dim.as<tvm::IntImmNode>();
    if (!imm)
      Fatal("Shape dimension is not a constant.");
    shape.push_back(imm->value);
  }
  return shape;
}

static mlir::Type getF32Type(mlir::OpBuilder *builder) {
  return builder->getF32Type();
}

static std::unordered_map<tvm::DataType, mlir::Type (*)(mlir::OpBuilder *)>
    typeMap{{tvm::DataType::Float(32), &getF32Type}};

inline static mlir::Type cvtTVMDataType(const tvm::DataType &dtype,
                                        mlir::OpBuilder *builder) {
  if (typeMap.count(dtype))
    return typeMap[dtype](builder);
  else
    Fatal("Data type is not supported.");
}

inline static mlir::TensorType
cvtRelayTensorType(const tvm::relay::TensorTypeNode *type,
                   mlir::OpBuilder *builder) {
  auto shape = cvtTVMShape(type->shape);
  auto dtype = cvtTVMDataType(type->dtype, builder);
  return mlir::RankedTensorType::get(shape, dtype);
}

inline static mlir::TensorType extractRelayVarType(const tvm::relay::Var &var,
                                                   mlir::OpBuilder *builder) {
  auto &type = var->type_annotation;
  if (!type.defined())
    Fatal("Relay variable {} is not type-annotated.", var->name_hint().c_str());
  auto tvmTensorType = type.as<tvm::relay::TensorTypeNode>();
  if (!tvmTensorType)
    Fatal("Variable {} is not of tensor type.", var->name_hint().c_str());
  return cvtRelayTensorType(tvmTensorType, builder);
}

mlir::ModuleOp RelayImporter::Import(tvm::IRModule tvmMod) {
  // Create MLIR module
  auto mod = mlir::ModuleOp::create(builder.getUnknownLoc());
  builder.setInsertionPointToEnd(mod.getBody());

  // Create prototype of main function
  auto relayMain = tvmMod->functions.at(tvmMod->GetGlobalVar("main"))
                       .as<tvm::relay::FunctionNode>();
  std::vector<mlir::Type> paramTypes;
  for (const auto &var : relayMain->params)
    paramTypes.push_back(extractRelayVarType(var, &builder));
  auto funcType = builder.getFunctionType(paramTypes, std::nullopt);
  auto mainFunc = builder.create<mlir::func::FuncOp>(cvtLoc(relayMain->span),
                                                     "main", funcType);

  // Add parameter values to symbol table
  auto entry = mainFunc.addEntryBlock();
  for (auto i = 0u; i < entry->getNumArguments(); i++) {
    auto &var = relayMain->params[i];
    auto value = entry->getArgument(i);
    exprValueMap.insert({var, value});
  }

  // Insert operations to function body
  builder.setInsertionPointToStart(entry);
  auto ret = VisitExpr(relayMain->body);
  builder.create<mlir::func::ReturnOp>(cvtLoc(relayMain->body->span), ret);

  return mod;
}

mlir::Value RelayImporter::VisitExpr(const tvm::relay::Expr &expr) {
  if (exprValueMap.count(expr))
    return exprValueMap[expr];
  auto ret = Base::VisitExpr(expr);
  exprValueMap.insert({expr, ret});
  return ret;
}

template <class T>
static mlir::DenseElementsAttr createDense(mlir::RankedTensorType type,
                                           char *data, size_t size) {
  return mlir::DenseElementsAttr::get(
      type, llvm::ArrayRef(reinterpret_cast<T *>(data),
                           reinterpret_cast<T *>(data + size)));
}

static std::unordered_map<tvm::DataType,
                          mlir::DenseElementsAttr (*)(mlir::RankedTensorType,
                                                      char *, size_t)>
    denseCreateFn{{tvm::DataType::Float(32), createDense<float>}};

mlir::Value
RelayImporter::VisitExpr_(const tvm::relay::ConstantNode *constant) {
  // Get tensor type for this constant
  auto tensor = constant->data;
  auto shape = llvm::ArrayRef(tensor->shape, tensor->shape + tensor->ndim);
  auto tvmDType = tensor.DataType();
  auto elemType = cvtTVMDataType(tvmDType, &builder);
  auto type = mlir::RankedTensorType::get(shape, elemType);
  auto size = tvm::runtime::GetDataSize(*tensor.operator->());

  // Create constant operation
  if (!denseCreateFn.count(tensor.DataType()))
    Fatal("Data type is not supported.");
  auto attr = denseCreateFn[tvmDType](
      type, reinterpret_cast<char *>(tensor->data), size);
  auto op = builder.create<ConstantOp>(cvtLoc(constant->span), type, attr);

  return op.getResult();
}

mlir::Value RelayImporter::VisitExpr_(const tvm::relay::VarNode *var) {
  return exprValueMap.at(tvm::GetRef<tvm::relay::Var>(var));
}

mlir::Value RelayImporter::VisitExpr_(const tvm::relay::CallNode *call) {
  auto relayOp = call->op.as<tvm::relay::OpNode>();
  if (!relayOp)
    Fatal("Call to non-operator expression is not supported.");
  std::vector<mlir::Value> operands;
  for (auto &arg : call->args)
    operands.push_back(VisitExpr(arg));
  auto op = relay::ConvertRelayOp(relayOp->name, operands, call->attrs,
                                  cvtLoc(call->span), &builder);
  return op->getResult(0);
}

} // namespace relay
