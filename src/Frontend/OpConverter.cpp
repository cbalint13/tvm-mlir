#include <mlir/IR/Builders.h>

#include <tvm/ir/op.h>
#include <tvm/relay/attrs/nn.h>

#include "tvm-mlir/Dialect/Relay/RelayOps.hpp"
#include "tvm-mlir/Frontend/OpConverter.hpp"
#include "tvm-mlir/Support/Common.hpp"

namespace relay {

inline static mlir::UnrankedTensorType
getSameElemTensorType(const mlir::Value &value) {
  return mlir::UnrankedTensorType::get(
      value.getType().cast<mlir::TensorType>().getElementType());
}

#define CVT_FUNC_PARAMS                                                        \
  const std::vector<mlir::Value> &args, const tvm::Attrs &attrs,               \
      mlir::Location loc, mlir::OpBuilder *builder

#define CVT_FUNC(Op) static mlir::OpState cvt##Op(CVT_FUNC_PARAMS)

template <class O, int... argIndices> inline CVT_FUNC(SameElemTypeNoAttr) {
  return builder->create<O>(loc, getSameElemTensorType(args[0]),
                            args[argIndices]...);
}

#define ONE_ARG 0
#define TWO_ARG 0, 1

CVT_FUNC(BiasAdd) {
  auto biasAddAttrs = attrs.as<tvm::relay::BiasAddAttrs>();
  return builder->create<BiasAddOp>(loc, getSameElemTensorType(args[0]),
                                    args[0], args[1], biasAddAttrs->axis);
}

static std::unordered_map<tvm::String, mlir::OpState (*)(CVT_FUNC_PARAMS)>
    cvtFuncs{{"nn.relu", cvtSameElemTypeNoAttr<ReLUOp, ONE_ARG>},
             {"nn.dense", cvtSameElemTypeNoAttr<DenseOp, TWO_ARG>},
             {"nn.bias_add", cvtBiasAdd}};

mlir::OpState ConvertRelayOp(const tvm::String &name, CVT_FUNC_PARAMS) {
  if (cvtFuncs.count(name))
    return cvtFuncs[name](args, attrs, loc, builder);
  else
    Fatal("Operator `{}` is not supported.", name.c_str());
}

} // namespace relay
