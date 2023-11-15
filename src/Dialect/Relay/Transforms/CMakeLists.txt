add_mlir_dialect_library(mlir_relay_transforms
  OBJECT
    ShapeInference.cpp
    OpFusion.cpp
  DEPENDS
    mlir_relay_pass_inc_gen
  LINK_LIBS PRIVATE
    mlir_relay_dialect
  PUBLIC
    MLIR
)
