add_llvm_executable(tvm-relay tvm-relay.cpp)
target_link_libraries(tvm-relay
  PRIVATE
    mlir_relay_dialect
    mlir_relay_frontend
    mlir_relay_transforms
    mlir_dialect_conversion
  PUBLIC
    MLIR
    tvm::tvm
)
