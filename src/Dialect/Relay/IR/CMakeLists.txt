add_mlir_dialect_library(mlir_relay_dialect
  OBJECT
    RelayDialect.cpp
    RelayOps.cpp
  DEPENDS
    MLIRRelayOpsIncGen
  LINK_LIBS PUBLIC
    MLIR
)
