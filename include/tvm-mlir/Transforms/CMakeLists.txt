set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Transforms)
add_public_tablegen_target(mlir_transforms_pass_inc_gen)
add_mlir_doc(Passes TransformsPasses ./ -gen-pass-doc)
