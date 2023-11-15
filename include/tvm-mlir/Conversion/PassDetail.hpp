#pragma once
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/BuiltinDialect.h>

#include "tvm-mlir/Conversion/Passes.hpp"
#include "tvm-mlir/Dialect/Relay/RelayDialect.hpp"

#define GEN_PASS_CLASSES
#include "tvm-mlir/Conversion/Passes.h.inc"
