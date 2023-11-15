#pragma once
#include <llvm/Support/Error.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/FormatVariadic.h>

#include <utility>

/// Fatal error
template <class S, class... Args>
[[noreturn]] inline void Fatal(const S &format, Args &&...args) {
  llvm::outs() << llvm::formatv(format, std::forward<Args>(args)...) << '\n';
  exit(-1);
}

/// Error
template <class S, class... Args>
inline void Error(const S &format, Args &&...args) {
  llvm::errs() << llvm::formatv(format, std::forward<Args>(args)...) << '\n';
}

/// Error
template <class S, class... Args>
inline void Debug(const S &format, Args &&...args) {
  llvm::dbgs() << llvm::formatv(format, std::forward<Args>(args)...) << '\n';
}
