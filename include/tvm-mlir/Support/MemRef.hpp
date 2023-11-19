#pragma once
#include <mlir/IR/BuiltinTypes.h>

class MemRef {
public:
  explicit MemRef(mlir::MemRefType type);

  ~MemRef() { std::free(data); }

  void LoadData(const void *src) const { std::memcpy(data, src, this->size); }

  template <class T> T *GetDataAs() const {
    return reinterpret_cast<T *>(data);
  }

  void PopulateLLJITArgs(llvm::SmallVector<void *> *args);

private:
  llvm::SmallVector<int64_t> shape, strides;
  size_t size;
  void *data = nullptr;
};
