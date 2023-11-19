#include "tvm-mlir/Support/MemRef.hpp"

MemRef::MemRef(mlir::MemRefType type) {
  // Initialize buffer shape and element type
  auto shapeArr = type.getShape();
  this->shape = llvm::SmallVector<int64_t>(shapeArr.begin(), shapeArr.end());
  auto elemSize = (type.getElementTypeBitWidth() + 7) / 8;

  // Compute strides
  this->strides = llvm::SmallVector<int64_t>(type.getRank(), 1);
  for (auto i = 0u; i < type.getRank() - 1; i++)
    this->strides[i] = this->strides[i + 1] * this->shape[i];

  // Allocate buffer memory
  this->size = type.getNumElements() * elemSize;
  this->data = std::malloc(this->size);
}

static int64_t kOffset = 0;

void MemRef::PopulateLLJITArgs(llvm::SmallVector<void *> *args) {
  args->push_back(&this->data);
  args->push_back(&this->data);
  args->push_back(&kOffset);
  for (auto &d : this->shape)
    args->push_back(&d);
  for (auto &s : this->strides)
    args->push_back(&s);
}
