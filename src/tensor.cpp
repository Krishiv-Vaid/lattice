#include "lattice/tensor.hpp"

#include <stdexcept>
#include <utility>

namespace lattice {

Tensor::Tensor(
    const std::vector<double>& data,
    const std::vector<std::size_t>& shape
)
    : storage_(
        std::make_shared<Storage>(data)
    ),
      shape_(shape),
      strides_(compute_strides(shape)),
      offset_(0) {

    if (data.size() != numel()) {
        throw std::invalid_argument(
            "Tensor data size does not match shape"
        );
    }
}

Tensor::Tensor(
    std::shared_ptr<Storage> storage,
    const std::vector<std::size_t>& shape,
    const std::vector<std::size_t>& strides,
    std::size_t offset
)
    : storage_(std::move(storage)),
      shape_(shape),
      strides_(strides),
      offset_(offset) {}

const std::vector<std::size_t>&
Tensor::shape() const noexcept {
    return shape_;
}

const std::vector<std::size_t>&
Tensor::strides() const noexcept {
    return strides_;
}

std::size_t Tensor::offset() const noexcept {
    return offset_;
}

std::size_t Tensor::ndim() const noexcept {
    return shape_.size();
}

std::size_t Tensor::numel() const noexcept {
    return compute_numel(shape_);
}

bool Tensor::is_contiguous() const noexcept {
    return (
        strides_
        == compute_strides(shape_)
    );
}

double& Tensor::at(
    const std::vector<std::size_t>& indices
) {
    return (*storage_)[
        storage_index(indices)
    ];
}

const double& Tensor::at(
    const std::vector<std::size_t>& indices
) const {
    return (*storage_)[
        storage_index(indices)
    ];
}

Tensor Tensor::transpose(
    std::size_t dim0,
    std::size_t dim1
) const {
    if (
        dim0 >= ndim()
        || dim1 >= ndim()
    ) {
        throw std::out_of_range(
            "Tensor dimension out of range"
        );
    }

    auto new_shape = shape_;
    auto new_strides = strides_;

    std::swap(
        new_shape[dim0],
        new_shape[dim1]
    );

    std::swap(
        new_strides[dim0],
        new_strides[dim1]
    );

    return Tensor(
        storage_,
        new_shape,
        new_strides,
        offset_
    );
}

Tensor Tensor::reshape(
    const std::vector<std::size_t>& shape
) const {
    if (!is_contiguous()) {
        throw std::invalid_argument(
            "Cannot reshape a non-contiguous tensor"
        );
    }

    if (
        compute_numel(shape)
        != numel()
    ) {
        throw std::invalid_argument(
            "Reshape cannot change number of elements"
        );
    }

    return Tensor(
        storage_,
        shape,
        compute_strides(shape),
        offset_
    );
}

Tensor Tensor::slice(
    std::size_t dim,
    std::size_t start,
    std::size_t stop,
    std::size_t step
) const {
    if (dim >= ndim()) {
        throw std::out_of_range(
            "Tensor dimension out of range"
        );
    }

    if (step == 0) {
        throw std::invalid_argument(
            "Slice step must be positive"
        );
    }

    if (start > shape_[dim]) {
        throw std::out_of_range(
            "Slice start out of range"
        );
    }

    if (stop > shape_[dim]) {
        throw std::out_of_range(
            "Slice stop out of range"
        );
    }

    if (stop < start) {
        throw std::invalid_argument(
            "Slice stop must be greater "
            "than or equal to start"
        );
    }

    auto new_shape = shape_;
    auto new_strides = strides_;

    const std::size_t length = (
        stop <= start
        ? 0
        : (
            (stop - start + step - 1)
            / step
        )
    );

    new_shape[dim] = length;

    const std::size_t new_offset = (
        offset_
        + start * strides_[dim]
    );

    new_strides[dim] *= step;

    return Tensor(
        storage_,
        new_shape,
        new_strides,
        new_offset
    );
}

std::vector<std::size_t>
Tensor::compute_strides(
    const std::vector<std::size_t>& shape
) {
    std::vector<std::size_t> strides(
        shape.size(),
        1
    );

    if (shape.empty()) {
        return strides;
    }

    for (
        std::size_t i = shape.size();
        i-- > 1;
    ) {
        strides[i - 1] = (
            strides[i]
            * shape[i]
        );
    }

    return strides;
}

std::size_t Tensor::compute_numel(
    const std::vector<std::size_t>& shape
) {
    if (shape.empty()) {
        return 1;
    }

    std::size_t total = 1;

    for (
        const auto dimension : shape
    ) {
        total *= dimension;
    }

    return total;
}

std::size_t Tensor::storage_index(
    const std::vector<std::size_t>& indices
) const {
    if (
        indices.size()
        != shape_.size()
    ) {
        throw std::invalid_argument(
            "Incorrect number of tensor indices"
        );
    }

    std::size_t index = offset_;

    for (
        std::size_t dimension = 0;
        dimension < shape_.size();
        ++dimension
    ) {
        if (
            indices[dimension]
            >= shape_[dimension]
        ) {
            throw std::out_of_range(
                "Tensor index out of range"
            );
        }

        index += (
            indices[dimension]
            * strides_[dimension]
        );
    }

    return index;
}

}  // namespace lattice