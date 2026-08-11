#include "lattice/tensor.hpp"

#include <stdexcept>

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

const std::vector<std::size_t>&
Tensor::shape() const noexcept {
    return shape_;
}

const std::vector<std::size_t>&
Tensor::strides() const noexcept {
    return strides_;
}

std::size_t Tensor::ndim() const noexcept {
    return shape_.size();
}

std::size_t Tensor::numel() const noexcept {
    if (shape_.empty()) {
        return 1;
    }

    std::size_t total = 1;

    for (const auto dimension : shape_) {
        total *= dimension;
    }

    return total;
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

std::size_t Tensor::storage_index(
    const std::vector<std::size_t>& indices
) const {
    if (indices.size() != shape_.size()) {
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