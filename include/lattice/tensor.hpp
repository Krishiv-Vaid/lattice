#pragma once

#include "lattice/storage.hpp"

#include <cstddef>
#include <memory>
#include <vector>

namespace lattice {

class Tensor {
public:
    Tensor(
        const std::vector<double>& data,
        const std::vector<std::size_t>& shape
    );

    const std::vector<std::size_t>&
    shape() const noexcept;

    const std::vector<std::size_t>&
    strides() const noexcept;

    std::size_t ndim() const noexcept;

    std::size_t numel() const noexcept;

    bool is_contiguous() const noexcept;

    double& at(
        const std::vector<std::size_t>& indices
    );

    const double& at(
        const std::vector<std::size_t>& indices
    ) const;

private:
    static std::vector<std::size_t>
    compute_strides(
        const std::vector<std::size_t>& shape
    );

    std::size_t storage_index(
        const std::vector<std::size_t>& indices
    ) const;

    std::shared_ptr<Storage> storage_;

    std::vector<std::size_t> shape_;

    std::vector<std::size_t> strides_;

    std::size_t offset_;
};

}  // namespace lattice