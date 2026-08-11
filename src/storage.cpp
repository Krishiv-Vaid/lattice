#include "lattice/storage.hpp"

#include <stdexcept>

namespace lattice {

Storage::Storage(
    std::size_t size
)
    : data_(
        std::make_shared<
            std::vector<double>
        >(size)
    ) {}

Storage::Storage(
    const std::vector<double>& values
)
    : data_(
        std::make_shared<
            std::vector<double>
        >(values)
    ) {}

std::size_t Storage::size() const noexcept {
    return data_->size();
}

double* Storage::data() noexcept {
    return data_->data();
}

const double* Storage::data() const noexcept {
    return data_->data();
}

double& Storage::operator[](
    std::size_t index
) {
    if (index >= data_->size()) {
        throw std::out_of_range(
            "Storage index out of range"
        );
    }

    return (*data_)[index];
}

const double& Storage::operator[](
    std::size_t index
) const {
    if (index >= data_->size()) {
        throw std::out_of_range(
            "Storage index out of range"
        );
    }

    return (*data_)[index];
}

}  // namespace lattice