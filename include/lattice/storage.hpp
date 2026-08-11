#pragma once

#include <cstddef>
#include <memory>
#include <vector>

namespace lattice {

class Storage {
public:
    explicit Storage(std::size_t size);

    explicit Storage(
        const std::vector<double>& values
    );

    std::size_t size() const noexcept;

    double* data() noexcept;

    const double* data() const noexcept;

    double& operator[](
        std::size_t index
    );

    const double& operator[](
        std::size_t index
    ) const;

private:
    std::shared_ptr<
        std::vector<double>
    > data_;
};

}  // namespace lattice