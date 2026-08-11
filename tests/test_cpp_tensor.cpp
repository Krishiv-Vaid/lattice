#include "lattice/tensor.hpp"

#include <cassert>
#include <iostream>
#include <vector>

int main() {
    lattice::Tensor tensor(
        {
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
        },
        {
            2,
            3,
        }
    );

    assert(
        tensor.ndim() == 2
    );

    assert(
        tensor.numel() == 6
    );

    assert(
        tensor.shape()
        == std::vector<std::size_t>({
            2,
            3,
        })
    );

    assert(
        tensor.strides()
        == std::vector<std::size_t>({
            3,
            1,
        })
    );

    assert(
        tensor.is_contiguous()
    );

    assert(
        tensor.at({0, 0}) == 1.0
    );

    assert(
        tensor.at({1, 2}) == 6.0
    );

    tensor.at({1, 1}) = 99.0;

    assert(
        tensor.at({1, 1}) == 99.0
    );

    std::cout
        << "C++ Tensor smoke test passed\n";

    return 0;
}