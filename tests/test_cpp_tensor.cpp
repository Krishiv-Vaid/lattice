#include "lattice/tensor.hpp"

#include <cassert>
#include <iostream>
#include <stdexcept>
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

    assert(tensor.ndim() == 2);
    assert(tensor.numel() == 6);
    assert(tensor.offset() == 0);

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

    assert(tensor.is_contiguous());

    assert(
        tensor.at({0, 0}) == 1.0
    );

    assert(
        tensor.at({1, 2}) == 6.0
    );

    auto transposed = tensor.transpose(
        0,
        1
    );

    assert(
        transposed.shape()
        == std::vector<std::size_t>({
            3,
            2,
        })
    );

    assert(
        transposed.strides()
        == std::vector<std::size_t>({
            1,
            3,
        })
    );

    assert(
        !transposed.is_contiguous()
    );

    assert(
        transposed.at({0, 0}) == 1.0
    );

    assert(
        transposed.at({2, 1}) == 6.0
    );

    transposed.at({1, 1}) = 99.0;

    assert(
        tensor.at({1, 1}) == 99.0
    );

    auto reshaped = tensor.reshape({
        3,
        2,
    });

    assert(
        reshaped.shape()
        == std::vector<std::size_t>({
            3,
            2,
        })
    );

    assert(
        reshaped.strides()
        == std::vector<std::size_t>({
            2,
            1,
        })
    );

    assert(
        reshaped.at({2, 1}) == 6.0
    );

    reshaped.at({0, 1}) = 42.0;

    assert(
        tensor.at({0, 1}) == 42.0
    );

    bool reshape_failed = false;

    try {
        transposed.reshape({
            6,
        });
    } catch (
        const std::invalid_argument&
    ) {
        reshape_failed = true;
    }

    assert(reshape_failed);

    bool invalid_dimension_failed = false;

    try {
        tensor.transpose(
            0,
            5
        );
    } catch (
        const std::out_of_range&
    ) {
        invalid_dimension_failed = true;
    }

    assert(
        invalid_dimension_failed
    );

    std::cout
        << "C++ Tensor view tests passed\n";

    return 0;
}