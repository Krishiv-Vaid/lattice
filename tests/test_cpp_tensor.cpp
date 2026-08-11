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
            7.0,
            8.0,
            9.0,
        },
        {
            3,
            3,
        }
    );

    assert(tensor.ndim() == 2);
    assert(tensor.numel() == 9);
    assert(tensor.offset() == 0);
    assert(tensor.is_contiguous());

    assert(
        tensor.shape()
        == std::vector<std::size_t>({
            3,
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
        tensor.at({0, 0}) == 1.0
    );

    assert(
        tensor.at({2, 2}) == 9.0
    );

    auto transposed = tensor.transpose(
        0,
        1
    );

    assert(
        transposed.shape()
        == std::vector<std::size_t>({
            3,
            3,
        })
    );

    assert(
        transposed.strides()
        == std::vector<std::size_t>({
            1,
            3,
        })
    );

    assert(!transposed.is_contiguous());

    transposed.at({1, 2}) = 99.0;

    assert(
        tensor.at({2, 1}) == 99.0
    );

    auto reshaped = tensor.reshape({
        1,
        9,
    });

    assert(
        reshaped.shape()
        == std::vector<std::size_t>({
            1,
            9,
        })
    );

    assert(
        reshaped.strides()
        == std::vector<std::size_t>({
            9,
            1,
        })
    );

    reshaped.at({0, 1}) = 42.0;

    assert(
        tensor.at({0, 1}) == 42.0
    );

    auto row_slice = tensor.slice(
        0,
        1,
        3
    );

    assert(
        row_slice.shape()
        == std::vector<std::size_t>({
            2,
            3,
        })
    );

    assert(
        row_slice.strides()
        == std::vector<std::size_t>({
            3,
            1,
        })
    );

    assert(
        row_slice.offset() == 3
    );

    assert(
        row_slice.at({0, 0}) == 4.0
    );

    assert(
        row_slice.at({1, 2}) == 9.0
    );

    row_slice.at({0, 2}) = 123.0;

    assert(
        tensor.at({1, 2}) == 123.0
    );

    auto column_slice = tensor.slice(
        1,
        1,
        3
    );

    assert(
        column_slice.shape()
        == std::vector<std::size_t>({
            3,
            2,
        })
    );

    assert(
        column_slice.strides()
        == std::vector<std::size_t>({
            3,
            1,
        })
    );

    assert(
        column_slice.offset() == 1
    );

    assert(
        column_slice.at({0, 0}) == 42.0
    );

    assert(
        column_slice.at({2, 1}) == 9.0
    );

    auto stepped = tensor.slice(
        1,
        0,
        3,
        2
    );

    assert(
        stepped.shape()
        == std::vector<std::size_t>({
            3,
            2,
        })
    );

    assert(
        stepped.strides()
        == std::vector<std::size_t>({
            3,
            2,
        })
    );

    assert(
        stepped.at({0, 0}) == 1.0
    );

    assert(
        stepped.at({0, 1}) == 3.0
    );

    assert(
        stepped.at({2, 1}) == 9.0
    );

    stepped.at({1, 1}) = 777.0;

    assert(
        tensor.at({1, 2}) == 777.0
    );

    auto nested = tensor
        .slice(0, 1, 3)
        .slice(1, 1, 3);

    assert(
        nested.shape()
        == std::vector<std::size_t>({
            2,
            2,
        })
    );

    assert(
        nested.offset() == 4
    );

    assert(
        nested.at({0, 0})
        == tensor.at({1, 1})
    );

    nested.at({1, 1}) = 555.0;

    assert(
        tensor.at({2, 2}) == 555.0
    );

    bool reshape_failed = false;

    try {
        transposed.reshape({
            9,
        });
    } catch (
        const std::invalid_argument&
    ) {
        reshape_failed = true;
    }

    assert(reshape_failed);

    bool zero_step_failed = false;

    try {
        tensor.slice(
            0,
            0,
            2,
            0
        );
    } catch (
        const std::invalid_argument&
    ) {
        zero_step_failed = true;
    }

    assert(zero_step_failed);

    bool invalid_dimension_failed = false;

    try {
        tensor.slice(
            5,
            0,
            1
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
        << "C++ Tensor slice tests passed\n";

    return 0;
}