
#include "operations/activations.hpp"

#include <cmath>
#include <algorithm>

namespace Activations {

// Softmax aplicada individualmente em cada linha
Matrix softmax(const Matrix& input) {

    Matrix result(input.rows(), input.cols());

    for (std::size_t i = 0; i < input.rows(); i++) {

        double maxValue = input(i, 0);

        // Encontrar o maior valor da linha
        for (std::size_t j = 1; j < input.cols(); j++) {
            maxValue = std::max(maxValue, input(i, j));
        }

        double sum = 0.0;

        // Exponencial com estabilizacao numerica
        for (std::size_t j = 0; j < input.cols(); j++) {

            result(i, j) = std::exp(input(i, j) - maxValue);

            sum += result(i, j);
        }

        // Normalizacao
        for (std::size_t j = 0; j < input.cols(); j++) {

            result(i, j) /= sum;
        }
    }

    return result;
}

// ReLU
Matrix relu(const Matrix& input) {

    Matrix result(input.rows(), input.cols());

    for (std::size_t i = 0; i < input.rows(); i++) {

        for (std::size_t j = 0; j < input.cols(); j++) {

            result(i, j) = std::max(0.0, input(i, j));
        }
    }

    return result;
}

// Sigmoid
Matrix sigmoid(const Matrix& input) {

    Matrix result(input.rows(), input.cols());

    for (std::size_t i = 0; i < input.rows(); i++) {

        for (std::size_t j = 0; j < input.cols(); j++) {

            double x = input(i, j);

            result(i, j) = 1.0 / (1.0 + std::exp(-x));
        }
    }

    return result;
}

// GELU aproximada
Matrix gelu(const Matrix& input) {

    Matrix result(input.rows(), input.cols());

    constexpr double sqrtTwoOverPi = 0.7978845608;
    constexpr double coefficient = 0.044715;

    for (std::size_t i = 0; i < input.rows(); i++) {

        for (std::size_t j = 0; j < input.cols(); j++) {

            double x = input(i, j);

            double inner =
                sqrtTwoOverPi *
                (x + coefficient * x * x * x);

            result(i, j) =
                0.5 * x * (1.0 + std::tanh(inner));
        }
    }

    return result;
}

}
