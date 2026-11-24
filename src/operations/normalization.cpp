
#include "operations/normalization.hpp"

#include <cmath>
#include <stdexcept>

LayerNorm::LayerNorm(
    std::size_t features,
    double epsilon
)
    : features_(features),
      epsilon_(epsilon),
      gamma_(1, features),
      beta_(1, features) {

    if (features == 0 || epsilon <= 0.0) {
        throw std::invalid_argument(
            "Parametros invalidos para LayerNorm."
        );
    }

    // Inicializacao dos parametros
    for (std::size_t j = 0; j < features_; j++) {

        gamma_(0, j) = 1.0;

        beta_(0, j) = 0.0;
    }
}

Matrix LayerNorm::forward(const Matrix& input) const {

    if (input.cols() != features_) {

        throw std::invalid_argument(
            "Numero de features incompatível."
        );
    }

    Matrix result(input.rows(), input.cols());

    for (std::size_t i = 0; i < input.rows(); i++) {

        double mean = 0.0;

        // Calcular media
        for (std::size_t j = 0; j < features_; j++) {

            mean += input(i, j);
        }

        mean /= static_cast<double>(features_);

        // Calcular variancia populacional
        double variance = 0.0;

        for (std::size_t j = 0; j < features_; j++) {

            double diff = input(i, j) - mean;

            variance += diff * diff;
        }

        variance /= static_cast<double>(features_);

        // Normalizar e aplicar parametros treinaveis
        for (std::size_t j = 0; j < features_; j++) {

            double normalized =
                (input(i, j) - mean) /
                std::sqrt(variance + epsilon_);

            result(i, j) =
                gamma_(0, j) * normalized +
                beta_(0, j);
        }
    }

    return result;
}
