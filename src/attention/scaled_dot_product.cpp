#include "attention/scaled_dot_product.hpp"

#include "operations/activations.hpp"
#include <cmath>
#include <stdexcept>

AttentionResult ScaledDotProductAttention::forward(
    const Matrix& Q,
    const Matrix& K,
    const Matrix& V
) {


    if (Q.cols() != K.cols()) {

        throw std::invalid_argument(
            "Q e K devem possuir a mesma dimensao de features."
        );
    }

    if (K.rows() != V.rows()) {

        throw std::invalid_argument(
            "K e V devem possuir a mesma quantidade de posicoes."
        );
    }


    const double d_k = static_cast<double>(K.cols());


    Matrix scores = Q.matmul(K.transpose());


    const double scale = 1.0 / std::sqrt(d_k);

    scores = scores * scale;


    Matrix weights = Activations::softmax(scores);


    Matrix output = weights.matmul(V);


    return {
        output,
        weights
    };
}