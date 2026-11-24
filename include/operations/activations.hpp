
#ifndef TRANSFORMER_ACTIVATIONS_HPP
#define TRANSFORMER_ACTIVATIONS_HPP

#include "core/matrix.hpp"

namespace Activations {

    Matrix softmax(const Matrix& input);

    Matrix relu(const Matrix& input);

    Matrix sigmoid(const Matrix& input);

    Matrix gelu(const Matrix& input);

}

#endif
