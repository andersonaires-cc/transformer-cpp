
#ifndef TRANSFORMER_NORMALIZATION_HPP
#define TRANSFORMER_NORMALIZATION_HPP

#include "core/matrix.hpp"

class LayerNorm {

private:

    std::size_t features_;

    double epsilon_;

    Matrix gamma_;
    Matrix beta_;

public:

    explicit LayerNorm(
        std::size_t features,
        double epsilon = 1e-5
    );

    Matrix forward(const Matrix& input) const;

};

#endif
