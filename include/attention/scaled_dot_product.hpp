#ifndef TRANSFORMER_SCALED_DOT_PRODUCT_HPP
#define TRANSFORMER_SCALED_DOT_PRODUCT_HPP

#include "core/matrix.hpp"

struct AttentionResult {
    Matrix output;
    Matrix weights;
};

class ScaledDotProductAttention 
{

public:
    static AttentionResult forward(
        const Matrix& Q,
        const Matrix& K,
        const Matrix& V
    );    
};



#endif