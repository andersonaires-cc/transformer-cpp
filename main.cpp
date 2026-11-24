
#include <iostream>

#include "core/matrix.hpp"
#include "attention/scaled_dot_product.hpp"

int main() {

    // Entrada de tres elementos
    Matrix X = {
        {1.0, 0.0},
        {0.0, 1.0},
        {1.0, 1.0}
    };

    // Configuracao simplificada para teste
    Matrix Q = X;
    Matrix K = X;
    Matrix V = X;

    std::cout << "\nINPUT:\n";
    X.print();

    // Executar Scaled Dot-Product Attention
    AttentionResult result =
        ScaledDotProductAttention::forward(Q, K, V);

    std::cout << "\nATTENTION WEIGHTS:\n";
    result.weights.print();

    std::cout << "\nATTENTION OUTPUT:\n";
    result.output.print();

    return 0;
}
