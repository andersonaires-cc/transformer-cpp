
#include "core/matrix.hpp"

#include <iostream>

int main() {

    Matrix A = {
        {1, 2},
        {3, 4}
    };

    Matrix B = {
        {5, 6},
        {7, 8}
    };

    std::cout << "Matriz A:\n";
    A.print();

    std::cout << "\nMatriz B:\n";
    B.print();

    Matrix C = A.matmul(B);

    std::cout << "\nMultiplicacao A * B:\n";
    C.print();

    std::cout << "\nTransposta de A:\n";
    A.transpose().print();

    std::cout << "\nSoma A + B:\n";
    (A + B).print();

    std::cout << "\nMatriz identidade:\n";
    Matrix::identity(3).print();

    return 0;
}
