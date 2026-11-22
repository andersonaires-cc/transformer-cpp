
#ifndef TRANSFORMER_MATRIX_HPP
#define TRANSFORMER_MATRIX_HPP

#include <vector>
#include <iostream>
#include <initializer_list>
#include <stdexcept>
#include <cstddef>

class Matrix {

private:
    std::size_t rows_;
    std::size_t cols_;
    std::vector<double> data_;

public:

    // Construtor
    Matrix(std::size_t rows, std::size_t cols);

    // Construtor com valores iniciais
    Matrix(std::initializer_list<std::initializer_list<double>> values);

    // Acesso aos elementos
    double& operator()(std::size_t row, std::size_t col);

    double operator()(std::size_t row, std::size_t col) const;

    // Dimensoes
    std::size_t rows() const;
    std::size_t cols() const;

    // Operacoes matriciais
    Matrix operator+(const Matrix& other) const;
    Matrix operator-(const Matrix& other) const;
    Matrix operator*(double scalar) const;

    // Multiplicacao matricial
    Matrix matmul(const Matrix& other) const;

    // Multiplicacao elemento a elemento
    Matrix hadamard(const Matrix& other) const;

    // Transposta
    Matrix transpose() const;

    // Impressao
    void print() const;

    // Matriz identidade
    static Matrix identity(std::size_t size);
};

#endif
