
#include "core/matrix.hpp"

#include <iomanip>

// Construtor
Matrix::Matrix(std::size_t rows, std::size_t cols)
    : rows_(rows), cols_(cols), data_(rows * cols, 0.0) {

    if (rows == 0 || cols == 0) {
        throw std::invalid_argument(
            "Matriz deve possuir dimensoes positivas."
        );
    }
}

// Construtor com inicializacao
Matrix::Matrix(
    std::initializer_list<std::initializer_list<double>> values
)
    : rows_(values.size()), cols_(0) {

    if (rows_ == 0 || values.begin()->size() == 0) {
        throw std::invalid_argument("Matriz vazia.");
    }

    cols_ = values.begin()->size();

    for (const auto& row : values) {

        if (row.size() != cols_) {
            throw std::invalid_argument(
                "Linhas com dimensoes diferentes."
            );
        }

        data_.insert(data_.end(), row.begin(), row.end());
    }
}

// Acesso mutavel
double& Matrix::operator()(
    std::size_t row,
    std::size_t col
) {

    if (row >= rows_ || col >= cols_) {
        throw std::out_of_range("Indice fora da matriz.");
    }

    return data_[row * cols_ + col];
}

// Acesso constante
double Matrix::operator()(
    std::size_t row,
    std::size_t col
) const {

    if (row >= rows_ || col >= cols_) {
        throw std::out_of_range("Indice fora da matriz.");
    }

    return data_[row * cols_ + col];
}

// Dimensoes
std::size_t Matrix::rows() const {
    return rows_;
}

std::size_t Matrix::cols() const {
    return cols_;
}

// Soma
Matrix Matrix::operator+(const Matrix& other) const {

    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument(
            "Dimensoes incompatíveis para soma."
        );
    }

    Matrix result(rows_, cols_);

    for (std::size_t i = 0; i < data_.size(); i++) {
        result.data_[i] = data_[i] + other.data_[i];
    }

    return result;
}

// Subtracao
Matrix Matrix::operator-(const Matrix& other) const {

    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument(
            "Dimensoes incompatíveis para subtracao."
        );
    }

    Matrix result(rows_, cols_);

    for (std::size_t i = 0; i < data_.size(); i++) {
        result.data_[i] = data_[i] - other.data_[i];
    }

    return result;
}

// Multiplicacao por escalar
Matrix Matrix::operator*(double scalar) const {

    Matrix result(rows_, cols_);

    for (std::size_t i = 0; i < data_.size(); i++) {
        result.data_[i] = data_[i] * scalar;
    }

    return result;
}

// Multiplicacao matricial
Matrix Matrix::matmul(const Matrix& other) const {

    if (cols_ != other.rows_) {
        throw std::invalid_argument(
            "Dimensoes incompatíveis para multiplicacao."
        );
    }

    Matrix result(rows_, other.cols_);

    for (std::size_t i = 0; i < rows_; i++) {

        for (std::size_t k = 0; k < cols_; k++) {

            for (std::size_t j = 0; j < other.cols_; j++) {

                result(i, j) +=
                    (*this)(i, k) * other(k, j);
            }
        }
    }

    return result;
}

// Multiplicacao elemento a elemento
Matrix Matrix::hadamard(const Matrix& other) const {

    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument(
            "Dimensoes incompatíveis para Hadamard."
        );
    }

    Matrix result(rows_, cols_);

    for (std::size_t i = 0; i < data_.size(); i++) {
        result.data_[i] = data_[i] * other.data_[i];
    }

    return result;
}

// Transposta
Matrix Matrix::transpose() const {

    Matrix result(cols_, rows_);

    for (std::size_t i = 0; i < rows_; i++) {

        for (std::size_t j = 0; j < cols_; j++) {

            result(j, i) = (*this)(i, j);
        }
    }

    return result;
}

// Matriz identidade
Matrix Matrix::identity(std::size_t size) {

    Matrix result(size, size);

    for (std::size_t i = 0; i < size; i++) {
        result(i, i) = 1.0;
    }

    return result;
}

// Impressao
void Matrix::print() const {

    for (std::size_t i = 0; i < rows_; i++) {

        for (std::size_t j = 0; j < cols_; j++) {

            std::cout << std::fixed
                      << std::setprecision(4)
                      << (*this)(i, j) << "\t";
        }

        std::cout << "\n";
    }
}
