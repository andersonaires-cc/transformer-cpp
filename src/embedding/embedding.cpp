
#include "embedding/embedding.hpp"

#include <cmath>
#include <random>
#include <stdexcept>

void Vocabulary::addToken(const std::string& token) {
    if (token.empty()) {
        throw std::invalid_argument("Token vazio.");
    }

    if (tokenToIndex_.find(token) != tokenToIndex_.end()) {
        return;
    }

    const std::size_t index = indexToToken_.size();

    tokenToIndex_[token] = index;
    indexToToken_.push_back(token);
}

std::size_t Vocabulary::getIndex(
    const std::string& token
) const {
    auto it = tokenToIndex_.find(token);

    if (it == tokenToIndex_.end()) {
        throw std::out_of_range("Token nao encontrado: " + token);
    }

    return it->second;
}

const std::string& Vocabulary::getToken(
    std::size_t index
) const {
    if (index >= indexToToken_.size()) {
        throw std::out_of_range("Indice de token invalido.");
    }

    return indexToToken_[index];
}

std::size_t Vocabulary::size() const {
    return indexToToken_.size();
}

std::vector<std::size_t> Vocabulary::encode(
    const std::vector<std::string>& tokens
) const {
    std::vector<std::size_t> indices;

    indices.reserve(tokens.size());

    for (const auto& token : tokens) {
        indices.push_back(getIndex(token));
    }

    return indices;
}

// --------------------------------------------------
// INPUT EMBEDDING
// --------------------------------------------------

InputEmbedding::InputEmbedding(
    std::size_t vocabularySize,
    std::size_t embeddingDimension
)
    : weights_(vocabularySize, embeddingDimension),
      vocabularySize_(vocabularySize),
      embeddingDimension_(embeddingDimension) {

    if (vocabularySize == 0 || embeddingDimension == 0) {
        throw std::invalid_argument(
            "Dimensoes do embedding devem ser maiores que zero."
        );
    }

    initializeWeights();
}

void InputEmbedding::initializeWeights() {
    // Inicializacao aleatoria Xavier/Glorot normal.
    const double stddev = std::sqrt(
        1.0 / static_cast<double>(embeddingDimension_)
    );

    std::random_device rd;
    std::mt19937 generator(rd());

    std::normal_distribution<double> distribution(
        0.0,
        stddev
    );

    for (std::size_t i = 0; i < vocabularySize_; ++i) {
        for (std::size_t j = 0; j < embeddingDimension_; ++j) {
            weights_(i, j) = distribution(generator);
        }
    }
}

Matrix InputEmbedding::forward(
    const std::vector<std::size_t>& tokenIndices
) const {

    Matrix output(
        tokenIndices.size(),
        embeddingDimension_
    );

    for (std::size_t i = 0; i < tokenIndices.size(); ++i) {

        const std::size_t tokenIndex = tokenIndices[i];

        if (tokenIndex >= vocabularySize_) {
            throw std::out_of_range(
                "Indice fora do vocabulario."
            );
        }

        for (std::size_t j = 0; j < embeddingDimension_; ++j) {
            output(i, j) = weights_(tokenIndex, j);
        }
    }

    return output;
}

const Matrix& InputEmbedding::weights() const {
    return weights_;
}

std::size_t InputEmbedding::embeddingDimension() const {
    return embeddingDimension_;
}
