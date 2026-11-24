
#ifndef EMBEDDING_HPP
#define EMBEDDING_HPP

#include "core/matrix.hpp"

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

class Vocabulary {
public:
    Vocabulary() = default;

    void addToken(const std::string& token);

    std::size_t getIndex(const std::string& token) const;

    const std::string& getToken(std::size_t index) const;

    std::size_t size() const;

    std::vector<std::size_t> encode(
        const std::vector<std::string>& tokens
    ) const;

private:
    std::unordered_map<std::string, std::size_t> tokenToIndex_;
    std::vector<std::string> indexToToken_;
};

class InputEmbedding {
public:
    InputEmbedding(
        std::size_t vocabularySize,
        std::size_t embeddingDimension
    );

    Matrix forward(
        const std::vector<std::size_t>& tokenIndices
    ) const;

    const Matrix& weights() const;

    std::size_t embeddingDimension() const;

private:
    Matrix weights_;
    std::size_t vocabularySize_;
    std::size_t embeddingDimension_;

    void initializeWeights();
};

#endif
