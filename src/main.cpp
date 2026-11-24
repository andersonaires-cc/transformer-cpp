
#include "embedding/embedding.hpp"

#include <iostream>
#include <vector>
#include <string>

int main() {

    // 1. Criar vocabulario
    Vocabulary vocabulary;

    vocabulary.addToken("eu");
    vocabulary.addToken("gosto");
    vocabulary.addToken("de");
    vocabulary.addToken("cafe");

    std::cout << "Tamanho do vocabulario: "
              << vocabulary.size() << "\n\n";

    // 2. Sequencia de entrada
    std::vector<std::string> sentence = {
        "eu",
        "gosto",
        "de",
        "cafe"
    };

    // 3. Converter tokens em indices
    std::vector<std::size_t> indices =
        vocabulary.encode(sentence);

    std::cout << "Indices dos tokens:\n";

    for (std::size_t index : indices) {
        std::cout << index << " ";
    }

    std::cout << "\n\n";

    // 4. Criar camada de Input Embedding
    const std::size_t embeddingDimension = 4;

    InputEmbedding embedding(
        vocabulary.size(),
        embeddingDimension
    );

    // 5. Gerar representacoes vetoriais
    Matrix output = embedding.forward(indices);

    std::cout << "Matriz de embeddings:\n";

    output.print();

    std::cout << "\nDimensoes da saida: "
              << output.rows()
              << " x "
              << output.cols()
              << "\n";

    return 0;
}
