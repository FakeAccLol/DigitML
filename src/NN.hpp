#ifndef NN_HPP
#define NN_HPP

#define INPUT_SIZE (28 * 28)
#define HIDDEN_SIZE 15
#define OUTPUT_SIZE 10

#include "../lib/matrix.h"
#include "dataset.hpp"
#include <cmath>

#define RELU


class NeuralNetwork {
    private:
        Matrix<double> weights1 = Matrix<double>(HIDDEN_SIZE, INPUT_SIZE),
                                weights2 = Matrix<double>(OUTPUT_SIZE, HIDDEN_SIZE);

        Matrix<double> weight_init(double max_weight, unsigned int width, unsigned int height) const;

        // Разделяем feed_forward на два отдельных вызова для разных слоев
        std::vector<double> feed_forward_hidden(
                const std::vector<double>& input,
                const Matrix<double>& weights) const;

    public:
        NeuralNetwork();
        NeuralNetwork(const NeuralNetwork& rhs) = default;
        virtual ~NeuralNetwork() = default;

        void train(
                const unsigned int iterations,
                const Matrix<unsigned char>& images,
                const Matrix<unsigned char>& labels);
        void compute_gradients_and_cost(
                const Matrix<unsigned char>& images,
                const Matrix<unsigned char>& labels,
                Matrix<double>& gradient_1,
                Matrix<double>& gradient_2,
                double& cost);
        unsigned int compute(const Example& e) const;

        std::vector<double> sigmoid(const std::vector<double>& x) const;
        std::vector<double> bent_identity(const std::vector<double>& x) const;
        std::vector<double> sigmoid_prime(const std::vector<double>& x) const;
        std::vector<double> isru(const std::vector<double>& x) const;
        std::vector<double> isru_prime(const std::vector<double>& x) const;

        // Methods needed for testing
        Matrix<double> get_weights2() const { return weights2; }
        std::vector<double> feed_forward_output(
                const std::vector<double>& input,
                const Matrix<double>& weights) const;
};

#include "NN.cpp"

#endif
