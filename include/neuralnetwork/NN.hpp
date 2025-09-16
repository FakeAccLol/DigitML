#pragma once

#include "dataset.hpp"
#include <vector>

namespace NN {

using datastruct::Matrix;
using std::vector;

const int HIDDEN_SIZE = 15;
const int OUTPUT_SIZE = 10;


class NeuralNetwork {
private:
  Matrix<double> weights1 = Matrix<double>(HIDDEN_SIZE, INPUT_SIZE);
  Matrix<double> weights2 = Matrix<double>(OUTPUT_SIZE, HIDDEN_SIZE);

	Matrix<double> weight_init(double max_weight, unsigned int width, unsigned int height);

  vector<double> feed_forward(const vector<double>& input, const Matrix<double>& weights);

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
  unsigned int compute(const Example& e);

  vector<double> sigmoid(const vector<double>& x);
  vector<double> bent_identity(const vector<double>& x);
  vector<double> sigmoid_prime(const vector<double>& x);
  vector<double> isru(const vector<double>& x);
  vector<double> isru_prime(const vector<double>& x);
};

} // namespace NN
