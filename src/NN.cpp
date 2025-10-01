#include <cstdlib>
#include <random>
#include <cassert>
#include <algorithm>
#include <stdexcept>
#include <limits> // Добавлено для std::numeric_limits
#include <cmath> // Добавлено для sqrt.

// TODO valarray?
std::vector<double> operator-(
        const std::vector<double>& lhs,
        const std::vector<double>& rhs) {
    if (lhs.size() != rhs.size()) {
        throw std::runtime_error("std::vector::operator-: Inconsistent size");
    }
    std::vector<double> result(lhs.size());
    for (unsigned int i = 0; i < lhs.size(); i++)
        result[i] = lhs[i] - rhs[i];
    return result;
}

NeuralNetwork::NeuralNetwork() {
    // Используем инициализацию Хе для ReLU.
    // Максимальный вес = sqrt(2 / (количество входных нейронов))
    const double max_weight_1 = sqrt(2.0 / (INPUT_SIZE + 1));
    const double max_weight_2 = sqrt(2.0 / (HIDDEN_SIZE + 1));
    
    weights1 = weight_init(max_weight_1, HIDDEN_SIZE, INPUT_SIZE + 1);
    weights2 = weight_init(max_weight_2, OUTPUT_SIZE, HIDDEN_SIZE + 1);
}

// Performs one training iteration using the data in images and labels
void NeuralNetwork::train(
        const unsigned int iterations,
        const Matrix<unsigned char>& images,
        const Matrix<unsigned char>& labels) {
    // The learning rate
    // Уменьшаем скорость обучения, чтобы предотвратить расхождение (divergence)
    const double alpha = 0.0001;

    for (unsigned int i = 0; i < iterations; ++i) {
      // Initialize the gradient matrices to 0
      Matrix<double> gradient_1(weights1.rows(), weights1.cols(), 0.0);
      Matrix<double> gradient_2(weights2.rows(), weights2.cols(), 0.0);
      double cost = 0.0;


      // Perform one step of gradient descent
      compute_gradients_and_cost(images, labels, gradient_1, gradient_2, cost);

      printf("Cost after %d iteration(s): %f\n", i+1, cost);

      weights1 = weights1 - gradient_1 * alpha;
      weights2 = weights2 - gradient_2 * alpha;
    }
}

std::vector<double> vectorize_label(unsigned char label) {
    std::vector<double> result(10, 0.0);
    result[(unsigned int)label] = 1.0;
    return result;
}

// Fixed log function to prevent log(0) errors
std::vector<double> log_safe(const std::vector<double>& vec) {
    std::vector<double> result(vec.size());
    // Эпсилон для предотвращения log(0)
    const double epsilon = std::numeric_limits<double>::epsilon();
    for (unsigned int i = 0; i < result.size(); ++i) {
        // Ограничиваем значение, чтобы оно не было меньше эпсилон
        double value = std::max(vec[i], epsilon);
        result[i] = log(value);
    }
    return result;
}

// Calculates the current cost and uses backpropagation to compute the gradients
void NeuralNetwork::compute_gradients_and_cost(
        const Matrix<unsigned char>& images,
        const Matrix<unsigned char>& labels,
        Matrix<double>& gradient_1,
        Matrix<double>& gradient_2,
        double& cost) {
    // The number of examples
    unsigned int m = images.rows();
    // The regularization parameter
    const double lambda = 1.0;
    // The small constant for numerical stability
    const double epsilon = 1e-9;

    for (unsigned int i = 0; i < m; ++i) {
        std::vector<double> first_layer(images[i].begin(), images[i].end());
        // The bias value
        first_layer.insert(first_layer.begin(), 1.0);

        // Используем новую функцию для скрытого слоя
        std::vector<double> hidden_layer_activated = feed_forward_hidden(first_layer, weights1);
        // The bias value
        hidden_layer_activated.insert(hidden_layer_activated.begin(), 1.0);

        // Используем новую функцию для выходного слоя
        std::vector<double> last_layer_activated = feed_forward_output(hidden_layer_activated, weights2);

        const std::vector<double> vector_outcome = vectorize_label(labels[i][0]);
        const std::vector<double> ones(10, 1.0);

        // Используем безопасный логарифм для предотвращения log(0)
        const double first_part = ((Matrix<double>(vector_outcome) * (double)(-1)).transpose() * log_safe(last_layer_activated))[0];
        // Также добавляем эпсилон здесь для численной стабильности
        std::vector<double> ones_minus_last_layer(ones.size());
        for(size_t j = 0; j < ones.size(); ++j) {
            ones_minus_last_layer[j] = ones[j] - last_layer_activated[j];
        }
        const double second_part = ((Matrix<double>(ones - vector_outcome)).transpose() * log_safe(ones_minus_last_layer))[0];
        
        // unregularized part of the error for this training example
        cost += 1.0/m * (first_part - second_part);

        // Backpropagation
        const Matrix<double> d3(last_layer_activated - vector_outcome);
        
        // The derivative of the hidden layer activation function needs to be applied
        #if defined RELU
            // Для ReLU, производная 1 для положительных значений и 0 для отрицательных.
            Matrix<double> d2((weights2.transpose() * d3).hadamard(Matrix<double>(relu_prime(hidden_layer_activated))));
        #else
            const std::vector<double> ones2(HIDDEN_SIZE + 1, 1);
            Matrix<double> d2((weights2.transpose() * d3).hadamard(Matrix<double>(hidden_layer_activated)).hadamard(Matrix<double>(ones2 - hidden_layer_activated)));
        #endif

        gradient_2 += d3 * Matrix<double>(hidden_layer_activated).transpose();

        // Remove the term in d2 corresponding to the bias node in the hidden layer
        std::vector<double> d2_vec(HIDDEN_SIZE);
        for (unsigned int i = 0; i < HIDDEN_SIZE; ++i) d2_vec[i] = d2[i+1][0];

        gradient_1 += Matrix<double>(d2_vec) * Matrix<double>(first_layer).transpose();
    }

    // Make copies of the weights matrices with the bias weights set to 0 so they're not regularized
    Matrix<double> temp_weights1(weights1);
    for (unsigned int i = 0; i < temp_weights1.rows(); ++i) {
      temp_weights1[i][0] = 0.0;
    }
    Matrix<double> temp_weights2(weights2);
    for (unsigned int i = 0; i < temp_weights2.rows(); ++i) {
      temp_weights2[i][0] = 0.0;
    }

    // Adjust the gradients
    gradient_1 = gradient_1 /((double)m) + temp_weights1*(lambda/m);
    gradient_2 = gradient_2 /((double)m) + temp_weights2*(lambda/m);

    // Regularize the cost
    double regularizationCost = 0.0;
    for (unsigned int i = 0; i < weights1.rows(); ++i) {
      // Don't regularize the bias terms
      for (unsigned int j = 1; j < weights1.cols(); ++j) {
        regularizationCost += weights1[i][j] * weights1[i][j];
      }
    }
    for (unsigned int i = 0; i < weights2.rows(); ++i) {
      // Don't regularize the bias terms
      for (unsigned int j = 1; j < weights2.cols(); ++j) {
        regularizationCost += weights2[i][j] * weights2[i][j];
      }
    }

    cost += lambda/(2*m) * regularizationCost;
}

inline std::vector<double> NeuralNetwork::feed_forward_hidden(
        const std::vector<double>& input,
        const Matrix<double>& weights) const {
    #if defined RELU
        return relu(weights * input);
    #elif defined PERS
        return bent_identity(weights * input);
    #else
        return sigmoid(weights * input);
    #endif
}

inline std::vector<double> NeuralNetwork::feed_forward_output(
        const std::vector<double>& input,
        const Matrix<double>& weights) const {
    // В выходном слое всегда используем Softmax
    std::vector<double> z = weights * input;
    
    // Численная стабильность: вычитаем максимальное значение
    double max_z = *std::max_element(z.begin(), z.end());
    
    std::vector<double> exp_z(z.size());
    double sum_exp_z = 0.0;
    for (size_t i = 0; i < z.size(); ++i) {
        exp_z[i] = exp(z[i] - max_z);
        sum_exp_z += exp_z[i];
    }
    
    std::vector<double> result(z.size());
    for (size_t i = 0; i < z.size(); ++i) {
        result[i] = exp_z[i] / sum_exp_z;
    }
    return result;
}

Matrix<double> NeuralNetwork::weight_init(double max_weight, unsigned int width, unsigned int height) const {
    std::random_device rd;
    std::mt19937 e2(rd());
    std::uniform_real_distribution<> dist(-max_weight, max_weight);

    Matrix<double> weights(width, height);
    for (int i = 0; i < weights.rows(); i++)
        for (int j = 0; j < weights.cols(); j++)
	       weights[i][j] = dist(e2);

    return weights;
}

unsigned int NeuralNetwork::compute(const Example& e) const {
    std::vector<double> first_layer(e.data, e.data + INPUT_SIZE);
    // The bias value
    first_layer.insert(first_layer.begin(), 1.0);
    std::vector<double> hidden_layer_activated, last_layer_activated;

    // Используем новую функцию для скрытого слоя
    hidden_layer_activated = feed_forward_hidden(first_layer, weights1);
    // The bias value
    hidden_layer_activated.insert(hidden_layer_activated.begin(), 1.0);

    // Используем новую функцию для выходного слоя
    last_layer_activated = feed_forward_output(hidden_layer_activated, weights2);

    unsigned int max_val_index = 0;
    for (int i = 1; i < 10; i++) {
	    if (last_layer_activated[i] > last_layer_activated[max_val_index])
	       max_val_index = i;
    }

    return max_val_index;
}

// TODO parallelize (now its really easy to valarray)
std::vector<double> NeuralNetwork::sigmoid(const std::vector<double>& x) const {
    std::vector<double> result(x.size());
    for (unsigned int i = 0; i < x.size(); i++)
        result[i] = 1 / (1 + exp(-x[i]));
    return result;
}

std::vector<double> NeuralNetwork::bent_identity(const std::vector<double>& x) const {
    std::vector<double> result(x.size());
    for (unsigned int i = 0; i < x.size(); i++)
        result[i] = (sqrt(pow(x[i], 2) + 1) - 1) / 2 + x[i];
    return result;
}

std::vector<double> NeuralNetwork::sigmoid_prime(const std::vector<double>& x) const {
    std::vector<double> result(x.size());
    for (unsigned int i = 0; i < result.size(); i++) {
        const double t = exp(x[i]);
        result[i] = t / ((1 + t) * (1 + t));
    }
    return result;
}

// New ReLU activation function
std::vector<double> NeuralNetwork::isru(const std::vector<double>& x) const {
  double alpha = 1.0;
  std::vector<double> result(x.size());
  for (unsigned int i = 0; i < x.size(); i++) {
    double xi = x[i];
    result[i] = xi / std::sqrt(1.0 + alpha * xi * xi);
  }
  return result;
}

// New ReLU derivative function
std::vector<double> NeuralNetwork::isru_prime(const std::vector<double>& x) const {
  double alpha = 1.0;
  std::vector<double> result(x.size());
  for (unsigned int i = 0; i < x.size(); i++) {
    double xi = x[i];
    double denominator = 1.0 + alpha * xi * xi;
    result[i] = 1.0 / std::pow(denominator, 1.5);
  }
  return result;
}
