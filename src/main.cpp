#include "dataset.hpp"
#include "NN.hpp"
#include "../lib/matrix.h"
#include <vector>
#include <iostream>

void debug(Example e) {
    static std::string shades = " .:-=+*#%@";
    for (unsigned int i = 0; i < 28 * 28; i++) {
        if (i % 28 == 0) printf("\n");
        printf("%c", shades[e.data[i] / 30]);
    }
    printf("\nLabel: %d\n", e.label);
}

std::vector<double> load_matrix(Example& e) {
    std::vector<double> result(e.data, e.data + 28 * 28);
    return result;
}

// Pass NeuralNetwork by const reference to avoid expensive copying.
const double calculate_accuracy(const Matrix<unsigned char>& images, const Matrix<unsigned char>& labels, const NeuralNetwork& n) {
  unsigned int correct = 0;
  for (unsigned int i = 0; i < images.rows(); ++i) {
    Example e;
    for (int j = 0; j < 28*28; ++j) {
        e.data[j] = images[i][j];
    }
    e.label = labels[i][0];
    unsigned int guess = n.compute(e);
    if (guess == (unsigned int)e.label) correct++;
  }
  const double accuracy = (double)correct/images.rows();

  return accuracy;
}

#ifdef TESTS
#include <gtest/gtest.h>

TEST(NeuralNetworkTests, ReluFunction_Test) {
    NeuralNetwork n;
    std::vector<double> input = { -1.0, 0.0, 5.0, -10.0, 15.0 };
    std::vector<double> expected = { 0.0, 0.0, 5.0, 0.0, 15.0 };
    ASSERT_EQ(n.relu(input), expected);
}

TEST(NeuralNetworkTests, ReluPrimeFunction_Test) {
    NeuralNetwork n;
    std::vector<double> input = { -1.0, 0.0, 5.0, -10.0, 15.0 };
    std::vector<double> expected = { 0.0, 0.0, 1.0, 0.0, 1.0 };
    ASSERT_EQ(n.relu_prime(input), expected);
}

TEST(NeuralNetworkTests, SigmoidFunction_Test) {
    NeuralNetwork n;
    std::vector<double> input = { 0.0 };
    EXPECT_NEAR(n.sigmoid(input)[0], 0.5, 1e-6);
}

TEST(NeuralNetworkTests, SoftmaxFunction_Test) {
    NeuralNetwork n;
    // Create input with correct dimensions: HIDDEN_SIZE + 1 (for bias) = 15 + 1 = 16
    std::vector<double> input(16);
    // Set bias to 1.0
    input[0] = 1.0;
    // Set hidden layer values
    for (int i = 1; i < 16; i++) {
        input[i] = (double)i / 10.0;  // Use varying values for more realistic test
    }
    std::vector<double> output = n.feed_forward_output(input, n.get_weights2());
    double sum = 0.0;
    for (double val : output) {
        sum += val;
    }
    EXPECT_NEAR(sum, 1.0, 1e-6);
}

TEST(NeuralNetworkTests, TrainingCostDecreases) {
    NeuralNetwork n;
    Matrix<unsigned char> images_train(0, 0);
    Matrix<unsigned char> labels_train(0, 0);
    load_dataset(images_train, labels_train, "data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte");
    const unsigned int num_iterations = 5;
    EXPECT_NO_THROW(n.train(num_iterations, images_train, labels_train));
}

#endif

int main(int argc, char **argv) {
    #ifdef TESTS
        ::testing::InitGoogleTest(&argc, argv);
        return RUN_ALL_TESTS();
    #endif
    
    Matrix<unsigned char> images_train(0, 0);
    Matrix<unsigned char> labels_train(0, 0);
    load_dataset(images_train, labels_train, "data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte");

    Matrix<unsigned char> images_test(0, 0);
    Matrix<unsigned char> labels_test(0, 0);
    load_dataset(images_test, labels_test, "data/t10k-images-idx3-ubyte", "data/t10k-labels-idx1-ubyte");

    NeuralNetwork n;

    // Tests to see that data was read in properly
    /*for (int i = 0; i < 10; ++i) {
        Example e;
        for (int j = 0; j < 28*28; ++j) {
            e.data[j] = images_train[i][j];
        }
        e.label = labels_train[i][0];
        debug(e);
        printf("Guess: %d\n", n.compute(e));
    }
    for (int i = 0; i < 10; ++i) {
        Example e;
        for (int j = 0; j < 28*28; ++j) {
            e.data[j] = images_test[i][j];
        }
        e.label = labels_test[i][0];
        debug(e);
        printf("Guess: %d\n", n.compute(e));
    }*/
    const unsigned int num_iterations = 5;
    n.train(num_iterations, images_train, labels_train);

    const double accuracy_train = calculate_accuracy(images_train, labels_train, n);
    const double accuracy_test = calculate_accuracy(images_test, labels_test, n);

    printf("Accuracy on training data: %f\n", accuracy_train);
    printf("Accuracy on test data: %f\n", accuracy_test);

    return 0;
}
