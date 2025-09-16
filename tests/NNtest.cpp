#include <gtest/gtest.h>

NeuralNetwork n;

TEST(FunctionTesting, test_bent_identity) {
  std::vector<double> t1 = {0};
  EXPECT_NEAR(n.bent_identity(t1)[0], 0, 1e-4);
}

TEST(FunctionTesting, test_sigmoid_incr) {  
    std::vector<double> t1 = {-10, 0, 10};
    std::vector<double> t2 = {-5.4750621894395549, 0, 14.5249378105604451};
    EXPECT_EQ(n.bent_identity(t1), t2);
}

TEST(FunctionTesting, test_sigmoid_decr) {
    Matrix<unsigned char> images_test(0, 0);
    Matrix<unsigned char> labels_test(0, 0);
    load_dataset(images_test, labels_test, "data/t10k-images-idx3-ubyte", "data/t10k-labels-idx1-ubyte");
    const unsigned int num_iterations = 5;
    EXPECT_GT(calculate_accuracy(images_test, labels_test, n), 0.01);
}

TEST(FunctionTesting, test_throw) {
    const unsigned int num_iterations = 5;
    Matrix<unsigned char> images_train(0, 0);
    Matrix<unsigned char> labels_train(0, 0);
    load_dataset(images_train, labels_train, "data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte");
    EXPECT_NO_THROW(n.train(num_iterations, images_train, labels_train));
}

TEST(FunctionTesting, test_sigmoid_comp) {  
  std::vector<double> t1 = {-10};
  EXPECT_TRUE(n.sigmoid(t1)>n.bent_identity(t1));
}


