#pragma once

#include "../datastructures/matrix.hpp"
#include <string>

namespace NN {

using datastruct::Matrix;
using std::string;

const int INPUT_SIZE = 28 * 28;

// A single instance of a training / testing example, a 28 x 28 grayscale
// image, and its corresponding (correct) label.
typedef struct {
    unsigned char data[INPUT_SIZE];
    unsigned char label;
} Example;

// This function is specifically tuned to load the MNIST database of handwritten
// digits (http://yann.lecun.com/exdb/mnist/)
void load_dataset(
        Matrix<unsigned char>& images,
        Matrix<unsigned char>& labels,
        const char *image_file_name,
        const char *label_file_name);

void assert(bool flag, string msg, int code); 
 

} // namespace NN

