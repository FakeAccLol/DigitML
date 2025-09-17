#pragma once

#include <cstddef>
#include <vector>
#include <valarray>

namespace datastruct {

using std::vector;
using std::valarray;

// TODO make everything const correct?

template <typename T> class Matrix {
private:
  valarray<T> m_data;
  size_t m_cols;
  size_t m_rows;

public:
  // Constructors
  Matrix(size_t rows, size_t cols);
  
  Matrix(size_t rows, size_t cols, const T& init_val);
  
  // Copy/move constructors
  Matrix(const Matrix<T>& rhs) = default;
  Matrix(const vector<T>& rhs) = default;
  Matrix& operator=(const Matrix&) = default;
  Matrix& operator=(Matrix&&) = default;

  // Destructor
  virtual ~Matrix();

  // Standard mathematical operations
  Matrix<T>& operator=(const Matrix<T>& rhs);

  Matrix<T> operator+(const Matrix<T>& rhs) const;
  Matrix<T>& operator+=(const Matrix<T>& rhs);
  Matrix<T> operator-(const Matrix<T>& rhs) const;
  Matrix<T>& operator-=(const Matrix<T>& rhs);
  Matrix<T> operator*(const Matrix<T>& rhs) const;
  Matrix<T>& operator*=(const Matrix<T>& rhs);
  Matrix<T> transpose() const;

  Matrix<T> operator+(const T& rhs) const;
  Matrix<T> operator-(const T& rhs) const;
  Matrix<T> operator*(const T& rhs) const;
  Matrix<T> operator/(const T& rhs) const;

  Matrix<T> hadamard(const Matrix<T>& rhs) const;
  Matrix<T> kronecker(const Matrix<T>& rhs) const;
  Matrix<T> concat(const Matrix<T>& rhs) const;

  vector<T> operator*(const vector<T>& rhs) const;
  vector<T> diag_vec();

  // Included this so users can do m[0][0] rather than m(0, 0)
  vector<T>& operator[] (const unsigned int x);
  const vector<T>& operator[] (const unsigned int x) const;

  unsigned int rows() const;
  unsigned int cols() const;

  // DEBUG
  void debug() const;
};
  
} //namespace project

#include "matrix.tpp"
