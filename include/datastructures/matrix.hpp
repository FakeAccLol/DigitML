#pragma once

#include <vector>

namespace datastruct {

using std::vector;

// TODO optimize possibly with valarrays instead?
// TODO make everything const correct?

template <typename T> class Matrix {
    private:
        vector<vector<T> > m_data;
        unsigned int m_cols, m_rows;
    public:
        // Standard constructor
        Matrix(unsigned int rows, unsigned int cols);
        // Standard constructor with initialization value
        Matrix(const unsigned int rows, const unsigned int cols, const double init_val);
        // Copy constructor
        Matrix(const Matrix<T>& rhs);
        // Column vector copy constructor
        Matrix(const vector<T>& rhs);
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
