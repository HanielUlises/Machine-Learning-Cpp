#pragma once
#include <stdexcept>
#include <Eigen/SVD>

#include "SVD.h"

inline void SVDReducer::fit(const Eigen::MatrixXd& X) {
    using namespace Eigen;

    if (X.rows() == 0 || X.cols() == 0)
        throw std::invalid_argument("Input matrix must be non-empty.");

    /*
      Note:
        For PCA, X should first be mean-centered. This class performs
        pure SVD to stay general, letting users choose preprocessing.
    */

    BDCSVD<MatrixXd> svd(X, ComputeThinU | ComputeThinV);

    U_ = svd.matrixU();              // m×r
    S_ = svd.singularValues();       // length r
    V_ = svd.matrixV();              // n×r
}

inline Eigen::MatrixXd
SVDReducer::transform(std::size_t k) const {
    if (k > S_.size())
        throw std::invalid_argument("k exceeds available singular vectors.");

    // Z = U_k Σ_k
    return U_.leftCols(k) * S_.head(k).asDiagonal();
}

inline Eigen::MatrixXd
SVDReducer::transform_new(const Eigen::MatrixXd& X_new,
                          std::size_t k) const {
    if (k > S_.size())
        throw std::invalid_argument("k exceeds available singular vectors.");

    // Projection for unseen samples: X_new V_k
    return X_new * V_.leftCols(k);
}

inline Eigen::MatrixXd
SVDReducer::reconstruct(const Eigen::MatrixXd& Z,
                        std::size_t k) const {
    if (k > S_.size())
        throw std::invalid_argument("k exceeds available singular vectors.");

    // Reconstruction: X_k = Z V_k^T
    return Z * V_.leftCols(k).transpose();
}
