#include "PCA.h"

PCA::PCA(int num_components) : num_components(num_components) {}

void PCA::fit(const Eigen::MatrixXd& data) {
    mean = data.colwise().mean();
    Eigen::MatrixXd centered_data = data.rowwise() - mean.transpose();
    Eigen::MatrixXd covariance_matrix = (centered_data.adjoint() * centered_data) / double(data.rows() - 1);

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen_solver(covariance_matrix);
    Eigen::MatrixXd eigen_vectors = eigen_solver.eigenvectors();
    Eigen::VectorXd eigen_values = eigen_solver.eigenvalues();

    std::vector<std::pair<double, Eigen::VectorXd>> eigen_pairs;
    for (int i = 0; i < eigen_values.size(); ++i) {
        eigen_pairs.push_back(std::make_pair(eigen_values[i], eigen_vectors.col(i)));
    }

    std::sort(eigen_pairs.begin(), eigen_pairs.end(),
              [](const std::pair<double, Eigen::VectorXd>& a, const std::pair<double, Eigen::VectorXd>& b) {
                  return a.first > b.first;
              });

    components = Eigen::MatrixXd(num_components, data.cols());
    for (int i = 0; i < num_components; ++i) {
        components.row(i) = eigen_pairs[i].second.transpose();
    }
}

Eigen::MatrixXd PCA::transform(const Eigen::MatrixXd& data) {
    Eigen::MatrixXd centered_data = data.rowwise() - mean.transpose();
    return centered_data * components.transpose();
}

Eigen::MatrixXd PCA::inverse_transform(const Eigen::MatrixXd& transformed_data) {
    return (transformed_data * components).rowwise() + mean.transpose();
}

Eigen::MatrixXd PCA::get_components() {
    return components;
}

Eigen::VectorXd PCA::get_mean() {
    return mean;
}
