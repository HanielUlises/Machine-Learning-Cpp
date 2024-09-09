#ifndef LDA_H
#define LDA_H

#include <vector>
#include <Eigen/Dense>

class LDA {
private:
    Eigen::MatrixXd meanVectors;
    Eigen::MatrixXd withinClassScatterMatrix;
    Eigen::MatrixXd betweenClassScatterMatrix;
    Eigen::MatrixXd projectionMatrix;
    int nClasses;

    void computeMeanVectors(const Eigen::MatrixXd& X, const Eigen::VectorXi& y);
    void computeScatterMatrices(const Eigen::MatrixXd& X, const Eigen::VectorXi& y);
    void computeProjectionMatrix();

public:
    LDA();
    void fit(const Eigen::MatrixXd& X, const Eigen::VectorXi& y);
    Eigen::MatrixXd transform(const Eigen::MatrixXd& X) const;
};

#endif // LDA_H
