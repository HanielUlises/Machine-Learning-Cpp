#ifndef LOGISTIC_REGRESSION_H
#define LOGISTIC_REGRESSION_H

#include <vector>

class LogisticRegression {
public:
    LogisticRegression(double learningRate = 0.01, int maxIterations = 1000);
    void fit(const std::vector<std::vector<double>>& X, const std::vector<int>& y);
    std::vector<int> predict(const std::vector<std::vector<double>>& X) const;
    double predict_proba(const std::vector<double>& x) const;

private:
    std::vector<double> weights;
    double learningRate;
    int maxIterations;

    double sigmoid(double z) const;
    std::vector<double> gradient(const std::vector<std::vector<double>>& X, const std::vector<int>& y) const;
    void updateWeights(const std::vector<std::vector<double>>& X, const std::vector<int>& y);
};

#endif // LOGISTIC_REGRESSION_H
