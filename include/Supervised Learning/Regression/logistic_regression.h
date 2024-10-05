#ifndef LOGISTIC_REGRESSION_H
#define LOGISTIC_REGRESSION_H

#include <vector>

class LogisticRegression {
public:
    LogisticRegression(double learning_rate = 0.01, int max_iterations = 1000);
    void fit(const std::vector<std::vector<double>>& input_features, const std::vector<int>& target_values);
    std::vector<int> predict(const std::vector<std::vector<double>>& input_features) const;
    double predict_proba(const std::vector<double>& input_vector) const;

private:
    std::vector<double> weights;
    double learning_rate;
    int max_iterations;

    double sigmoid(double z) const;
    std::vector<double> compute_gradient(const std::vector<std::vector<double>>& input_features, const std::vector<int>& target_values) const;
    void update_weights(const std::vector<std::vector<double>>& input_features, const std::vector<int>& target_values);
};

#endif // LOGISTIC_REGRESSION_H
