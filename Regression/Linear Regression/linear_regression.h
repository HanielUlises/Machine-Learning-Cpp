#ifndef LINEAR_REGRESSION_H
#define LINEAR_REGRESSION_H

#include <vector>
#include <string>
#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;

class LinearRegression {
public:
    LinearRegression();
    void fit(const std::vector<double>& X, const std::vector<double>& y);
    std::vector<double> predict(const std::vector<double>& X) const;
    double score(const std::vector<double>& X, const std::vector<double>& y) const;
    void plot_regression_line(const std::vector<double>& X, const std::vector<double>& y, const std::string& title = "Regression Line") const;
    
    double get_slope() const { return slope; }
    double get_intercept() const { return intercept; }

private:
    double slope;
    double intercept;
    double mean(const std::vector<double>& v) const;
    double covariance(const std::vector<double>& X, const std::vector<double>& y) const;
    double variance(const std::vector<double>& v) const;
};

#endif // LINEAR_REGRESSION_H
