#ifndef PLOTTING_H
#define PLOTTING_H

#include <vector>
#include <string>
#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;

void plot_regression_line(const std::vector<double>& X, const std::vector<double>& y, const std::vector<double>& y_pred, const std::string& title) {
    // Original data 
    plt::scatter(X, y, 10, {{"color", "red"}}); 
    
    // Regression line
    plt::plot(X, y_pred);
    
    plt::title(title);
    plt::xlabel("X");
    plt::ylabel("y");
    
    plt::show();
}

#endif // PLOTTING_H
