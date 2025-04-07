#include <iostream>
#include <string>
#include <vector>
#include <cmath>

// Base class representing a distance metric
class Distance {
public:
    Distance(const std::string& name) : _name(name) {}
    ~Distance() {}

    // Pure virtual operator for calculating the distance between two vectors
    virtual double operator() (const std::vector<double>& x, const std::vector<double>& y) const = 0;

    // Returns the name of the distance metric
    const std::string& name() { return _name; }

private:
    std::string _name;
};

// Derived class for calculating Euclidean distance
class Euclidean_Distance : public Distance {
public:
    Euclidean_Distance() : Distance("Euclidean Distance") {}

    // Computes the Euclidean distance between two vectors
    double operator() (const std::vector<double>& x, const std::vector<double>& y) const {
        // Returns -1 if the vectors are of different sizes
        if (x.size() != y.size()) {
            return -1.0;
        }
        
        double distance_sum = 0.0; 

        // Accumulates the squared differences for each dimension
        for (size_t i = 0; i < x.size(); i++) {
            distance_sum += (x[i] - y[i]) * (x[i] - y[i]);
        }

        // Returns the square root of the sum of squared differences
        return sqrt(distance_sum);
    }
};
