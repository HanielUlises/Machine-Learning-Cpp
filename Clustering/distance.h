#include <iostream>
#include <string>
#include <vector>
#include <cmath>

class Distance{
    public:
        Distance(const std::string& name) : _name(name) {}
        ~Distance() {}

        virtual double operator() (const std::vector<double>& x, const std::vector<double> &y) const = 0;
        const std::string& name() {return _name;}
    private:
        std::string _name;
};

class Euclidean_Distance : public Distance{
    public:
        Euclidean_Distance () : Distance("Euclidean Distance") {}

        double operator () (const std::vector<double>& x, const std::vector<double>& y) const{
            if(x.size() != y.size()){
                return -1.0;
            }
            double distance_sum = 0.0f;
            
            for(size_t i = 0; i < x.size(); i++){
                distance_sum += (x[i] - y[i]) * (x[i] - y [i]);
            }
            return sqrt(distance_sum);
        }
};