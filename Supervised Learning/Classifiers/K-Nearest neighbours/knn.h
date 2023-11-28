#ifndef KNN_H
#define KNN_H

#include <vector>
#include <cmath>
#include <algorithm>

struct Point{
    double x;
    double y;
};

class KNN{
    public:
        KNN(const std::vector<Point>& trainingData);
        Point classify (const Point& p, int k);
    private:
        double calculateDistance (const Point& p1, const Point& p2);
        std::vector<Point> trainingData;
};

#endif