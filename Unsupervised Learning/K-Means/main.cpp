#include "KMeans.h"
#include "matplotlibcpp.h"
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

namespace plt = matplotlibcpp;

std::vector<Point> readCSV(const std::string& filename) {
    std::vector<Point> points;
    std::ifstream file(filename);
    std::string line;
    while (std::getline(file, line)) {
        std::stringstream linestream(line);
        std::string x, y;
        std::getline(linestream, x, ',');
        std::getline(linestream, y, ',');
        points.push_back(Point{std::stod(x), std::stod(y)});
    }
    return points;
}

void plotData(const std::vector<Point>& points, const std::vector<Point>& centroids) {
    std::vector<double> x, y, cx, cy;
    for (const auto& p : points) {
        x.push_back(p.x);
        y.push_back(p.y);
    }
    for (const auto& c : centroids) {
        cx.push_back(c.x);
        cy.push_back(c.y);
    }

    plt::scatter(x, y, 10);
    plt::scatter(cx, cy, 100, {{"color", "red"}}); 
    plt::show();
}

int main() {
    std::string filename = "/mnt/data/data.csv";

    std::vector<Point> data = readCSV(filename);

    int k = 2;
    KMeans kmeans(k, data);
    kmeans.run(100);

    std::vector<Point> centroids = kmeans.getCentroids();

    plotData(data, centroids);

    return 0;
}
