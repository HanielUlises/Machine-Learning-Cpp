#include "KMeans.h"
#include "matplotlibcpp.h"
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <cctype>

namespace plt = matplotlibcpp;

// Utility function to trim whitespace from both ends of a string
static inline void trim(std::string &s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) {
        return !std::isspace(ch);
    }));
    s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) {
        return !std::isspace(ch);
    }).base(), s.end());
}

std::vector<Point> readCSV(const std::string& filename) {
    std::vector<Point> points;
    std::ifstream file(filename);
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue; // Skip empty lines

        std::stringstream linestream(line);
        std::string x, y;
        if (!std::getline(linestream, x, ',') || !std::getline(linestream, y, ',')) continue; // Skip lines without proper format

        trim(x);
        trim(y);

        try {
            points.push_back(Point{std::stod(x), std::stod(y)});
        } catch (const std::invalid_argument& e) {
            // Handle the exception, e.g., skip the line or log an error
            continue;
        }
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
    std::string filename = "data.csv";

    std::vector<Point> data = readCSV(filename);

    int k = 3;
    KMeans kmeans(k, data);
    kmeans.run(100);

    std::vector<Point> centroids = kmeans.getCentroids();

    plotData(data, centroids);

    return 0;
}