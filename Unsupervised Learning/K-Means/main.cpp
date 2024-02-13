#include "KMeans.h"
#include "matplotlibcpp.h"
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <cctype>

namespace plt = matplotlibcpp;

static inline void trim(std::string &s) {
    // Function to remove leading and trailing whitespaces from a string
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) {
        return !std::isspace(ch);
    }));
    s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) {
        return !std::isspace(ch);
    }).base(), s.end());
}

std::vector<Point> readCSV(const std::string& filename) {
    std::vector<Point> points;  // Store parsed points from the CSV
    std::ifstream file(filename);  // Open file stream for reading
    std::string line;  // String to hold each line of the file

    // Iterate through each line of the file
    while (std::getline(file, line)) {
        if (line.empty()) continue;

        std::stringstream linestream(line);  // Create string stream to process the line
        std::string x, y; // Strings to hold x and y values

        // Extract x and y values from the line separated by comma
        if (!std::getline(linestream, x, ',') || !std::getline(linestream, y, ',')) continue; 

        trim(x); // Remove leading and trailing whitespaces from x
        trim(y); // Remove leading and trailing whitespaces from y

        try {
            points.push_back(Point{std::stod(x), std::stod(y)}); // Convert x and y to double and add to points
        } catch (const std::invalid_argument& e) {
            continue; // Skip invalid data
        }
    }
    return points; // Return vector of parsed points
}

void plotData(const std::vector<Point>& points, const std::vector<Point>& centroids) {
    // Plot data points and centroids using matplotlibcpp
    std::vector<double> x, y, cx, cy; // Separate vectors for x and y coordinates of points and centroids
    for (const auto& p : points) {
        x.push_back(p.x); // Extract x coordinate of each point
        y.push_back(p.y); // Extract y coordinate of each point
    }
    for (const auto& c : centroids) {
        cx.push_back(c.x); // Extract x coordinate of each centroid
        cy.push_back(c.y); // Extract y coordinate of each centroid
    }

    // Scatter plot data points and centroids
    plt::scatter(x, y, 10); // Plot data points with size 10
    plt::scatter(cx, cy, 100, {{"color", "red"}}); // Plot centroids with size 100 and red color
    plt::show(); // Display the plot
}


int main() {
    // I ain't commenting none of this
    std::string filename = "data.csv";

    std::vector<Point> data = readCSV(filename);

    int k = 3;
    KMeans kmeans(k, data);
    kmeans.run(100);

    std::vector<Point> centroids = kmeans.getCentroids();

    plotData(data, centroids);

    return 0;
}