#include <iostream>
#include "HierarchicalClustering.h"

int main() {
    Eigen::MatrixXd data(5, 2);
    data << 1.0, 1.0,
            1.5, 2.0,
            3.0, 4.0,
            5.0, 7.0,
            3.5, 5.0;

    HierarchicalClustering hc(2);
    hc.fit(data);
    std::vector<int> labels = hc.get_labels();

    std::cout << "Cluster labels:\n";
    for (size_t i = 0; i < labels.size(); ++i) {
        std::cout << "Data point " << i << " is in cluster " << labels[i] << std::endl;
    }

    return 0;
}
