#include "cluster.h"
#include "distance.h"

int main (int argc, char **argv){
    Cluster c1(0);
    Cluster c2(1);

    for(size_t i = 0; i < 5; i++){
        c1.add_item(i);
        c2.add_item(i + 5);
    }
    
    c2.remove_item(7);

    std::set<size_t> data_1, data_2;
    data_1 = c1.data();
    data_2 = c2.data();

    std::set<size_t>::const_iterator it;
    std::cout<< "Cluster 1 has " << c1.size() << " items: "<< std::endl;

    for (it = data_1.begin(); it != data_1.end(); ++it) {
        std::cout << *it << " ";
    }
    std::cout << std::endl;

    std::cout << "Cluster 2 has " << c2.size() << " items: " << std::endl;

    for (it = data_2.begin(); it != data_2.end(); ++it) {
        std::cout << *it << " ";
    }
    std::cout << std::endl;

    std::vector<double> x(2), y(2);
    x[0] = 0; x[1] = 0;
    y[0] = 1; y[1] = 1;

    Euclidean_Distance ed;
    std::cout << "The " << ed.name() << " between x and y is : " << ed(x,y) << std::endl;
    return 0;
}