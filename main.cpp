#include "minimumDistanceClassifier.h"

int main (){
    std::vector<Instance<2>> trainingData = {
        {{1.0, 2.0}, 0},
        {{2.0, 3.0}, 1},
        {{3.0, 4.0}, 0},
        // More training instances can be added
    };

    // New instance to be tested 
    Instance<2> newInstance = {{2.5,2.4},-1};

    int result = classify(trainingData, newInstance);

    if (result){
        std::cout << "The new instance is classified as class" << result << std::endl;
    }

    return 0;
}