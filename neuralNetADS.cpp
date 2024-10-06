// Author: Alden Sahi
// Date: Oct 2024
// Project Name: NeuralNetADS
//Project Description: Implementing a Neural Network using CPP

class Network {
    public:
        //constructor
        Network(topology);
        // member function of Network class
        void feedFoward(inputVals);
        void backProp(targetVals);
        // for const correctness
        void getResults(resultsVals) const;
    private:
};

int main() {
    // instance of model
    Network myNetwork(topology);

    // feeds input data to model
    myNetwork.feedFoward(inputVals);
    // performs back propogation
    myNetwork.backProp(targetVals);
    // to display results
    myNetwork.getResults(resultVals);
}