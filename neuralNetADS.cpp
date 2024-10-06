// Author: Alden Sahi
// Date: Oct 2024
// Project Name: NeuralNetADS
//Project Description: Implementing a Neural Network using CPP
#include <vector>

using namespace std;

class Neuron;

typedef vector<Neuron> Layer;

class Network {
    public:
        //constructor
        Network( const vector<unsigned> &topology);
        void feedFoward(const vector<double> &inputVals);
        void backProp(const vector<double> &targetVals);
        void getResults(vector<double> &resultsVals) const;
    private:
        // m_layers[layerNum][NeuronNum]
        vector<Layer> m_layers;
};

Network:: Network( const vector<unsigned> &topology) {
    unsigned numLayers = topology.size();
    unsigned numNeurons = topology.size();

    for (unsigned layerNum = 0 ; layerNum < numLayers; ++layerNum) {
        m_layers.push_back(Layer());
        for(unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum ) {
            m_layers.back().push_back(Neuron());
        }
    }
}

int main() {
    // instance of model
    vector<unsigned> topology;
    Network myNetwork(topology);

    vector<double> inputVals;
    // feeds input data to model
    myNetwork.feedFoward(inputVals);

    vector<double> targetVals;
    // performs back propogation
    myNetwork.backProp(targetVals);

    vector<double> resultVals;
    // to display results
    myNetwork.getResults(resultVals);
}