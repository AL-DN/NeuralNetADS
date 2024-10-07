// Author: Alden Sahi
// Date: Oct 2024
// Project Name: NeuralNetADS
//Project Description: Implementing a Implicityly and Fully Connected Neural Network using CPP
#include <vector>
#include <cstdlib>
#include <iostream>

using namespace std;

struct Connection 
{
    double weight;
    double deltaweight;

};

class Neuron;
typedef vector<Neuron> Layer;

/*------------------ NEURON CLASS ----------------*/
class Neuron {
    public:
    Neuron(unsigned numOutputs);

    private:
        static double randomWeight(void) { return rand()/double(RAND_MAX)}
        double m_ouputVal;
        // pf type Connection since we need to store weight and the change in weight
        vector<Connection> m_outputWeights;

};

/*-------------- External Neuron Constructor Definition-----------------*/


Neuron::Neuron(unsigned numOutputs) {
    for (unsigned i = 0; i < numOutputs; ++i ) {
        m_outputWeights.push_back(Connection());
        m_outputWeights.back().weight= randomWeight();
    }
};


/*-------------------Network Class------------*/
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

/*-------------- External Network Constructor Definition-----------------*/

Network:: Network( const vector<unsigned> &topology) {
    unsigned numLayers = topology.size();
    
    for (unsigned layerNum = 0 ; layerNum < numLayers; ++layerNum) {
        m_layers.push_back(Layer());
        unsigned numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];

        for(unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum ) {
            m_layers.back().push_back(Neuron(numOutputs));
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