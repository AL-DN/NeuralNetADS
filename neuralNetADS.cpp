// Author: Alden Sahi
// Date: Oct 2024
// Project Name: NeuralNetADS
// Project Description: Implementing an Implicitly and Fully Connected Neural Network using CPP

#include <vector>
#include <cstdlib>
#include <iostream>
#include <cassert>
#include <cmath>
#include <sstream>
#include <fstream>

using namespace std;

struct Connection
{
    double weight;
    double deltaWeight;
};

class Neuron;
typedef vector<Neuron> Layer;

/*------------------ NEURON CLASS ----------------*/

class Neuron
{
public:
    Neuron(unsigned numOutputs, unsigned myIndex);
    void setOutputVal(double val) { m_outputVal = val; }
    double getOutputVal(void) const { return m_outputVal; }
    void feedForward(const Layer &prevLayer);
    void calcOutputGradients(double targetVal);
    void calcHiddenGradients(const Layer &nextLayer);
    void updateInputWeights(Layer &prevLayer);

private:
    vector<Connection> m_outputWeights;
    double m_outputVal;
    unsigned m_myIndex;
    double m_gradient;
    static double eta;   // learning rate
    static double alpha; // momentum

    static double activationFunction(double x) { return tanh(x); }
    static double activationFunctionDerivative(double x);
    static double randomWeight(void) { return rand() / double(RAND_MAX); }
    double sumDOW(const Layer &nextLayer) const;
};

// initalize eta / alpha
double Neuron::eta = 0.15;
double Neuron::alpha = 0.5;

/*-------------- External Neuron Constructor Definition-----------------*/

Neuron::Neuron(unsigned numOutputs, unsigned myIndex)
{
    /* Constructor Summary:
        Iterates through all o
    */
    for (unsigned i = 0; i < numOutputs; ++i)
    {
        m_outputWeights.push_back(Connection());
        m_outputWeights.back().weight = randomWeight();
    }

    m_myIndex = myIndex;
};

/*-------------- Member Functions of Neuron -----------------*/

void Neuron::feedForward(const Layer &prevLayer)
{
    /* Member Function Summary:
        Sums the previous layers outputs with their corresponding weights
        including the bias node

        Params: A reference to the previos layer(in order to get their output value)
    */
    double sum = 0.0;

    for (int n = 0; n < prevLayer.size(); ++n)
    {
        sum += prevLayer[n].getOutputVal() * prevLayer[n].m_outputWeights[m_myIndex].weight;
    }

    m_outputVal = Neuron::activationFunction(sum);
}

double Neuron::activationFunctionDerivative(double x)
{

    // derivative of tanh activation function specified in contructor
    return 1.0 - (x * x);
}

void Neuron::calcOutputGradients(double targetVal)
{
    double delta = targetVal - m_outputVal;
    m_gradient = delta * Neuron::activationFunctionDerivative(m_outputVal);
}

void Neuron::calcHiddenGradients(const Layer &nextLayer)
{
    double dow = sumDOW(nextLayer);
    m_gradient = dow * Neuron::activationFunctionDerivative(m_outputVal);
}

double Neuron::sumDOW(const Layer &nextLayer) const
{
    double sum = 0.0;

    // sum total error given to inputs of next layer
    for (unsigned n = 0; n < nextLayer.size() - 1; ++n)
    {
        sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
    }

    return sum;
}

void Neuron::updateInputWeights(Layer &prevLayer)
{
    for (unsigned n = 0; n < prevLayer.size(); ++n)
    {
        Neuron &neuron = prevLayer[n];
        double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;

        double newDeltaWeight =
            eta * // learning rate
            neuron.getOutputVal() * m_gradient * alpha * oldDeltaWeight;
    }
}

/*-------------------Network Class------------*/

class Network
{

public:
    Network(const vector<unsigned> &topology);
    void feedForward(const vector<double> &inputVals);
    void backProp(const vector<double> &targetVals);
    void getResults(vector<double> &resultsVals) const;
    double getRecentAverageError(void) const { return m_recentAverageError; }

private:
    // m_layers[layerNum][NeuronNum]
    vector<Layer> m_layers;
    double m_error;
    double m_recentAverageError;
    double m_recentAverageSmoothingFactor;
};

/*-------------- External Network Constructor Definition-----------------*/

Network::Network(const vector<unsigned> &topology)
{
    /* Network Constructor Summary:
        Populates m_layers with a number of layers corresponding to entries in topology
        For each entry it will creat neurons based on user specified unsigned int
    */

    unsigned numLayers = topology.size();

    for (unsigned layerNum = 0; layerNum < numLayers; ++layerNum)
    {
        m_layers.push_back(Layer());
        unsigned numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];

        for (unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum)
        {
            m_layers.back().push_back(Neuron(numOutputs, neuronNum));
        }
        m_layers.back().back().setOutputVal(1.0);
    }
}

/*---------------- NETWORK MEMBER FUNCTIONS-------------------*/

void Network::feedForward(const vector<double> &inputVals)
{
    assert(inputVals.size() == m_layers[0].size() - 1);

    // latch inputVals to input neurons
    for (int i = 0; i < inputVals.size(); ++i)
    {
        m_layers[0][i].setOutputVal(inputVals[i]);
    }

    // forward propogate (FOR ALL OTHER LAYERS)
    for (unsigned layerNum = 1; layerNum < m_layers.size(); ++layerNum)
    {
        Layer &prevLayer = m_layers[layerNum - 1];
        // for each neuron
        for (unsigned n = 0; n < m_layers[layerNum].size() - 1; ++n)
        {
            // connect neuron to input vals of next layer
            m_layers[layerNum][n].feedForward(prevLayer);
        }
    }
}

void Network::backProp(const vector<double> &targetVals)
{

    // Calculate Root Mean Square Error of output neuron errors

    Layer &outputLayer = m_layers.back();
    m_error = 0.0;

    for (unsigned n = 0; n < outputLayer.size() - 1; ++n)
    {
        double delta = targetVals[n] - outputLayer[n].getOutputVal();
        m_error += delta * delta;
        m_error = sqrt(m_error);
    }

    // calculates recent average

    m_recentAverageError =
        (m_recentAverageError * m_recentAverageSmoothingFactor + m_error) / (m_recentAverageSmoothingFactor + 1.0);

    // calculates output layer gradient
    for (unsigned n = 0; n < outputLayer.size() - 1; ++n)
    {
        outputLayer[n].calcOutputGradients(targetVals[n]);
    }

    // calculates gradients of hidden layer(s)
    for (unsigned l = m_layers.size() - 2; l > 0; --l)
    {
        Layer &hiddenLayer = m_layers[l];
        Layer &nextLayer = m_layers[l + 1];

        for (unsigned n = 0; n < hiddenLayer.size(); ++n)
        {
            hiddenLayer[n].calcHiddenGradients(nextLayer);
        }
    }

    // for all layers from outputs to first hidden layer,
    // update connect weight

    for (unsigned l = m_layers.size() - 1; l > 0; --l)
    {
        Layer &layer = m_layers[l];
        Layer &prevLayer = m_layers[l - 1];

        for (unsigned n = 0; n < m_layers[l].size() - 1; ++n)
        {
            layer[n].updateInputWeights(prevLayer);
        }
    }
}

void Network::getResults(vector<double> &resultsVals) const
{
    resultsVals.clear();

    for (unsigned n = 0; n < m_layers.back().size() - 1; ++n)
    {
        resultsVals.push_back(m_layers.back()[n].getOutputVal());
    }
}

void showVectorVals(string label, vector<double> &v)
{
    cout << label << " ";
    for (unsigned i = 0; i < v.size(); ++i)
    {
        cout << v[i] << " ";
    }

    cout << endl;
}
/*------------- TrainingData Class ------------------*/

class TrainingData
{
public:
    TrainingData(const string filename);
    bool isEof(void) { return m_trainingDataFile.eof(); }
    void getTopology(vector<unsigned> &topology);

    unsigned getNextInputs(vector<double> &inputsVals);
    unsigned getTargetOutputs(vector<double> &targetOutputVals);

private:
    ifstream m_trainingDataFile;
};

void TrainingData::getTopology(vector<unsigned> &topology)
{
    string line;
    string label;

    if (!getline(m_trainingDataFile, line))
    {
        cout << "Failed to read from file." << endl;
        abort();
    }

    stringstream ss(line);
    ss >> label;
    if (this->isEof() || label.compare("topology:") != 0)
    {
    cout << "Expected 'topology:' but found: " << label << endl;        abort();
    }

    while (!ss.eof())
    {
        unsigned n;
        ss >> n;
        topology.push_back(n);
    }

    return;
}

TrainingData::TrainingData(const string filename)
{
    m_trainingDataFile.open(filename.c_str());
}

unsigned TrainingData::getNextInputs(vector<double> &inputVals)
{
    inputVals.clear();

    string line;
    getline(m_trainingDataFile, line);
    stringstream ss(line);

    string label;
    ss >> label;
    if (label.compare("in:") == 0)
    {
        double oneValue;
        while (ss >> oneValue)
        {
            inputVals.push_back(oneValue);
        }
    }
    return inputVals.size();
}

unsigned TrainingData::getTargetOutputs(vector<double> &targetOutputVals)
{
    targetOutputVals.clear();

    string line;
    getline(m_trainingDataFile, line);
    stringstream ss(line);

    string label;
    ss >> label;
    if (label.compare("out:") == 0)
    {
        double oneValue;
        while (ss >> oneValue)
        {
            targetOutputVals.push_back(oneValue);
        }
    }

    return targetOutputVals.size();
}

int main()
{
    // creates Training Data

    TrainingData trainData("TrainData.txt");

    vector<unsigned> topology;
    trainData.getTopology(topology);
    Network myNetwork(topology);

    vector<double> inputVals, targetVals, resultVals;
    int trainingPass = 0;

    while (!trainData.isEof())
    {
        ++trainingPass;
        cout << endl
             << "Pass " << trainingPass;

        if (trainData.getNextInputs(inputVals) != topology[0])
        {
            break;
        }
        showVectorVals(": Inputs:", inputVals);
        myNetwork.feedForward(inputVals);

        myNetwork.getResults(resultVals);
        showVectorVals("Outputs:", resultVals);

        trainData.getTargetOutputs(targetVals);
        showVectorVals("Targets:", targetVals);
        assert(targetVals.size() == topology.back());

        myNetwork.backProp(targetVals);

        cout << "Net recent average error: "
             << myNetwork.getRecentAverageError() << endl;
    }

    cout << endl
         << "SUCCESS!";
}
