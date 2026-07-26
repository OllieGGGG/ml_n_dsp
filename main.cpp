#include <iostream>
#include <fstream>
#include <random>
#include <vector>
#include <cassert>
#include <cmath>

#include "stdio.h"

using namespace std;

struct Tanh {
    vector<double> outputs;
    vector<double> dinputs;

    vector<double> forward(vector<double> inputs) {
        outputs.resize(inputs.size());
        for (unsigned idx = 0; idx < inputs.size(); idx++) {
            outputs[idx] = tanh(inputs[idx]);
        }
        return outputs;
    }

    void backward(vector<double> dvalues) {
        assert(dvalues.size() == outputs.size());
        dinputs.resize(outputs.size());
        for (unsigned idx = 0; idx < outputs.size(); idx++) {
            dinputs[idx] = dvalues[idx] * (1 - pow(outputs[idx], 2));
        }
    }
};

struct ReLU {
    vector<double> outputs;
    vector<double> dinputs;

    vector<double> forward(vector<double> inputs) {
        outputs.resize(inputs.size());
        for (unsigned idx = 0; idx < inputs.size(); idx++) {
            outputs[idx] = inputs[idx] > 0 ? inputs[idx] : 0;
        }
        return outputs;
    }

    void backward(vector<double> dvalues) {
        assert(dvalues.size() == outputs.size());
        dinputs.resize(outputs.size());
        for (unsigned idx = 0; idx < outputs.size(); idx++) {
            dinputs[idx] = dvalues[idx] * (outputs[idx] > 0 ? 1 : 0);
        }
    }
};

struct Layer {
    vector<vector<double>> weights;
    vector<double> biases;
    vector<double> _inputs;

    // It's 2d because every weight for every neuron weight is unique and influence the output
    vector<vector<double>> dweights;
    // It's 1d because every neuron in the layer receive the same array of inputs and that's why we can just them all
    // across all neurons in the layer
    vector<double> dinputs;

    int main() {
        return 0;
    }
    Layer(unsigned int inputs_num, unsigned int outputs_num) {
        std::random_device dev;
        std::mt19937 rng(dev());
        std::uniform_int_distribution<std::mt19937::result_type> dist6(1, 600);
        for (int i = 0; i < outputs_num; i++) {
            weights.push_back(vector<double>(inputs_num));
            for (unsigned w_idx = 0; w_idx < inputs_num; w_idx++) {
                weights[i][w_idx] = (double)dist6(rng) / 10000;
            }
        }
        biases = vector<double>(outputs_num);
    }
    ~Layer() {};

    void print_weights() {
        printf("Sizes: %d, %d\n", weights.size(), weights[0].size());
        for (unsigned neuron_idx = 0; neuron_idx < weights.size(); neuron_idx++) {
            printf("Neuron: [");
            for (unsigned w_idx = 0; w_idx < weights[0].size(); w_idx++) {
                printf("%f, ", weights[neuron_idx][w_idx]);
            }
            printf("]\n");
        }
    }

    vector<double> forward(vector<double> inputs) {
        assert(inputs.size() == weights[0].size());
        _inputs = inputs;
        vector<double> outputs(weights.size());
        // Iterate over each neuron
        for (unsigned int neuron_idx = 0; neuron_idx < weights.size(); neuron_idx++) {
            outputs[neuron_idx] = 0;
            // Iterate over each weight in neuron and multiply it inputs
            for (unsigned int w_idx = 0; w_idx < weights[0].size(); w_idx++) {
                outputs[neuron_idx] += inputs[w_idx] * weights[neuron_idx][w_idx];
            }
        }
        return outputs;
    }

    void backward(vector<double> dvalues) {
        assert(dvalues.size() == weights.size());

        // Counting dweights
        dweights.resize(weights.size());
        for (unsigned neuron_idx = 0; neuron_idx < weights.size(); neuron_idx++) {
            dweights[neuron_idx].resize(weights[neuron_idx].size());
            for (unsigned w_idx = 0; w_idx < weights[neuron_idx].size(); w_idx++) {
                dweights[neuron_idx][w_idx] = _inputs[w_idx] * dvalues[neuron_idx];
            }
        }

        dinputs.resize(weights[0].size());
        std::fill(dinputs.begin(), dinputs.end(), 0);
        // Counting dinputs
        // First of all iterate over all inputs idx because we gonna sum all the effects of that input through all neurons
        // x1, x2, x3
        // dx1 = dn1 * dw11 + dn2 * dw21 + dn3 * dw31
        for (unsigned input_idx = 0; input_idx < weights[0].size(); input_idx++) {
            for (unsigned neuron_idx = 0; neuron_idx < weights.size(); neuron_idx++) {
                dinputs[input_idx] += dvalues[neuron_idx] * weights[neuron_idx][input_idx];
            }
        }
    }
};

// Generate train dataset from the sine function
void generate_train_dataset(vector<vector<double>> &x_train, vector<double> &y_train) {
    for (double x = -3.14; x < 3.13; x += 0.01) {
        x_train.push_back(vector<double>{x});
        y_train.push_back(sin(x));
    }
}

void optimize_layer(Layer &layer) {
    for (unsigned neuron_idx = 0; neuron_idx < layer.weights.size(); neuron_idx++) {
        for (unsigned w_idx = 0; w_idx < layer.weights[0].size(); w_idx++) {
            layer.weights[neuron_idx][w_idx] -= layer.dweights[neuron_idx][w_idx] * 0.008;
        }
    }
}

template <typename T>
void print_vector(const vector<T>& v) {
    cout << "[";
    for (size_t i = 0; i < v.size(); ++i) {
        cout << v[i];
        if (i + 1 < v.size()) cout << ", ";
    }
    cout << "]\n";
}

void save_plot_data(const vector<vector<double>>& x_train,
                    const vector<double>& y_true,
                    const vector<double>& y_pred) {
    std::ofstream f("plot.dat");
    for (size_t i = 0; i < x_train.size(); ++i) {
        f << x_train[i][0] << " " << y_true[i] << " " << y_pred[i] << "\n";
    }
}

void save_csv(const vector<vector<double>>& x, const vector<double>& y_true, const vector<double>& y_pred) {
    ofstream f("plot_data.csv");
    f << "x,sinx,pred\n";
    for (size_t i = 0; i < x.size(); ++i) {
        f << x[i][0] << "," << y_true[i] << "," << y_pred[i] << "\n";
    }
}

int main() {
    vector<vector<double>> x_train;
    vector<double> y_train;

    generate_train_dataset(x_train, y_train);
    printf("Size of input dataset: %d\n", x_train.size());

    vector<double> x_test(x_train.size());
    vector<double> y_test(y_train.size());

    //x_test = x_train;

    // 1 input - 2 outputs -> 2 neurons
    auto layer1 = Layer(1, 128);
    auto relu1 = Tanh();
    auto layer2 = Layer(128, 1);
    auto tanh = Tanh();

    unsigned epochs = 2000;
    while (epochs-- > 0) {
        for (unsigned input_idx = 0; input_idx < x_train.size(); input_idx++) {
            vector<double> inputs = x_train[input_idx];

            //printf("New path\n");
            // Forward pass
            vector<double> res = layer1.forward(inputs);
            //print_vector(res);
            res = relu1.forward(res);
            //print_vector(res);
            res = layer2.forward(res);
            //print_vector(res);
            res = tanh.forward(res);
            //print_vector(res);
            //printf("Actual value: %f\n", y_train[input_idx]);
            //printf("Your's value: %f\n", res[0]);

            y_test[input_idx] = res[0];

            // Backward pass
            tanh.backward(vector<double>{y_test[input_idx] - y_train[input_idx]});
            layer2.backward(tanh.dinputs);
            relu1.backward(layer2.dinputs);
            layer1.backward(relu1.dinputs);

            optimize_layer(layer2);
            optimize_layer(layer1);
        }
        double mse = 0;
        for (unsigned i = 0; i < x_train.size(); i++) {
            mse += pow(y_test[i] - y_train[i], 2);
        }
        mse /= x_train.size();
        printf("MSE: %f\n", mse);
        if (epochs == 1) {
            save_csv(x_train, y_train, y_test);
        }
    }

    vector<double> test_values = {-0.99, -0.35, 0, 0.22, 0.66, 0.99};
    for (unsigned i = 0; i < test_values.size(); i++) {
        printf("Input value: %f\n", test_values[i]);
        vector<double> res = layer1.forward(vector<double>{test_values[i]});
        //print_vector(res);
        res = relu1.forward(res);
        //print_vector(res);
        res = layer2.forward(res);
        //print_vector(res);
        res = tanh.forward(res);
        //print_vector(res);
        //printf("Actual value: %f\n", y_train[input_idx]);
        printf("Your's value: %f\n", res[0]);
    }

    return 0;
}
