#include <cmath>
#include <cstring>
#include <cassert>
#include <random>

#include <stdio.h>
#include <stdint.h>

using namespace std;

struct Matrix2d {
    double **mat;
    ssize_t rows;
    ssize_t cols;
};

// In the future this function will allocate memory for 2d matrix as sequential elements to make sure lately we can use acceleration
Matrix2d *allocate_2d_matrix(ssize_t rows, ssize_t cols) {
    Matrix2d *matrix = (Matrix2d *)malloc(sizeof(Matrix2d));
    assert(matrix != NULL);
    matrix->mat = (double **)malloc(rows * sizeof(double *));
    assert(matrix->mat != NULL);
    for (unsigned r = 0; r < rows; r++) {
        matrix->mat[r] = (double *)malloc(cols * sizeof(double));
        memset(matrix->mat[r], 0, cols * sizeof(double));
        assert(matrix->mat[r] != NULL);
    }
    matrix->rows = rows;
    matrix->cols = cols;
    return matrix;
}

void deallocate_2d_matrix(Matrix2d *matrix) {
    for (unsigned row = 0; row < matrix->rows; row++) {
        assert(matrix->mat[row] != NULL);
        free(matrix->mat[row]);
    }
    free(matrix->mat);
    free(matrix);
}

void zeroed_matrix(Matrix2d *matrix) {
    for (unsigned r = 0; r < matrix->rows; r++) {
        memset(matrix->mat[r], 0, matrix->cols * sizeof(double));
    }
}

Matrix2d *transpose(Matrix2d *matrix) {
    Matrix2d *transposed_matrix = allocate_2d_matrix(matrix->cols, matrix->rows);

    for (unsigned r = 0; r < matrix->rows; r++) {
        for (unsigned c = 0; c < matrix->cols; c++) {
            transposed_matrix->mat[c][r] = matrix->mat[r][c];
        }
    }

    return transposed_matrix;
}

Matrix2d *multiply(Matrix2d *m1, Matrix2d *m2) {
    assert(m1->cols == m2->rows);

    Matrix2d *output = allocate_2d_matrix(m1->rows, m2->cols);

    for (unsigned m1_row = 0; m1_row < m1->rows; m1_row++) {
        for (unsigned m2_col = 0; m2_col < m2->cols; m2_col++) {
            for (unsigned m2_row = 0; m2_row < m2->rows; m2_row++) {
                output->mat[m1_row][m2_col] += m1->mat[m1_row][m2_row] * m2->mat[m2_row][m2_col];
            }
        }
    }

    return output;
}

Matrix2d *subtraction(Matrix2d *m1, Matrix2d *m2) {
    assert(m1->cols == m2->cols);
    assert(m1->rows == m2->rows);

    Matrix2d *output = allocate_2d_matrix(m1->rows, m1->cols);

    for (unsigned row = 0; row < m1->rows; row++) {
        for (unsigned col = 0; col < m1->cols; col++) {
            output->mat[row][col] = m1->mat[row][col] - m2->mat[row][col];
        }
    }
    return output;
}

void divide(Matrix2d *m, double val) {
    for (unsigned row = 0; row < m->rows; row++) {
        for (unsigned col = 0; col < m->cols; col++) {
            m->mat[row][col] /= val;
        }
    }
}

void multiply(Matrix2d *m, double val) {
    for (unsigned row = 0; row < m->rows; row++) {
        for (unsigned col = 0; col < m->cols; col++) {
            m->mat[row][col] *= val;
        }
    }
}

void print_matrix(Matrix2d *mat, bool print_sizes = false) {
    if (print_sizes) {
        printf("Sizes: %d, %d\n", mat->rows, mat->cols);
    }
    for (unsigned inp_idx = 0; inp_idx < mat->rows; inp_idx++) {
        printf("R_%d: [", inp_idx);
        for (unsigned sample_idx = 0; sample_idx < mat->cols; sample_idx++) {
            printf("%f, ", mat->mat[inp_idx][sample_idx]);
        }
        printf("]\n");
    }
}

struct Tanh {
    Matrix2d *outputs;
    Matrix2d *dinputs;

    Matrix2d *forward(Matrix2d *inputs) {
        outputs = allocate_2d_matrix(inputs->rows, inputs->cols);
        for (unsigned sample_idx = 0; sample_idx < inputs->cols; sample_idx++) {
            for (unsigned sample_feature_idx = 0; sample_feature_idx < inputs->rows; sample_feature_idx++) {
                outputs->mat[sample_feature_idx][sample_idx] = tanh(inputs->mat[sample_feature_idx][sample_idx]);
            }
        }
        return outputs;
    }

    void backward(Matrix2d *dvalues) {
        assert(outputs != NULL);
        assert(outputs->rows == dvalues->rows);
        assert(outputs->cols == dvalues->cols);

        dinputs = allocate_2d_matrix(dvalues->rows, dvalues->cols);
        // derivative of Tanh(x) = 1 - (Tanh(x)** 2);
        for (unsigned sample_idx = 0; sample_idx < dvalues->cols; sample_idx++) {
            for (unsigned sample_feature_idx = 0; sample_feature_idx < dvalues->rows; sample_feature_idx++) {
                dinputs->mat[sample_feature_idx][sample_idx] = dvalues->mat[sample_feature_idx][sample_idx] * (1 - pow(outputs->mat[sample_feature_idx][sample_idx], 2));
            }
        }
    }

    void finish_pass() {
        deallocate_2d_matrix(outputs);
        deallocate_2d_matrix(dinputs);
    }
};

struct ReLU {
    Matrix2d *outputs;
    Matrix2d *dinputs;

    Matrix2d *forward(Matrix2d *inputs) {
        outputs = allocate_2d_matrix(inputs->rows, inputs->cols);
        for (unsigned sample_idx = 0; sample_idx < inputs->cols; sample_idx++) {
            for (unsigned sample_feature_idx = 0; sample_feature_idx < inputs->rows; sample_feature_idx++) {
                outputs->mat[sample_feature_idx][sample_idx] = inputs->mat[sample_feature_idx][sample_idx] > 0 ? inputs->mat[sample_feature_idx][sample_idx] : 0;
            }
        }
        return outputs;
    }

    void backward(Matrix2d *dvalues) {
        assert(outputs != NULL);
        assert(outputs->rows == dvalues->rows);
        assert(outputs->cols == dvalues->cols);

        dinputs = allocate_2d_matrix(dvalues->rows, dvalues->cols);
        // derivative of ReLU = value >= 0 ? 1 : 0
        for (unsigned sample_idx = 0; sample_idx < dvalues->cols; sample_idx++) {
            for (unsigned sample_feature_idx = 0; sample_feature_idx < dvalues->rows; sample_feature_idx++) {
                dinputs->mat[sample_feature_idx][sample_idx] = dvalues->mat[sample_feature_idx][sample_idx] * (outputs->mat[sample_feature_idx][sample_idx] >= 0 ? 1 : 0);
            }
        }
    }

    void finish_pass() {
        deallocate_2d_matrix(outputs);
        deallocate_2d_matrix(dinputs);
    }
};

struct SoftMax {
};

// Representation of the Layer in NN:
// Z = W X + b
// [w11, w12, w13]  [x1]
// [w21, w22, w23]  [x2]
// [w31, w32, w33]  [x3]
struct Layer {
    Matrix2d *weights;
    Matrix2d *biases;
    Matrix2d *_inputs;

    Matrix2d *dweights;
    Matrix2d *dinputs;

    int main() {
        return 0;
    }
    // inputs_num - amount of inputs for one neuron
    // outputs_num - amount of neurons in layer
    Layer(unsigned int inputs_num, unsigned int outputs_num) {
        std::random_device dev;
        std::mt19937 rng(dev());
        // NOTE(Oleg): I don't know why is that range, but I know that it should be small numbers that normally distributed in some range depending
        // on the amount of input parameters
        std::normal_distribution<double> dist(0.0, sqrt(2.0 / (double)inputs_num));

        weights = allocate_2d_matrix(outputs_num, inputs_num);

        for (unsigned neuron_idx = 0; neuron_idx < outputs_num; neuron_idx++) {
            for (unsigned w_idx = 0; w_idx < inputs_num; w_idx++) {
                weights->mat[neuron_idx][w_idx] = dist(rng);
            }
        }
        biases = allocate_2d_matrix(outputs_num, 1);
    }
    ~Layer() {
        deallocate_2d_matrix(weights);
        deallocate_2d_matrix(biases);
    };

    Matrix2d *forward(Matrix2d *inputs) {
        assert(inputs != NULL);
        assert(inputs->rows == weights->cols);

        _inputs = allocate_2d_matrix(inputs->rows, inputs->cols);

        // Remember current inputs for future backpropagation
        for (unsigned r = 0; r < inputs->rows; r++) {
            for (unsigned c = 0; c < inputs->cols; c++) {
                _inputs->mat[r][c] = inputs->mat[r][c];
            }
        }
        
        return multiply(weights, inputs);
    }

    void backward(Matrix2d *dvalues) {
        //assert(dvalues->rows == weights->rows);

        // Counting gradient for weights. From matrix perspective it's actually this:
        //  s
        //  a
        //  m
        //  p
        //  l
        //  e
        //  |
        // [dy11] [x11, x12] - sample = [dw11, dw12]
        // [dy12]                       [dw21, dw22]
        // That's why we need to transpose input matrix befor doing multiplication. Currently every sample in input
        // stored as new COLUMN.
        //
        // Formula for multiple samples:
        // dw = (dy11 * x11 + dy21 * x21 ...) / N

        // Transpose inputs before make multiplication because currently every sample is ROW in the matrix
        Matrix2d *inputs_transposed = transpose(_inputs);
        dweights = multiply(dvalues, inputs_transposed);
        divide(dweights, dvalues->cols);
        deallocate_2d_matrix(inputs_transposed);

        // Counting gradient for inputs. Matrix perspective:
        // transpose gradient from next layer
        // [dy11, dy12] [w11, w12] = [dx1, dx2]
        //              [w21, w22]
        Matrix2d *transposed_dvalues = transpose(dvalues);
        Matrix2d *tmp_dinputs = multiply(transposed_dvalues, weights);
        dinputs = transpose(tmp_dinputs);

        deallocate_2d_matrix(transposed_dvalues);
        deallocate_2d_matrix(tmp_dinputs);
    }

    void finish_pass() {
        deallocate_2d_matrix(_inputs);
        deallocate_2d_matrix(dweights);
        deallocate_2d_matrix(dinputs);
    }

    void optimize(double learning_rate) {
        for (unsigned neuron_idx = 0; neuron_idx < weights->rows; neuron_idx++) {
            for (unsigned w_idx = 0; w_idx < weights->cols; w_idx++) {
                weights->mat[neuron_idx][w_idx] -= dweights->mat[neuron_idx][w_idx] * learning_rate;
            }
        }
    }
};

struct MSE {
    double loss(Matrix2d *y_pred, Matrix2d *y_train) {
        double loss = 0;
        for (unsigned col = 0; col < y_pred->cols; col++) {
            loss += pow(y_train->mat[0][col] - y_pred->mat[0][col], 2);
        }
        loss /= y_pred->cols;
        return loss;
    };

    Matrix2d *dloss(Matrix2d *y_pred, Matrix2d *y_train) {
        Matrix2d *matrix = subtraction(y_pred, y_train);
        multiply(matrix, 2);
        divide(matrix, y_pred->cols);
        return matrix;
    }
};

// Implementation of cross categorical loss function with activation soft max inside
struct CrossCategoricalLoss {
    Matrix2d *outputs;
    Matrix2d *dinputs;

    // This forward method only for soft max activation
    Matrix2d *forward(Matrix2d *inputs) {
        return outputs;
    }

    // y_train will be an array of zeros and only one value will be not 0 and it will multiply the most probable element prediction
    double loss(Matrix2d *inputs, Matrix2d *y_train) {
        // Make forward path for activation soft max
        outputs = allocate_2d_matrix(inputs->rows, inputs->cols);

        for (unsigned sample_idx = 0; sample_idx < inputs->cols; sample_idx++) {
            double exp_sum = 0;

            for (unsigned sample_feature_idx = 0; sample_feature_idx < inputs->rows; sample_feature_idx++) {
                exp_sum += exp(inputs->mat[sample_feature_idx][sample_idx]);
            }
            for (unsigned sample_feature_idx = 0; sample_feature_idx < inputs->rows; sample_feature_idx++) {
                outputs->mat[sample_feature_idx][sample_idx] = exp(inputs->mat[sample_feature_idx][sample_idx]) / exp_sum;
            }
        }
        
        // Count loss
        double loss = 0;
        for (unsigned sample_idx = 0; sample_idx < outputs->cols; sample_idx++) {
            for (unsigned sample_feature_idx = 0; sample_feature_idx < outputs->rows; sample_feature_idx++) {
                loss += y_train->mat[sample_feature_idx][sample_idx] * log(outputs->mat[sample_feature_idx][sample_idx]);
            }
        }
        return loss * (-1);
    }

    void backward(Matrix2d *y_train) {
        dinputs = allocate_2d_matrix(y_train->rows, y_train->cols);

        for (unsigned sample_idx = 0; sample_idx < outputs->cols; sample_idx++) {
            for (unsigned sample_feature_idx = 0; sample_feature_idx < outputs->rows; sample_feature_idx++) {
                dinputs->mat[sample_feature_idx][sample_idx] = outputs->mat[sample_feature_idx][sample_idx] - y_train->mat[sample_feature_idx][sample_idx];
            }
        }
    }

    void finish_pass() {
        deallocate_2d_matrix(outputs);
        deallocate_2d_matrix(dinputs);
    }
};

