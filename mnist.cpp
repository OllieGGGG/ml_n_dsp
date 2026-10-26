#include <fstream>
#include <vector>
#include <sstream>
#include <iostream>

#include "nn.cpp"

void fill_mnist_datasets(std::vector<std::pair<uint8_t, std::vector<uint8_t>>> &data, std::string file_name) {
    std::ifstream train_set_file(file_name);
    if (!train_set_file.is_open()) {
        printf("Can not open file\n");
        exit(-1);
    }

    std::string img_line;
    while (std::getline(train_set_file, img_line)) {
        stringstream ss(img_line);
        std::string token;

        getline(ss, token, ',');
        uint8_t number = atoi(token.c_str());
        std::vector<uint8_t> img_data;
        while (getline(ss, token, ',')) {
            img_data.push_back(atoi(token.c_str()));
        }
        data.push_back({number, img_data});
    }
}

struct NN {
    Layer layer1 = Layer(784, 512);
    ReLU relu1 = ReLU();
    Layer layer2 = Layer(512, 10);
    CrossCategoricalLoss cross_cat = CrossCategoricalLoss();

    double forward(Matrix2d *x_train_batch, Matrix2d *y_train_batch, Matrix2d **output) {
        Matrix2d *res1 = layer1.forward(x_train_batch);
        Matrix2d *res2 = relu1.forward(res1);
        Matrix2d *res3 = layer2.forward(res2);

        double loss = cross_cat.loss(res3, y_train_batch);

        if (output != NULL) *output = cross_cat.outputs;
        deallocate_2d_matrix(res3);
        deallocate_2d_matrix(res1);

        return loss;
    }

    void backward(Matrix2d *y_train_batch) {
        cross_cat.backward(y_train_batch);
        layer2.backward(cross_cat.dinputs);
        relu1.backward(layer2.dinputs);
        layer1.backward(relu1.dinputs);
    }

    void optimize(double learning_rate) {
        layer2.optimize(learning_rate);
        layer1.optimize(learning_rate);
    }

    void finish() {
        layer2.finish_pass();
        layer1.finish_pass();
        relu1.finish_pass();
        cross_cat.finish_pass();
    }
};

void print_image(const std::vector<uint8_t>& pixels)
{
    constexpr std::size_t width = 28;
    constexpr std::size_t height = 28;
    constexpr char shades[] = " .:-=+*#%@";
    constexpr std::size_t shadeCount = sizeof(shades) - 1;

    if (pixels.size() != width * height) {
        throw std::invalid_argument("Expected exactly 784 pixels");
    }

    for (std::size_t y = 0; y < height; ++y) {
        for (std::size_t x = 0; x < width; ++x) {
            const auto pixel = pixels[y * width + x];
            const char shade = shades[pixel * (shadeCount - 1) / 255];

            std::cout << shade << shade;
        }
        std::cout << '\n';
    }
}

int main() {
    std::vector<std::pair<uint8_t, std::vector<uint8_t>>> mnist_train;
    std::vector<std::pair<uint8_t, std::vector<uint8_t>>> mnist_test;

    fill_mnist_datasets(mnist_train, std::string("mnist_train_set.csv"));
    fill_mnist_datasets(mnist_test, std::string("mnist_test_set.csv"));

    NN nn;

    // Training
    unsigned batch_size = 8;
    unsigned epochs = 1;

    Matrix2d *x_train_batch = allocate_2d_matrix(mnist_train[0].second.size(), batch_size);
    Matrix2d *y_train_batch = allocate_2d_matrix(10, batch_size);

    while (epochs-- > 0) {
        for (unsigned start = 0; start < mnist_train.size(); start += batch_size) {
            unsigned end = start + batch_size;
            if (end > mnist_train.size()) {
                break;
            }
            printf("Start idx: %d, ", start);

            // Fill batch of input and output train data
            zeroed_matrix(y_train_batch);
            for (unsigned sample_idx = 0; sample_idx < batch_size; sample_idx++) {
                // Copy pixel values
                for (unsigned pixel = 0; pixel < mnist_train[0].second.size(); pixel++) {
                    x_train_batch->mat[pixel][sample_idx] = mnist_train[start + sample_idx].second[pixel] / 255.0;
                }

                uint8_t target_number = mnist_train[start + sample_idx].first;
                y_train_batch->mat[target_number][sample_idx] = 1;
            }

            double loss = nn.forward(x_train_batch, y_train_batch, NULL);
            printf("Loss: %f\n", loss);

            //
            // Backward pass
            //
            // Why for example not to take derivative directly from output of neural network?
            // Because with backward propagation we want to learn how parameters in NN influence the output and we want to make
            // change of parameters to change the output, here is the question: which output do we want?
            // And that's why we use MSE as loss function because with MSE we know in which direction we want to move. In that case we want to make it close to 0

            nn.backward(y_train_batch);
            nn.optimize(0.003);
            nn.finish();
        }
    }
    deallocate_2d_matrix(x_train_batch);
    deallocate_2d_matrix(y_train_batch);

    // Counting accuracy
    {
        Matrix2d *x_train_batch = allocate_2d_matrix(mnist_train[0].second.size(), 1);
        Matrix2d *y_train_batch = allocate_2d_matrix(10, 1);
        unsigned cnt_right = 0;
        for (unsigned sample_idx = 0; sample_idx < mnist_test.size(); sample_idx++) {
            zeroed_matrix(x_train_batch);
            zeroed_matrix(y_train_batch);
            // Copy pixel values
            for (unsigned pixel = 0; pixel < mnist_test[0].second.size(); pixel++) {
                x_train_batch->mat[pixel][0] = mnist_test[sample_idx].second[pixel] / 255.0;
            }

            uint8_t target_number = mnist_test[sample_idx].first;
            y_train_batch->mat[target_number][0] = 1;

            Matrix2d *output;
            nn.forward(x_train_batch, y_train_batch, &output);

            uint8_t max_value_idx = 0;
            double max_value = 0.0;
            for (unsigned idx = 0; idx < 10; idx++) {
                if (output->mat[idx][0] >= max_value) {
                    max_value = output->mat[idx][0];
                    max_value_idx = idx;
                }
            }
            if (max_value_idx == target_number) {
                cnt_right++;
            }
            printf("sample idx: %d\n", sample_idx);
        }
        printf("Accuracy: %f\n", (double)cnt_right / (double)mnist_test.size() * 100.0);
        deallocate_2d_matrix(x_train_batch);
        deallocate_2d_matrix(y_train_batch);
    }

    {
        while (true) {
            int sample_idx;
            printf("Enter an index: ");
            scanf("%d", &sample_idx);
            printf("You choose sample index: %d\n", sample_idx);

            Matrix2d *x_test_batch = allocate_2d_matrix(mnist_test[sample_idx].second.size(), 1);
            for (unsigned pixel = 0; pixel < mnist_test[sample_idx].second.size(); pixel++) {
                x_test_batch->mat[pixel][0] = mnist_test[sample_idx].second[pixel] / 255.0;
            }
            Matrix2d *y_test_batch = allocate_2d_matrix(10, 1);
            uint8_t target_number = mnist_test[sample_idx].first;
            y_test_batch->mat[target_number][0] = 1;

            printf("You choose:\n");
            print_image(mnist_test[sample_idx].second);

            Matrix2d *output;
            nn.forward(x_test_batch, y_test_batch, &output);
            printf("NN output: \n");
            print_matrix(output);

            deallocate_2d_matrix(x_test_batch);
            deallocate_2d_matrix(y_test_batch);
        }
    }

    return 0;
}
