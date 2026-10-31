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

struct NN {
    Layer *layer1;
    ReLU *relu1;
    Layer *layer2;
    CrossCategoricalLoss *cross_cat;

    NN(struct Arena *global_arena, struct Arena *train_arena, struct Arena *pass_arena) {
        void *buf = arenaPush(global_arena, sizeof(Layer), 4);
        layer1 = ::new (arenaPush(global_arena, sizeof(Layer), 4)) Layer(train_arena, pass_arena, 784, 256);
        relu1 = ::new (arenaPush(global_arena, sizeof(ReLU), 4)) ReLU();
        layer2 = ::new (arenaPush(global_arena, sizeof(Layer), 4)) Layer(train_arena, pass_arena, 256, 10);
        cross_cat = ::new (arenaPush(global_arena, sizeof(CrossCategoricalLoss), 4)) CrossCategoricalLoss();
    }

    double forward(struct Arena *train_arena, struct Arena *pass_arena, Matrix2d *x_train_batch, Matrix2d *y_train_batch, Matrix2d **output) {
        Matrix2d *res1 = layer1->forward(train_arena, pass_arena, x_train_batch);
        Matrix2d *res2 = relu1->forward(pass_arena, res1);
        Matrix2d *res3 = layer2->forward(train_arena, pass_arena, res2);

        double loss = cross_cat->loss(pass_arena, res3, y_train_batch);

        if (output != NULL) *output = cross_cat->outputs;

        return loss;
    }

    void backward(struct Arena *train_arena, struct Arena *pass_arena, Matrix2d *y_train_batch) {
        cross_cat->backward(pass_arena, y_train_batch);
        layer2->backward(train_arena, pass_arena, cross_cat->dinputs);
        relu1->backward(pass_arena, layer2->dinputs);
        layer1->backward(train_arena, pass_arena, relu1->dinputs);
    }

    void optimize(double learning_rate) {
        layer2->optimize(learning_rate);
        layer1->optimize(learning_rate);
    }

    double count_accuracy(struct Arena *train_arena, struct Arena *pass_arena, std::vector<std::pair<uint8_t, std::vector<uint8_t>>> &mnist_test) {
        Matrix2d *x_train_batch = allocate_2d_matrix(train_arena, mnist_test[0].second.size(), 1);
        Matrix2d *y_train_batch = allocate_2d_matrix(train_arena, 10, 1);
        unsigned cnt_right = 0;
        for (unsigned sample_idx = 0; sample_idx < mnist_test.size(); sample_idx++) {
            zeroed_matrix(x_train_batch);
            zeroed_matrix(y_train_batch);
            // Copy pixel values
            for (unsigned pixel = 0; pixel < mnist_test[0].second.size(); pixel++) {
                x_train_batch->mat[pixel][0] = mnist_test[sample_idx].second[pixel];
            }
            min_max_normalize(x_train_batch);

            uint8_t target_number = mnist_test[sample_idx].first;
            y_train_batch->mat[target_number][0] = 1;

            Matrix2d *output;
            forward(train_arena, pass_arena, x_train_batch, y_train_batch, &output);

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
            arenaClear(pass_arena);
        }
        return (double)cnt_right / (double)mnist_test.size() * 100.0;
    }

    void fill_weights(std::string weights_file_name) {
        // Skip verification here. This file should contain 266 rows and every row is the neuron.
        // 255 neurons for 1 layer(784 weights) and 10 neurons for 2 layer(256 weights)
        std::ifstream weights_file(weights_file_name);
        if (!weights_file.is_open()) {
            printf("Can not open file\n");
            exit(-1);
        }

        std::string neuron_line;
        unsigned layer1_neurons = 256;
        int cnt = 0;
        while (layer1_neurons--) {
            std::getline(weights_file, neuron_line);
            stringstream ss(neuron_line);
            std::string token;

            unsigned weight_idx = 0;
            while (getline(ss, token, ',')) {
                layer1->weights->mat[255 - layer1_neurons][weight_idx++] = atof(token.c_str());
                //neuron_weights.push_back(atoi(token.c_str()));
            }
        }
        unsigned layer2_neurons = 10;
        while (layer2_neurons--) {
            std::getline(weights_file, neuron_line);
            stringstream ss(neuron_line);
            std::string token;

            unsigned weight_idx = 0;
            while (getline(ss, token, ',')) {
                layer2->weights->mat[9 - layer2_neurons][weight_idx++] = atof(token.c_str());
                //neuron_weights.push_back(atoi(token.c_str()));
            }
        }
    }
};

uint8_t global_arena_buf[1024 * 128];
struct Arena global_arena;
// For objects that live whole training process
uint8_t train_arena_buf[1024 * 1024 * 32];
struct Arena train_arena;
// For objects that live only in one pass
struct Arena pass_arena;
uint8_t pass_arena_buf[1024 * 1024 * 32]; 

int main() {
    arenaInit(&global_arena, global_arena_buf, sizeof(global_arena_buf));
    arenaInit(&train_arena, train_arena_buf, sizeof(train_arena_buf));
    arenaInit(&pass_arena, pass_arena_buf, sizeof(pass_arena_buf));

    std::vector<std::pair<uint8_t, std::vector<uint8_t>>> mnist_train;
    std::vector<std::pair<uint8_t, std::vector<uint8_t>>> mnist_test;

    fill_mnist_datasets(mnist_train, std::string("mnist_train_set.csv"));
    fill_mnist_datasets(mnist_test, std::string("mnist_test_set.csv"));

    NN nn(&global_arena, &train_arena, &pass_arena);
    print_matrix(nn.layer1->weights);

    // Training
    {
        unsigned batch_size = 8;
        unsigned epochs = 1;

        Matrix2d *x_train_batch = allocate_2d_matrix(&train_arena, mnist_train[0].second.size(), batch_size);
        Matrix2d *y_train_batch = allocate_2d_matrix(&train_arena, 10, batch_size);

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
                        x_train_batch->mat[pixel][sample_idx] = mnist_train[start + sample_idx].second[pixel];
                    }

                    uint8_t target_number = mnist_train[start + sample_idx].first;
                    y_train_batch->mat[target_number][sample_idx] = 1;
                }

                min_max_normalize(x_train_batch);

                double loss = nn.forward(&train_arena, &pass_arena, x_train_batch, y_train_batch, NULL);
                printf("Loss: %f\n", loss);

                //
                // Backward pass
                //
                // Why for example not to take derivative directly from output of neural network?
                // Because with backward propagation we want to learn how parameters in NN influence the output and we want to make
                // change of parameters to change the output, here is the question: which output do we want?
                // And that's why we use MSE as loss function because with MSE we know in which direction we want to move. In that case we want to make it close to 0

                nn.backward(&train_arena, &pass_arena, y_train_batch);
                nn.optimize(0.003);

                arenaClear(&pass_arena);
            }
        }
    }

    // Counting accuracy
    double accuracy = nn.count_accuracy(&train_arena, &pass_arena, mnist_test);
    printf("Accuracy with original weights: %f\n", accuracy);
    nn.fill_weights("mnist_weights.csv");
    accuracy = nn.count_accuracy(&train_arena, &pass_arena, mnist_test);
    printf("Accuracy with tinygrad weights: %f\n", accuracy);

    arenaClear(&train_arena);

    //// Interactive loop
    //{
    //    while (true) {
    //        int sample_idx;
    //        printf("Enter an index: ");
    //        scanf("%d", &sample_idx);
    //        printf("You choose sample index: %d\n", sample_idx);

    //        Matrix2d *x_test_batch = allocate_2d_matrix(mnist_test[sample_idx].second.size(), 1);
    //        for (unsigned pixel = 0; pixel < mnist_test[sample_idx].second.size(); pixel++) {
    //            x_test_batch->mat[pixel][0] = mnist_test[sample_idx].second[pixel] / 255.0;
    //        }
    //        Matrix2d *y_test_batch = allocate_2d_matrix(10, 1);
    //        uint8_t target_number = mnist_test[sample_idx].first;
    //        y_test_batch->mat[target_number][0] = 1;

    //        printf("You choose:\n");
    //        print_image(mnist_test[sample_idx].second);

    //        Matrix2d *output;
    //        nn.forward(x_test_batch, y_test_batch, &output);
    //        printf("NN output: \n");
    //        print_matrix(output);

    //        deallocate_2d_matrix(x_test_batch);
    //        deallocate_2d_matrix(y_test_batch);
    //    }
    //}
    //
    arenaClear(&global_arena);

    return 0;
}
