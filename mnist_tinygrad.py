import numpy as np

from tinygrad import nn, Context, TinyJit
from tinygrad.tensor import Tensor
from tinygrad.nn.state import safe_save, safe_load, get_state_dict, load_state_dict

def read_dataset(file_name, rows):
    x_set = np.empty((rows, 784), np.uint8)
    y_set = np.empty(rows, np.uint8)
    idx = 0
    with open(file_name, "r") as f:
        line = f.readline().split(",")
        while line != ['']:
            target_number = int(line[0])
            x_set[idx] = np.array([int(c) for c in line[1:]])
            y_set[idx] = target_number
            idx += 1

            line = f.readline().split(",")
    return Tensor(x_set), Tensor(y_set)

class MnistNet:
    def __init__(self):
        self.l1 = Tensor.kaiming_uniform(784, 128)
        self.l2 = Tensor.kaiming_uniform(128, 10)

    def __call__(self, x):
        return x.flatten(1).dot(self.l1).relu().dot(self.l2)

x_train, y_train = read_dataset("mnist_train_set.csv", 60000)
#print(f"X train len: {len(x_train)}, y train len: {len(y_train)}")
x_test, y_test = read_dataset("mnist_test_set.csv", 10000)
#print(f"X train len: {len(x_test)}, y train len: {len(y_test)}")

model = MnistNet()
optim = nn.optim.Adam([model.l1, model.l2], lr = 0.001)
batch_size = 128

@Context(TRAINING=1)
def step():
    samp = Tensor.randint(batch_size, high=x_train.shape[0])

    x_train_batch = x_train[samp]
    y_train_batch = y_train[samp]

    optim.zero_grad()
    out = model(x_train_batch)
    loss = out.sparse_categorical_crossentropy(y_train_batch).backward()
    optim.step()
    return loss

jit_step = TinyJit(step)
for s in range(10000):
    loss = jit_step() 
    if s % 100 == 0:
        acc = (model(x_test).argmax(axis=1) == y_test).mean().item()
        print(f"Acc: {acc}")

#print("Start measuring accuracy")
#avg_acc = 0
#for s in range(1000):
#    samp = Tensor.randint(batch_size, high=x_test.shape[0])
#
#    x_test_batch = x_test[samp]
#    y_test_batch = y_test[samp]
#
#    out = model(x_test_batch)
#    pred = out.argmax(axis=-1)
#    avg_acc += (pred == y_test_batch).mean()
#
#    if s % 100 == 0:
#        print(f"Avg acc: {(avg_acc / s).numpy()}")
#print(f"Avg acc: {(avg_acc / 1000).numpy()}")

# Saving the model
state_dict = get_state_dict(model)
safe_save(state_dict, "tinygrad_mnist.safetensors")

# Interactive
def print_mnist(arr):
    pixels = np.asarray(arr, dtype=float).reshape(28, 28)
    chars = np.array(list(" .:-=+*#%@"))

    # Normalize brightness to [0, 1].
    low, high = pixels.min(), pixels.max()
    if high > low:
        pixels = (pixels - low) / (high - low)
    else:
        pixels = np.zeros_like(pixels)

    indices = (pixels * (len(chars) - 1)).astype(int)

    for row in indices:
        print("".join(chars[i] * 2 for i in row))

while True:
    idx = int(input("Provide the number: "))
    sample = x_test[idx].numpy()
    print_mnist(sample)
    out = model(x_test[idx:idx+1]).softmax()
    print(f"Model output: {out.numpy()}")

