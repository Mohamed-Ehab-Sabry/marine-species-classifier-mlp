import cv2 # Library for dealing with images, the biggest and the fastest, written in cpp, wrapped in python.
import os # Library for dealing wiht the operating system, getting folders, files, and much much more.
import numpy as org_np # Numerical Python, library for doing most of the mathematical operations needed in machine learning.
import cupy as np # NumPy, but runs on the GPU.
import matplotlib.pyplot as plt


def one_hot_encoding(img_true_classes, num_of_classes):
    num_of_images = img_true_classes.shape[0]
    one_hot_encoded_matrix = np.zeros((num_of_images, num_of_classes))
    one_hot_encoded_matrix[np.arange(num_of_images), img_true_classes] = 1

    return one_hot_encoded_matrix

def load_dataset(data_dir, img_size = 64):
    features = []
    labels = []

    classes_names = [entry.name for entry in os.scandir(data_dir) if entry.is_dir()]
    classes_names = sorted(classes_names)

    for class_index, class_name in enumerate(classes_names):
        class_path = os.path.join(data_dir,class_name)
        print(f"Loading Class {class_index}: {class_name}")

        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            
            try:
                img_np_arr = cv2.imread(img_path)

                if img_np_arr is None:
                    continue
                
                img_resized_arr = cv2.resize(img_np_arr, (img_size,img_size))
                img_flatten_arr = img_resized_arr.flatten()
                features.append(img_flatten_arr)
                labels.append(class_index)

            except Exception as e:
                print(f"Error loading {img_name}: {e}")


    X = np.array(features)
    y = np.array(labels)

    X = X/255
    y = one_hot_encoding(y, len(classes_names))

    return X, y, classes_names

def split_data(X, y, train_ratio = 0.7, v_ratio = 0.15, test_ratio = 0.15):
    num_of_imgs = X.shape[0]
    indices = np.random.permutation(num_of_imgs)
    train_index = int(train_ratio * num_of_imgs)
    v_index = int(v_ratio * num_of_imgs)
    test_index = int(test_ratio * num_of_imgs) # just for indication, not really needed

    train_idx = indices[:train_index]
    v_idx = indices[train_index: train_index + v_index]
    test_idx = indices[train_index + v_index:]

    X_train, X_validation, X_test = X[train_idx], X[v_idx], X[test_idx]
    y_train, y_validation, y_test = y[train_idx], y[v_idx], y[test_idx]

    return X_train, X_validation, X_test, y_train, y_validation, y_test

def build_nueral_network(in_size, out_size):
    # number of hiddent layers
    hid_layers_num = int(input("So, how many hidden layers do you need for your MLP ?\nNumber of hidden layers: "))
    hid_sizes = []

    # number of nuerons per layer
    for i in range(hid_layers_num):
        hid_layer_size = int(input(f"Please enter the number of nuerons that you need in layer number {i + 1}: "))
        hid_sizes.append(hid_layer_size)

    # choose activations function
    activ_func = int(input("""
                           Lastly, what is the Activation function you want to use in the hidden layers?
                           1) Sigmoid
                           2) Relu
                           Choose (1/2): 
                           """))

    net_sizes = in_size + hid_sizes + out_size
    return net_sizes , activ_func

# initialize the weights randomly
def intialize_param(net_sizes, activ_func):        
    weights = []
    biases = []

    for i in range(len(net_sizes) - 1):
        input_dim = net_sizes[i] 
        output_dim =  net_sizes[i + 1]

        if i < len(net_sizes) - 2:
            # Sigmoid
            if activ_func == 1:
                weight = np.random.randn(output_dim, input_dim) * np.sqrt(1 / input_dim)
            # Relu
            elif activ_func == 2:
                weight = np.random.randn(output_dim, input_dim) * np.sqrt(2 / input_dim)
                
        else:
            weight = np.random.randn(output_dim, input_dim) * np.sqrt(1 / input_dim)
             
        weights.append(weight)
        bias = np.zeros((output_dim, 1))
        biases.append(bias)

    return weights, biases

# ====================================================
# Activation functions
# ====================================================
def Sigmoid(z):
    return 1 / (1 + np.exp(-z))

def Relu(z):
    return np.maximum(0, z)

def SoftMax(z):
    # 1. Numerical Stability Trick
    # If Z contains large numbers wil cause an overflow error.
    # By subtracting the max value from every column, the largest number becomes 0.
    # e^0 = 1, which is safe. The math result remains identical.
    # axis=0 means "find max down the column" (for each image separately).
    # keepdims = true => keeps the values that we divide the matrix by in a proper shape to do so (as in vecotr not elements)

    # SoftMax = e^z/sum(e^z)
    shift_z = z - np.max(z, axis=0, keepdims=True)
    exp_z = np.exp(shift_z)
    sum_exp_z = np.sum(exp_z, axis=0, keepdims= True)

    A = exp_z / sum_exp_z

    return A

# ===============================================================
# Forward and Backward propagations
# ===============================================================
def forward_propagation(X, weights, biases, activ_func):

    A = X.T # input for each layer initialized with the input layer

    # to save the values of A(inputs to the layer) and Z(Netj) for the backward propagation
    # Why save Z? simply because we use its derivative in the backword path
    cache_A = []
    cache_Z = []
    cache_A.append(A)

    # enumerate gives automatic counter inside loop without the need to manually increment
    # it returns [index, tubel(weights, biases)]
    for layer_index, (w,b) in enumerate(zip(weights, biases)):
        Z = np.dot(w,A) + b # Netj = Z = wx + b
        cache_Z.append(Z)

        if layer_index == len(weights) - 1: # if it is the ouput layer
            A = SoftMax(Z)
            cache_A.append(A)
        else:
            if activ_func == 1: # hidden with Sigmoid as an activation function
                A = Sigmoid(Z)
                cache_A.append(A)
            elif activ_func == 2: # hidden with Relu as an acitivation function
                A = Relu(Z)
                cache_A.append(A)
    
    return cache_A, cache_Z

# functions derivatives
def sigmoid_derivative(z):
    sig = Sigmoid(z)
    return sig * (1 - sig)

def relu_derivative(z):
    return (z > 0).astype(float)

# Backward prop
def backward_propagation(cache_A, cache_Z, weights, y, activ_func, m, lambda_reg = 0.0):
    num_layers = len(weights)
    weight_gradients = []
    bias_gradients = []

    # dA = Predictions - True Labels
    dA = cache_A[-1] - y.T

    # Backpropagate through each layer (from last to first)
    for layer_index in range(num_layers - 1, -1, -1):
        # Get Z and A for this layer
        Z_curr = cache_Z[layer_index]
        A_prev = cache_A[layer_index]
        W_curr = weights[layer_index]

        # Calculate dZ (derivative with respect to Z)
        if layer_index == num_layers - 1:
            dZ = dA
        else:
            # Hidden layers: dZ = dA * activation_derivative(Z)
            if activ_func == 1:  # Sigmoid
                dZ = dA * sigmoid_derivative(Z_curr)
            elif activ_func == 2:  # ReLU
                dZ = dA * relu_derivative(Z_curr)
        
        # Calculate dW (gradient for weights)
        # dW = (1/m) * dZ * A_prev.T
        dW = (1/m) * np.dot(dZ, A_prev.T)
        if lambda_reg > 0.0:
            dW += (lambda_reg / m) * W_curr
        weight_gradients.insert(0, dW)
        
        # Calculate db (gradient for biases)
        # db = (1/m) * sum of dZ along all samples
        db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
        bias_gradients.insert(0, db)
        
        # Calculate dA for next iteration (previous layer)
        # dA_prev = W.T * dZ
        if layer_index > 0:
            dA = np.dot(W_curr.T, dZ)
    
    return weight_gradients, bias_gradients

# ===============================================================
# Loss functions
# ===============================================================
def calculate_loss(predictions, y_true, m, weights = None, lambda_reg = 0.0):
    # Add small epsilon to avoid log(0)
    epsilon = 1e-15
    predictions = np.clip(predictions, epsilon, 1 - epsilon)

    # Cross-entropy for multi-class classification
    # y_true.T is (num_classes, m), predictions is (num_classes, m)
    # We multiply element-wise and sum
    cross_entropy_loss = -(1/m) * np.sum(y_true.T * np.log(predictions))
    l2_reg = 0.0
    if weights is not None and lambda_reg > 0.0:
        for w in weights:
            l2_reg += np.sum(w**2)
        l2_reg = (lambda_reg / (2 * m)) * l2_reg

    
    return cross_entropy_loss + l2_reg

# Updating the parameters
def update_parameters(weights, biases, weight_gradients, bias_gradients, learning_rate):
    updated_weights = []
    updated_biases = []
    
    for i in range(len(weights)):
        # Update weights: W = W - learning_rate * dW
        updated_W = weights[i] - learning_rate * weight_gradients[i]
        updated_weights.append(updated_W)
        
        # Update biases: b = b - learning_rate * db
        updated_b = biases[i] - learning_rate * bias_gradients[i]
        updated_biases.append(updated_b)
    
    return updated_weights, updated_biases

def calc_acc(X, y_true_one_hot, weights, biases, activ_func):
    a_cache, _ = forward_propagation(X, weights, biases, activ_func)
    probabilities = a_cache[-1]

    predictions = np.argmax(probabilities, axis = 0)
    true_labels = np.argmax(y_true_one_hot, axis = 1)

    acc = np.mean(predictions == true_labels)
    return acc * 100

# ===============================================================
# Actual training
# ===============================================================
def train(X_train, y_train, X_test, y_test, net_sizes, activ_func, epochs = 1000, learning_rate = 0.05, lambda_reg = 0.01):

    num_of_training_imgs = X_train.shape[0]
    weights, biases = intialize_param(net_sizes, activ_func)

    loss_history = []
    train_acc_history = []
    test_acc_history = []

    print(f"Training on GPU for {epochs} epochs...")

    for i in range(epochs):
        a_cache, z_cache = forward_propagation(X_train,weights,biases,activ_func)
        weight_grads, bias_grads = backward_propagation(a_cache, z_cache, weights, y_train, activ_func, num_of_training_imgs, lambda_reg = lambda_reg)       
        updated_weights, updated_biases = update_parameters(weights, biases, weight_grads, bias_grads, learning_rate)
        weights, biases = updated_weights, updated_biases


        if i % 100 == 0:
            current_loss = calculate_loss(a_cache[-1], y_train, num_of_training_imgs, weights = weights, lambda_reg = lambda_reg)
            train_acc = calc_acc(X_train, y_train, weights, biases, activ_func)
            test_acc = calc_acc(X_test, y_test, weights, biases, activ_func)

            loss_history.append(current_loss)
            train_acc_history.append(train_acc)
            test_acc_history.append(test_acc)
            print(f"Epoch {i}: Loss {current_loss: .4f} | Train Acc: {train_acc: .1f}% | Test Acc: {test_acc: .1f}%")

    return weights, biases, loss_history, train_acc_history, test_acc_history

# ===============================================================
# Prediction function
# ===============================================================
def predict(X_test, y_test, weights, biases,activ_func, classes_names):
    a_cache, _ = forward_propagation(X_test, weights, biases, activ_func)
    probabilities_of_output_class = a_cache[-1]
    predicted_indices_gpu = np.argmax(probabilities_of_output_class, axis = 0)
    true_indices_gpu = np.argmax(y_test, axis = 1)
    predicted_indices_cpu = predicted_indices_gpu.get()
    true_indices_cpu = true_indices_gpu.get()

    predicted_names = org_np.array(classes_names)[predicted_indices_cpu]
    true_names = org_np.array(classes_names)[true_indices_cpu]

    return predicted_names, true_names

# ===============================================================
# Ploting
# ===============================================================
def plot_results(loss_history, train_acc, test_acc):
    # Plot Loss and Accuracy side by side
    plt.figure(figsize=(12, 5))
    
    # Subplot 1: Loss
    plt.subplot(1, 2, 1)
    plt.plot(loss_history, label='Training Loss', color='red')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs (x100)')
    plt.ylabel('Loss')
    plt.legend()
    
    # Subplot 2: Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label='Train Accuracy', color='blue')
    plt.plot(test_acc, label='Test Accuracy', color='green')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epochs (x100)')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.show()

# ===============================================================
# Running the modle
# ===============================================================
# X, y, classes_names = load_dataset(r"C:\\Life\\FCAI_Stuff\\Third_year_AI\\Intro_to_ML\Assignments\Sea_Animals")
# X_train, X_v, X_test, y_train, y_v, y_test = split_data(X, y, train_ratio = 0.7, v_ratio = 0.15, test_ratio = 0.15)

# input_layer_size = [X_train.shape[1]] # 1) Number of features per image
# output_layer_size = [len(classes_names)] # 2) Number of classes

# net_sizes, activ_func = build_nueral_network(input_layer_size, output_layer_size)

# print("\nStarting Training on GPU...")
# trained_weights, trained_biases, loss_hist, train_acc_hist, test_acc_hist = train(X_train, y_train, X_test, y_test, net_sizes, activ_func, epochs = 1500, learning_rate = 0.1)

# print("\nRunning Prediction on Test Set...")
# pred_names, true_names = predict(X_test, y_test, trained_weights, trained_biases, activ_func, classes_names)

# print("\n--- Test Results ---")
# correct_count = 0
# total_count = len(pred_names)

# for i in range(total_count):
#     if pred_names[i] == true_names[i]:
#         correct_count += 1
#     # Simple print format
#     print(f"Predicted: {pred_names[i]:<15} | Actual: {true_names[i]}")

# # 6. Final Accuracy
# accuracy = (correct_count / total_count) * 100
# print(f"\nFinal Test Accuracy: {accuracy:.2f}%")

# # 7. Plot Graphs
# # plot_results(loss_hist, train_acc_hist, test_acc_hist)

# ===================================================================
# Sanity check on sklearn digits dataset (small 8x8 grayscale images)
# ===================================================================
from sklearn.datasets import load_digits

def load_sklearn_digits_dataset():
    digits = load_digits()
    # Digits images are 8x8 with values in [0, 16]; normalize to [0, 1]
    X_np = digits.images.astype(org_np.float32) / 16.0
    X_flat = X_np.reshape(len(X_np), -1)
    # Move data to GPU (cupy)
    X = np.asarray(X_flat)
    y_int = np.asarray(digits.target, dtype=np.int64)
    y = one_hot_encoding(y_int, 10)
    class_names = [str(i) for i in range(10)]
    return X, y, class_names


# Load digits dataset and split
X_digits, y_digits, class_names_digits = load_sklearn_digits_dataset()
X_train_d, X_v_d, X_test_d, y_train_d, y_v_d, y_test_d = split_data(
    X_digits, y_digits, train_ratio=0.7, v_ratio=0.15, test_ratio=0.15
)

# Build network with same interactive helper
input_layer_size_d = [X_train_d.shape[1]]
output_layer_size_d = [len(class_names_digits)]
net_sizes_d, activ_func_d = build_nueral_network(input_layer_size_d, output_layer_size_d)

print("\nStarting Training on GPU with sklearn digits...")
trained_w_d, trained_b_d, loss_hist_d, train_acc_hist_d, test_acc_hist_d = train(
    X_train_d,
    y_train_d,
    X_test_d,
    y_test_d,
    net_sizes_d,
    activ_func_d,
    epochs=800,
    learning_rate=0.1,
    lambda_reg=0.0,
)

print("\nRunning Prediction on Digits Test Set...")
pred_names_d, true_names_d = predict(
    X_test_d, y_test_d, trained_w_d, trained_b_d, activ_func_d, class_names_digits
)

print("\n--- Digits Test Results (first 25) ---")
preview_count = min(len(pred_names_d), 25)
for i in range(preview_count):
    print(f"Predicted: {pred_names_d[i]:<3} | Actual: {true_names_d[i]}")

digits_accuracy = (org_np.sum(pred_names_d == true_names_d) / len(pred_names_d)) * 100
print(f"\nFinal Digits Test Accuracy: {digits_accuracy:.2f}%")

# plot_results(loss_hist_d, train_acc_hist_d, test_acc_hist_d)

# =======================================================================
# Harder sanity check on sklearn Olivetti faces dataset (64x64 grayscale)
# =======================================================================
from sklearn.datasets import fetch_olivetti_faces

def load_olivetti_faces_dataset():
    faces = fetch_olivetti_faces()
    # faces.images: (400, 64, 64), values already in [0, 1]
    X_np = faces.images.astype(org_np.float32)
    X_flat = X_np.reshape(len(X_np), -1)
    X = np.asarray(X_flat)  # move to GPU
    y_int = np.asarray(faces.target, dtype=np.int64)
    num_classes = int(y_int.max()) + 1
    y = one_hot_encoding(y_int, num_classes)
    class_names = [f"person_{i}" for i in range(num_classes)]
    return X, y, class_names


# Load Olivetti faces dataset and split
X_faces, y_faces, class_names_faces = load_olivetti_faces_dataset()
X_train_f, X_v_f, X_test_f, y_train_f, y_v_f, y_test_f = split_data(
    X_faces, y_faces, train_ratio=0.7, v_ratio=0.15, test_ratio=0.15
)

# Build network for faces (input size 4096, 40 classes)
input_layer_size_f = [X_train_f.shape[1]]
output_layer_size_f = [len(class_names_faces)]
net_sizes_f, activ_func_f = build_nueral_network(input_layer_size_f, output_layer_size_f)

print("\nStarting Training on GPU with Olivetti faces (harder task)...")
trained_w_f, trained_b_f, loss_hist_f, train_acc_hist_f, test_acc_hist_f = train(
    X_train_f,
    y_train_f,
    X_test_f,
    y_test_f,
    net_sizes_f,
    activ_func_f,
    epochs=1200,
    learning_rate=0.05,
    lambda_reg=0.01,
)

print("\nRunning Prediction on Olivetti Test Set...")
pred_names_f, true_names_f = predict(
    X_test_f, y_test_f, trained_w_f, trained_b_f, activ_func_f, class_names_faces
)

print("\n--- Olivetti Test Results (first 25) ---")
preview_count_f = min(len(pred_names_f), 25)
for i in range(preview_count_f):
    print(f"Predicted: {pred_names_f[i]:<10} | Actual: {true_names_f[i]}")

faces_accuracy = (org_np.sum(pred_names_f == true_names_f) / len(pred_names_f)) * 100
print(f"\nFinal Olivetti Test Accuracy: {faces_accuracy:.2f}%")

# plot_results(loss_hist_f, train_acc_hist_f, test_acc_hist_f)
