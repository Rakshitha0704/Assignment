
import numpy as np
import pickle
from sklearn.metrics import classification_report
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def one_hot_encode(labels, num_classes):
    encoded = np.zeros((labels.size, num_classes))
    encoded[np.arange(labels.size), labels] = 1
    return encoded


def initialize_parameters(input_size, hidden_size, output_size):
    W1 = np.random.randn(input_size, hidden_size) * 0.01
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_size) * 0.01
    b2 = np.zeros((1, output_size))
    return W1, b1, W2, b2

def relu(Z):
    return np.maximum(0, Z)

def relu_derivative(Z):
    return (Z > 0).astype(float)

def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return expZ / expZ.sum(axis=1, keepdims=True)

def forward_propagation(X, W1, b1, W2, b2):
    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = softmax(Z2)
    cache = (Z1, A1, Z2, A2)
    return A2, cache

def backward_propagation(X, Y, cache, W1, W2):
    Z1, A1, Z2, A2 = cache
    m = X.shape[0]

    dZ2 = A2 - Y
    dW2 = np.dot(A1.T, dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m

    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = np.dot(X.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m

    return dW1, db1, dW2, db2

def update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate):
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    return W1, b1, W2, b2

def predict(X, W1, b1, W2, b2):
    A2, _ = forward_propagation(X, W1, b1, W2, b2)
    return np.argmax(A2, axis=1)

def save_model_parameters(W1, b1, W2, b2, filename="model_params.pkl"):
    with open(filename, "wb") as file:
        pickle.dump((W1, b1, W2, b2), file)

def load_model_parameters(filename="model_params.pkl"):
    with open(filename, "rb") as file:
        return pickle.load(file)

def train(X_train, y_train, X_test, y_test, hidden_size=64, learning_rate=0.01, epochs=50, batch_size=128):
    input_size = X_train.shape[1]
    output_size = 10  

    W1, b1, W2, b2 = initialize_parameters(input_size, hidden_size, output_size)

    y_train_encoded = one_hot_encode(y_train, output_size)

    for epoch in range(epochs):
        indices = np.arange(X_train.shape[0])
        np.random.shuffle(indices)
        X_train = X_train[indices]
        y_train_encoded = y_train_encoded[indices]

        for i in range(0, X_train.shape[0], batch_size):
            X_batch = X_train[i:i + batch_size]
            y_batch = y_train_encoded[i:i + batch_size]

            A2, cache = forward_propagation(X_batch, W1, b1, W2, b2)
            dW1, db1, dW2, db2 = backward_propagation(X_batch, y_batch, cache, W1, W2)

            W1, b1, W2, b2 = update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate)

        train_predictions = predict(X_train, W1, b1, W2, b2)
        train_accuracy = np.mean(train_predictions == y_train)

        print(f"Epoch {epoch + 1}/{epochs}, Training Accuracy: {train_accuracy:.4f}")

    save_model_parameters(W1, b1, W2, b2)

    test_predictions = predict(X_test, W1, b1, W2, b2)
    test_accuracy = np.mean(test_predictions == y_test)

    print(f"Test Accuracy: {test_accuracy:.4f}")

    print("Classification Report:")
    print(classification_report(y_test, test_predictions, target_names=[
        'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'
    ]))

if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    X_train = X_train.reshape(X_train.shape[0], -1).astype(np.float32) / 255.0
    X_test = X_test.reshape(X_test.shape[0], -1).astype(np.float32) / 255.0
    y_train = y_train.flatten()
    y_test = y_test.flatten()

    train(X_train, y_train, X_test, y_test)

    W1, b1, W2, b2 = load_model_parameters()
 
    test_image = X_test[0].reshape(1, -1)
    prediction = predict(test_image, W1, b1, W2, b2)
    predicted_class = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'][prediction[0]]

    print(f"The model has predicted: {predicted_class}")


    img = X_test[0].reshape(32, 32, 3)  
    plt.imshow(img)
    plt.title(f"Predicted: {predicted_class}")
    plt.show()