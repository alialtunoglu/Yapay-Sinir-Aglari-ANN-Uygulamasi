import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from layers import DenseLayer
from activation_functions import Sigmoid, ReLU

# Yapay Sinir Ağı Sınıfı
class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        if self.layers:
            layer.initialize(self.layers[-1].output_size)
        elif layer.input_size is not None:
            layer.initialize(layer.input_size)
        self.layers.append(layer)

    def train(self, inputs, outputs, epochs, learning_rate):
        for epoch in range(epochs):
            
            output = inputs
            for layer in self.layers:
                output = layer.forward(output)
            
            loss = np.mean(np.abs(outputs - output))

            error = outputs - output
            for layer in reversed(self.layers):
                error = layer.backward(error, learning_rate)

            if (epoch + 1) % 100 == 0 or epoch == 0:
                predictions = (output > 0.5).astype(int)
                accuracy = np.mean(predictions == outputs)
                print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f} - Accuracy: {accuracy * 100:.2f}%")


    def predict(self, inputs):
        output = inputs
        for layer in self.layers:
            output = layer.forward(output)
        return output

# Veriyi yükleme ve hazırlama
dir = "/content/drive/MyDrive/YapayZekaDers/"
data_path = "heart_statlog_cleveland_hungary_final.csv"
data = pd.read_csv(data_path)

outputs = data['target'].values.reshape(-1, 1)
inputs = data.drop(['target'], axis=1).values

scaler = StandardScaler()
inputs_normalized = scaler.fit_transform(inputs)


X_train, X_test, y_train, y_test = train_test_split(inputs_normalized, outputs, test_size=0.2, random_state=15)

# Yapay Sinir Ağı Modeli Oluşturma ve Eğitme
nn = NeuralNetwork()
nn.add_layer(DenseLayer(input_size=X_train.shape[1], output_size=10, activation_function=ReLU))  
nn.add_layer(DenseLayer(output_size=5, activation_function=ReLU))   
nn.add_layer(DenseLayer(output_size=1, activation_function=Sigmoid))   

nn.train(X_train, y_train, epochs=1000, learning_rate=0.01)


# Test seti doğruluk oranı
predictions = (nn.predict(X_test) > 0.5).astype(int)
accuracy = np.mean(predictions == y_test)
print(f"Test seti doğruluk oranı: {accuracy * 100:.2f}%")

# Karışıklık Matrisi Görselleştirme
cm = confusion_matrix(y_test, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.show()
