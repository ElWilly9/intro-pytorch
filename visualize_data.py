import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

# Generar datos
X, y = make_moons(n_samples=1000, noise=0.2, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Estandarizar los datos
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convertir a tensores de PyTorch
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Definir y entrenar el modelo
class SimpleNet(torch.nn.Module):
    def __init__(self, input_dim=2, hidden_dim=16, output_dim=2):
        super(SimpleNet, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.model(x)

# Crear y entrenar el modelo
model = SimpleNet()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Entrenamiento
n_epochs = 100
batch_size = 32
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

for epoch in range(n_epochs):
    model.train()
    for features, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Crear una malla de puntos para la frontera de decisión
x_min, x_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
y_min, y_max = X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# Predecir la clase para cada punto en la malla
grid_points = np.c_[xx.ravel(), yy.ravel()]
grid_points_tensor = torch.tensor(grid_points, dtype=torch.float32)
model.eval()
with torch.no_grad():
    grid_predictions = model(grid_points_tensor)
    grid_predictions = torch.argmax(grid_predictions, dim=1).numpy()

# Reshape las predicciones para que coincidan con la forma de la malla
grid_predictions = grid_predictions.reshape(xx.shape)

# Crear la visualización
plt.figure(figsize=(12, 5))

# Subplot para datos de entrenamiento con frontera de decisión
plt.subplot(1, 2, 1)
plt.contourf(xx, yy, grid_predictions, alpha=0.3, cmap='RdYlBu')
plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], c='blue', label='Clase 0')
plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], c='red', label='Clase 1')
plt.title('Datos de Entrenamiento con Frontera de Decisión')
plt.xlabel('Característica 1')
plt.ylabel('Característica 2')
plt.legend()

# Subplot para datos de prueba con frontera de decisión
plt.subplot(1, 2, 2)
plt.contourf(xx, yy, grid_predictions, alpha=0.3, cmap='RdYlBu')
plt.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1], c='blue', label='Clase 0')
plt.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], c='red', label='Clase 1')
plt.title('Datos de Prueba con Frontera de Decisión')
plt.xlabel('Característica 1')
plt.ylabel('Característica 2')
plt.legend()

plt.tight_layout()
plt.show() 