import matplotlib.pyplot as plt

# Valores de precisão originais
accuracies = [18721.63, 18744.26, 18746.28, 18749.58, 18751.24, 18753.65, 18756.33, 18758.50, 18759.88, 18760.85]

# Números de épocas
epochs = list(range(1, 11))  # Número de épocas

max_accuracy = max(accuracies)
min_accuracy = min(accuracies)
normalized_accuracies = [(acc - min_accuracy) / (max_accuracy - min_accuracy) * 62 for acc in accuracies]

# Crie o gráfico de linha
plt.figure(figsize=(10, 6))
plt.plot(epochs, normalized_accuracies, marker='o', linestyle='-', color='b')
plt.title('Progressão da Precisão ao longo das Épocas')
plt.xlabel('Época')
plt.ylabel('Precisão Normalizada (%)')
plt.grid(True)

# Exiba o gráfico
plt.show()
