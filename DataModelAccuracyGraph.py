import matplotlib.pyplot as plt

# Valores de precisão originais
accuracies = [1276.09, 1295.97, 1296.39, 1296.65, 1296.79, 1297.04, 1297.24, 1297.41, 1297.60, 1297.82]

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
