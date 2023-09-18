import os
import csv
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

# Função para carregar os dados
def load_data(train_file, test_file):
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)
    return train_data, test_data

# Função para preprocessar os dados
def preprocess_data(train_data, test_data):
    # Combine as colunas relevantes em uma única coluna de texto
    train_data['combined_text'] = train_data['movie_name'] + ' ' + train_data['synopsis'] + ' ' + train_data['genre']
    test_data['combined_text'] = test_data['movie_name'] + ' ' + test_data['synopsis']

    # Obtenha a lista de todos os gêneros únicos em ambos os conjuntos de treinamento e teste
    all_genres = set()
    for genres in train_data['genre']:
        all_genres.update(genres.split(', '))
    for genres in test_data['genre']:
        all_genres.update(genres.split(', '))

    # Mapear gêneros únicos para índices
    genre_to_index = {genre: i for i, genre in enumerate(all_genres)}

    # Tokenize as avaliações de filmes usando CountVectorizer
    vectorizer = CountVectorizer(max_features=200)
    X_train = vectorizer.fit_transform(train_data['combined_text'])
    X_test = vectorizer.transform(test_data['combined_text'])

    # Transforme as matrizes esparsas em matrizes densas
    X_train = X_train.astype(np.float32).toarray()  # Altere o tipo de dado para float32
    X_test = X_test.astype(np.float32).toarray()    # Altere o tipo de dado para float32

    # Crie vetores de gênero para cada filme
    y_train = np.zeros((len(train_data), len(all_genres)), dtype=int)
    y_test = np.zeros((len(test_data), len(all_genres)), dtype=int)

    for i, genres in enumerate(train_data['genre']):
        for genre in genres.split(', '):
            y_train[i, genre_to_index[genre]] = 1

    for i, genres in enumerate(test_data['genre']):
        for genre in genres.split(', '):
            y_test[i, genre_to_index[genre]] = 1

    return X_train, y_train, X_test, y_test

# Função para criar um DataLoader
def create_dataloader(X, y, batch_size=32, shuffle=True):
    data = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle)
    return loader

# Função para treinar o modelo
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total += batch_y.size(0)
            correct += ((output >= 0.5).float() == batch_y).sum().item()

        accuracy = 100 * correct / total
        print(f"Epoch {epoch + 1}: Loss: {total_loss:.4f}, Accuracy: {accuracy:.2f}%")

# Função para salvar a precisão em um arquivo CSV
def save_accuracy_to_csv(filename, accuracy_values):
    try:
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Epoch", "Accuracy"])
            for epoch, accuracy in enumerate(accuracy_values, start=1):
                writer.writerow([epoch, accuracy])
        print(f"Valores de precisão salvos em {filename}")
    except Exception as e:
        print(f"Erro ao salvar o arquivo CSV: {str(e)}")

if __name__ == "__main__":
    # Carregar os dados
    train_data, test_data = load_data('train.csv', 'test.csv')

    # Preprocessar os dados
    X_train, y_train, X_test, y_test = preprocess_data(train_data, test_data)

    # Criar DataLoader para dados de treinamento
    train_loader = create_dataloader(X_train, y_train, batch_size=32, shuffle=True)

    # Definir a arquitetura da rede neural
    vocab_size = X_train.shape[1]
    num_genres = y_train.shape[1]
    model = nn.Sequential(
        nn.Linear(vocab_size, 100),
        nn.ReLU(),
        nn.Linear(100, num_genres),
        nn.Sigmoid()
    )

    # Definir função de perda e otimizador
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Treinar o modelo
    accuracy_values = []
    train_model(model, train_loader, criterion, optimizer, num_epochs=10)

    # Salvar a precisão em um arquivo CSV
    save_accuracy_to_csv("accuracy_values.csv", accuracy_values)
