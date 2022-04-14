#!/usr/bin/python3.9
# Licence MIT copyright (c) 2022 Stéphane Lassalvy
import numpy as np
from scipy.linalg import block_diag
def my_SquaredNorm(vector):
    # Norme du vecteur "vector"
    result = np.dot(np.transpose(vector), vector)
    return(result)

def my_Norm(vector):
    result = np.sqrt(my_SquaredNorm(vector))
    return result

def my_Projector(vector, onVector):
    # Projection orthogonale du vecteur "vector" sur le vecteur "onVector"
    result = np.dot(vector, onVector) * (onVector / my_SquaredNorm(onVector))
    return(result)

def my_HouseholderMatrix(squareMatrix):
    squareMatrix = np.matrix(squareMatrix)
    x = squareMatrix[0:len(squareMatrix), 0]
    alpha = float(- x[0] / np.abs(x[0]) * my_Norm(x))
    ei = [0] * len(x)
    ei[0] = 1
    ei = np.transpose(np.matrix(ei))
    u = x - alpha * ei
    v = np.matrix(u / my_Norm(u))
    I = np.identity(len(squareMatrix))
    Qi = I - 2 * np.dot(v, np.transpose(v))
    return Qi

def my_HouseholderDecomposition(squareMatrix):
    # Algorithme de Householder
    # squareMatrix : une matrice carrée de plein rang
    #
    # Sortie
    # Q : matrice de la décomposition A = QR où Q est une matrice orthonormale telle que t(Q)Q = Identité
    print("Décomposition d'une matrice carrée de plein rang A sous la forme A = QR par la méthode de Householder")
    print("Q matrice orthonormale,")
    print("R matrice triangulaire supérieure.")
    print("")
    print("Base de départ, matrice A :")
    squareMatrix = np.matrix(squareMatrix)
    print(squareMatrix)
    print("")
    t = len(squareMatrix) - 1
    QListe = [0] * t
    Q1 = my_HouseholderMatrix(squareMatrix)
    print("Matrice Q1")
    print(Q1)
    A1 = np.dot(Q1 , squareMatrix)
    Aprime = A1[1:len(A1), 1:len(A1)]
    QListe[0] = Q1
    for i in range(2, len(squareMatrix)):
        print(f"Itération {i}")
        Qi = my_HouseholderMatrix(Aprime)
        Identitei = np.identity(i-1)
        Ai = np.dot(Qi, Aprime)
        Aprime = Ai[1:len(Ai), 1:len(Ai)]
        Qi = np.matrix(block_diag(Identitei, Qi))
        print("Matrice Qi")
        print(Qi)
        QListe[i-1] = Qi
    Q = QListe[0]
    for i in range(1, len(QListe)):
        Q = np.dot(Q, QListe[i])
    print("Matrice Q :")
    print(Q)
    R = np.matrix(np.round(np.dot(np.transpose(Q), squareMatrix), 3))
    print("Matrice R :")
    print(R)
    print("Produit QR :")
    QR = np.dot(Q,R)
    print(QR)
    print("QR arrondi à 3 décimales :")
    print(np.round(QR,3))
    return QListe

# Exécution du programme sur un exemple
my_mat = [[1,2,3], [4,5,6], [7,8,17]]
my_mat = np.array(my_mat)
my_HouseholderDecomposition(my_mat)
