#!/usr/bin/python3.9
# Licence GPL3 copyright (c) 2022 Stéphane Lassalvy
import numpy as np
def my_SquaredNorm(vector):
    # Norme du vecteur "vector"
    result = np.dot(vector, vector)
    return(result)

def my_Projector(vector, onVector):
    # Projection orthogonale du vecteur "vector" sur le vecteur "onVector"
    result = np.dot(vector, onVector) * (onVector / my_SquaredNorm(onVector))
    return(result)

def my_GramSchmidt(squareMatrix):
    # Algorithme de Gram-Schmith d'orthonormalisation d'une base d'un Espace Vectoriel
    # Entrée :
    # squareMatrix : une matrice de plein range
    #
    # Sortie
    # Q : matrice de la décomposition A = QR où Q est une matrice orthonormale telle que t(Q)Q = Identit
    print("Décomposition d'une matrice de plein rang A sous la forme A = QR")
    print("Q matrice orthonormale,")
    print("R matrice triangulaire supérieure.")
    print("")
    print("Base de départ, matrice A :")
    print(squareMatrix)
    print("")
    transposeMatrix = squareMatrix.T
    a1 = transposeMatrix[0]
    u1 = a1
    e1 = u1 / np.sqrt(my_SquaredNorm(u1))
    eiMatrix = [e1]
    length = len(transposeMatrix)
    for i in range(1, length):
        ai = transposeMatrix[i]
        ui = ai
        for j in range(len(eiMatrix)):
            ej = eiMatrix[j]
            ui = ui - my_Projector(ai, ej)
        ei = np.array(ui / np.sqrt(my_SquaredNorm(ui)))
        eiMatrix = np.vstack((eiMatrix, ei))
    eiMatrix = eiMatrix.T
    Q = eiMatrix
    print("Base orthonormalisée, matrice Q :")
    print(Q)
    print("")
    print("Vérification de l'orthonormalité de Q, calcul de t(Q)Q :")
    reconstructionA = np.round(np.dot(Q.T, Q),3)
    print(reconstructionA)
    print("")
    # Matrice R
    print("Matrice diagonale supérieure R de la décomposition A = QR :")
    TR = [[0 for _ in range(length)] for _ in range(length)]
    for i in range(length):
        for j in range(length):
            ai = transposeMatrix[i]
            ej = Q.T[j]
            TR[i][j] = np.dot(ai, ej)
    TR = np.array(TR)
    R = TR.T
    print(np.round(R,3))
    print("")
    print("Produit QR :")
    print(np.dot(Q,R))
    return(Q)

# Exécution du programme sur un exemple
my_mat = [[1,2,3], [4,5,6], [7,8,17]]
my_mat = np.array(my_mat)
my_GramSchmidt(my_mat)