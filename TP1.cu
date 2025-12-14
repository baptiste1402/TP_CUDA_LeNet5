#include <stdio.h>
#include <stdlib.h>
#include <time.h> 
#include <cuda.h> 

void MatrixInit(float *M, int n, int p);
void MatrixPrint(float *M, int n, int p);
void MatrixAdd(float *M1, float *M2, float *Mout, int n, int p);
__global__ void cudaMatrixAdd(float *M1, float *M2, float *Mout, int n, int p);

int main() {
    // Paramètres n (lignes) et p (colonnes)
    int n = 5; 
    int p = 5; 

    // Allocation de mémoire (Step 1 du TP)
    // On utilise un tableau 1D pour simuler la 2D : taille = n * p
    float *M1 = (float*)malloc(n * p * sizeof(float));
    float *M2 = (float*)malloc(n * p * sizeof(float));
    float *Mout = (float*)malloc(n * p * sizeof(float));

    

    // Initialisation de l'aléatoire
    srand(time(NULL));
 // Initialise le générateur aléatoire
    MatrixInit(M1, n, p);
    MatrixInit(M2, n, p);
    printf("Matrices initialisees.\n");
    // --- Étape 3 : Affichage ---
    printf("Matrice M1 :\n");
    MatrixPrint(M1, n, p);

    printf("Matrice M2 :\n");
    MatrixPrint(M2, n, p);

    // --- Étape 4 : Addition ---
    printf("Calcul de l'addition M1 + M2...\n");
    MatrixAdd(M1, M2, Mout, n, p);

    printf("Resultat Mout :\n");
    MatrixPrint(Mout, n, p);

    // ... LES APPELS DE FONCTIONS IRONT ICI ...

    // Libération de la mémoire à la fin
    
    // --- PARTIE GPU ---
    printf("\n=== DEBUT PARTIE GPU ===\n");

    // 1. Pointeurs pour le GPU (Device)
    float *d_M1, *d_M2, *d_Mout;

    // 2. Allocation mémoire sur le GPU
    // On alloue la même taille que sur le CPU (n * p * sizeof(float))
    cudaMalloc((void**)&d_M1, n * p * sizeof(float));
    cudaMalloc((void**)&d_M2, n * p * sizeof(float));
    cudaMalloc((void**)&d_Mout, n * p * sizeof(float));

    // 3. Copie des données du CPU (Host) vers le GPU (Device)
    cudaMemcpy(d_M1, M1, n * p * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_M2, M2, n * p * sizeof(float), cudaMemcpyHostToDevice);
    // 4. Lancement du Kernel
    // Syntaxe <<< Nombre de Blocs, Nombre de Threads par Bloc >>>
    printf("Lancement du kernel GPU avec %d blocs et %d threads par bloc...\n", n, p);
    cudaMatrixAdd<<<n, p>>>(d_M1, d_M2, d_Mout, n, p);

    // Attendre que le GPU finisse (synchronisation)
    cudaDeviceSynchronize();
    // 5. Copie du résultat du GPU (Device) vers le CPU (Host)
    // On écrase l'ancien contenu de Mout (celui calculé par le CPU)
    cudaMemcpy(Mout, d_Mout, n * p * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Resultat Addition GPU :\n");
    MatrixPrint(Mout, n, p);

    // 6. Libération de la mémoire GPU
    cudaFree(d_M1);
    cudaFree(d_M2);
    cudaFree(d_Mout);
    free(M1);
    free(M2);
    free(Mout);

    return 0;
}

void MatrixInit(float *M, int n, int p) {
    for (int i = 0; i < n * p; i++) {
        float r = (float)rand() / (float)RAND_MAX; 
        M[i] = 2.0f * r - 1.0f; // Entre -1 et 1
    }
}

void MatrixPrint(float *M, int n, int p) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            // Accès linéaire : ligne i, colonne j -> indice i*p + j
            printf("%.2f\t", M[i * p + j]);
        }
        printf("\n"); // Retour à la ligne après chaque ligne de la matrice
    }
    printf("\n"); // Saut de ligne supplémentaire pour la lisibilité
}

void MatrixAdd(float *M1, float *M2, float *Mout, int n, int p) {
    // On parcourt tous les éléments un par un (taille totale = n * p)
    for (int i = 0; i < n * p; i++) {
        // La case i de la sortie est la somme des cases i des entrées
        Mout[i] = M1[i] + M2[i];
    }
}

__global__ void cudaMatrixAdd(float *M1, float *M2, float *Mout, int n, int p) {
    // Selon la consigne :
    // - Lignes (i) correspondent aux blocks -> blockIdx.x
    // - Colonnes (j) correspondent aux threads -> threadIdx.x
    
    int i = blockIdx.x; 
    int j = threadIdx.x;

    // Vérification de sécurité (pour ne pas sortir de la mémoire si p < threads)
    if (i < n && j < p) {
        // Calcul de l'index linéaire (comme dans la version CPU)
        int index = i * p + j;
        
        // Addition
        Mout[index] = M1[index] + M2[index];
    }
}