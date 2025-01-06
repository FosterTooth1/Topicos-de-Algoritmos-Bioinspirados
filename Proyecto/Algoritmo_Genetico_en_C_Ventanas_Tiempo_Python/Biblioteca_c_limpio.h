#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<time.h>
#include<math.h>

// Estructuras
// Estructura para un individuon (Almacena el genotipo y el fitness)
typedef struct{
    int *genotipo;
    double fitness;
}individuo;

// Estructura para una población (Almacena un arreglo de individuos y su tamaño)
typedef struct{
    individuo *individuos;
    int tamano;
}poblacion;

// Estructura para ordenar distancias (Almacena la distancia y el índice)(Usado en la heurística de remoción de abruptos)
typedef struct {
    double distancia;
    int indice;
} DistanciaOrdenada;

//Funciones principales del algoritmo genético
//Asigna memoria para una población
poblacion *crear_poblacion(int tamano, int longitud_genotipo);
//Crea permutaciones aleatorias para cada individuo de la población
void crear_permutaciones(poblacion *poblacion, int longitud_genotipo);
//Evalúa a la población basandose en las distancias entre las ciudades
void evaluar_poblacion(poblacion *poblacion, double **distancias, int longitud_genotipo, double ventanas_tiempo[][2]);
//Evalúa a un individuo basandose en las distancias entre las ciudades
double evaluar_individuo(int *individuo, double **distancias, int longitud_genotipo, double ventanas_tiempo[][2]);
//Ordena a la población de acuerdo a su fitness mediante el algoritmo de introsort
void ordenar_poblacion(poblacion *poblacion);
//Selecciona a los padres de la población mediante un torneo de fitness
void seleccionar_padres_torneo(poblacion *Poblacion, poblacion *padres, int num_competidores, int longitud_genotipo);
//Cruza a los padres para generar a los hijos dependiendo de una probabilidad de cruce
void cruzar_individuos(poblacion *padres, poblacion *hijos, int num_pob, int longitud_genotipo, int m, double **distancias, double probabilidad_cruce, double ventanas_tiempo[][2]);
//Muta a un individuo dependiendo de una probabilidad de mutación
void mutar_individuo(individuo *individuo, double **distancias, double probabilidad_mutacion, int longitud_genotipo, double ventanas_tiempo[][2]);
//Actualiza a la población con los nuevos individuos (hijos)
void actualizar_poblacion(poblacion **destino, poblacion *origen, int longitud_genotipo);
//Libera la memoria usada para la población
void liberar_poblacion(poblacion *poblacion);


//Funciones auxiliares del cruzamiento
//La heurística se encarga de remover abruptos en la ruta intercamdiando ciudades mal posicionadas
void heuristica_abruptos(int *ruta, int num_ciudades, int m, double **distancias, double ventanas_tiempo[][2]);
//Cycle crossover se encarga de cruzar a dos padres para generar a dos hijos
void cycle_crossover(int *padre1, int *padre2, int *hijo, int num_ciudades);

//Funciones auxiliares de ordenamiento
// Introsort es un algoritmo de ordenamiento híbrido que combina QuickSort, HeapSort e InsertionSort
void introsort_util(individuo *arr, int *profundidad_max, int inicio, int fin);
//Calcula el logaritmo base 2 de un número para medir la profundidad de recursividad que puede alcanzar QuickSort
int log2_suelo(int n);
//Particiona el arreglo para el QuickSort (Funcion auxiliar de Introsort en especifico para el QuickSort)
int particion(individuo *arr, int bajo, int alto);
//Calcula la mediana de tres elementos (Funcion auxiliar de Introsort en especifico para el QuickSort)
int mediana_de_tres(individuo *arr, int a, int b, int c);
//Intercambia dos individuos (Funcion auxiliar de Introsort en especifico para el QuickSort)
void intercambiar_individuos(individuo *a, individuo *b);
//Insertion sort es un algoritmo de ordenamiento simple y eficiente para arreglos pequeños
void insertion_sort(individuo *arr, int izquierda, int derecha);
//Heapsort es un algoritmo de ordenamiento basado en árboles binarios
void heapsort(individuo *arr, int n);
//Heapify es una función auxiliar para heapsort
void heapify(individuo *arr, int n, int i);

//Funciones auxiliares de manipulación de arreglos (Usadas en la heurística de remoción de abruptos)
//Compara dos distancias para ordenarlas
int comparar_distancias(const void* a, const void* b);
//Inserta un elemento en una posición específica del arreglo
void insertar_en_posicion(int* array, int longitud, int elemento, int posicion);
//Elimina un elemento de una posición específica del arreglo
void eliminar_de_posicion(int* array, int longitud, int posicion);