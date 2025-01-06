#ifndef BIBLIOTECA_CUDA_H
#define BIBLIOTECA_CUDA_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <curand.h>

// ----------------------------------------------------
// Estructuras para CPU
// ----------------------------------------------------
typedef struct {
    int *genotipo;
    double fitness;
} individuo;

typedef struct {
    individuo *individuos;
    int tamano;
} poblacion;

// ----------------------------------------------------
// Estructuras para GPU
// ----------------------------------------------------
typedef struct {
    int *genotipo;   // Puntero a genotipo en la GPU
    double fitness;  
} individuo_gpu;

typedef struct {
    individuo_gpu *individuos; 
    int tamano;               
} poblacion_gpu;

// ----------------------------------------------------
// Estructura auxiliar para ordenar distancias en GPU
// ----------------------------------------------------
typedef struct {
    double distancia;
    int indice;
} DistanciaOrdenadaGPU;

// ----------------------------------------------------
// Macros / funciones inline para manejo de errores CUDA
// ----------------------------------------------------
void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true);

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

// ----------------------------------------------------
// Prototipos de kernels y funciones CUDA
// ----------------------------------------------------
__global__ void setup_curand_kernel(curandState *states, unsigned long seed);

void obtenerConfiguracionCUDA(int *blockSize, int *minGridSize, int *gridSize, int N);

__global__ void evaluar_poblacion_kernel(individuo_gpu *poblacion, double *distancias, double *ventanas_de_tiempo,
                                         int tamano_poblacion, int longitud_genotipo);

__global__ void seleccionar_padres_kernel(individuo_gpu *poblacion, individuo_gpu *padres,
                                           int num_competidores, int tamano_poblacion,
                                           int longitud_genotipo, curandState *states);

__global__ void cruzar_individuos_kernel(individuo_gpu *padres, individuo_gpu *hijos,
                                         double *distancias, double *ventanas_de_tiempo, double prob_cruce,
                                         int tamano_poblacion, int longitud_genotipo,
                                         int m, curandState *states);

__global__ void mutar_individuos_kernel(individuo_gpu *individuos, double *distancias, double *ventanas_de_tiempo,
                                        double prob_mutacion, int tamano_poblacion,
                                        int longitud_genotipo, curandState *states);

// ----------------------------------------------------
// Funciones device/host auxiliares
// ----------------------------------------------------
__device__ double evaluar_individuo_gpu(int *ruta, double *distancias, double *ventanas_de_tiempo, int num_ciudades);
__device__ void cycle_crossover_device(const int *p1, const int *p2, int *child, int *visitado, int num_ciudades);
__device__ void heuristica_abruptos_gpu(int *ruta, int num_ciudades, int m, double *distancias, double *ventanas_de_tiempo, int *ruta_temp, DistanciaOrdenadaGPU *dist_ordenadas);
__device__ int  comparar_distancias_gpu(DistanciaOrdenadaGPU a, DistanciaOrdenadaGPU b);
__device__ void insertar_en_posicion_gpu(int* array, int longitud, int elemento, int posicion);
__device__ void eliminar_de_posicion_gpu(int* array, int longitud, int posicion);

// ----------------------------------------------------
// Prototipos de funciones para manejo de poblaciones CPU
// ----------------------------------------------------
poblacion *crear_poblacion(int tamano, int longitud_genotipo);
void crear_permutaciones(poblacion *poblacion, int longitud_genotipo);
void ordenar_poblacion(poblacion *poblacion);
void actualizar_poblacion(poblacion **destino, poblacion *origen, int longitud_genotipo);
void liberar_poblacion(poblacion *poblacion);

// ----------------------------------------------------
// Prototipos de funciones para ordenamiento en CPU
// ----------------------------------------------------
void introsort_util(individuo *arr, int *profundidad_max, int inicio, int fin);
int  log2_suelo(int n);
int  particion(individuo *arr, int bajo, int alto);
int  mediana_de_tres(individuo *arr, int a, int b, int c);
void intercambiar_individuos(individuo *a, individuo *b);
void insertion_sort(individuo *arr, int izquierda, int derecha);
void heapsort(individuo *arr, int n);
void heapify(individuo *arr, int n, int i);

// ----------------------------------------------------
// Prototipos de funciones auxiliares para copiar
// poblaciones entre CPU y GPU
// ----------------------------------------------------
void copiarPoblacionCPUaGPU(const poblacion *pobCPU, 
                            individuo_gpu *pobGPU, 
                            int *genotiposGPU,
                            int tamPobl, 
                            int longGen);

void copiarPoblacionGPUaCPU(poblacion *pobCPU, 
                            const individuo_gpu *pobGPU, 
                            const int *genotiposGPU,
                            int tamPobl, 
                            int longGen);

#endif // BIBLIOTECA_CUDA_H
