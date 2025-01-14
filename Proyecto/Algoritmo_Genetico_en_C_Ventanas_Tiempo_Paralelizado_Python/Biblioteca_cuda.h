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
#include <math.h>


// ----------------------------------------------------
// Estructuras para CPU
// ----------------------------------------------------

// Estructura para un individuo en la CPU
// Almacena el genotipo y el fitness del individuo
typedef struct {
    int *genotipo;      // Arreglo que representa el genotipo del individuo
    double fitness;     // Valor de fitness del individuo
} individuo;

// Estructura para una población en la CPU
// Almacena un arreglo de individuos y su tamaño
typedef struct {
    individuo *individuos;  // Arreglo de individuos
    int tamano;             // Tamaño de la población
} poblacion;

// ----------------------------------------------------
// Estructuras para GPU
// ----------------------------------------------------

// Estructura para un individuo en la GPU
// Almacena el puntero al genotipo en la GPU y el fitness
typedef struct {
    int *genotipo;   // Puntero al genotipo en la GPU
    double fitness;  // Valor de fitness del individuo
} individuo_gpu;

// Estructura para una población en la GPU
// Almacena un arreglo de individuos en la GPU y su tamaño
typedef struct {
    individuo_gpu *individuos; // Arreglo de individuos en la GPU
    int tamano;                // Tamaño de la población
} poblacion_gpu;

// ----------------------------------------------------
// Estructura auxiliar para ordenar distancias en GPU
// ----------------------------------------------------

// Estructura para almacenar una distancia y su índice correspondiente en la GPU (Heuristica de Abruptos)
typedef struct {
    double distancia; // Valor de la distancia
    int indice;       // Índice asociado a la distancia
} DistanciaOrdenadaGPU;

// Para la reducción del mínimo (fitness, idx)
struct MinData {
    double fitness;
    int    idx;
};

// ----------------------------------------------------
// Macros / funciones inline para manejo de errores CUDA
// ----------------------------------------------------

// Función para verificar y manejar errores de CUDA
void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true);

// Macro para simplificar la verificación de errores CUDA
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

// ----------------------------------------------------
// Prototipos de kernels y funciones CUDA
// ----------------------------------------------------

// Kernel para inicializar los estados de curand en la GPU
__global__ void setup_curand_kernel(curandState *states, unsigned long seed);

// Función para obtener la configuración óptima de bloques y grids en CUDA
void obtenerConfiguracionCUDA(int *blockSize, int *minGridSize, int *gridSize, int N);

// Kernel para evaluar la población en la GPU
__global__ void evaluar_poblacion_kernel(individuo_gpu *poblacion, double *distancias, double *ventanas_de_tiempo,
                                         int tamano_poblacion, int longitud_genotipo);

// Kernel para seleccionar padres mediante torneo en la GPU
__global__ void seleccionar_padres_kernel(individuo_gpu *poblacion, individuo_gpu *padres,
                                           int num_competidores, int tamano_poblacion,
                                           int longitud_genotipo, curandState *states);

// Kernel para cruzar individuos y generar hijos en la GPU
__global__ void cruzar_individuos_kernel(individuo_gpu *padres, individuo_gpu *hijos,
                                         double *distancias, double *ventanas_de_tiempo, double prob_cruce,
                                         int tamano_poblacion, int longitud_genotipo,
                                         int m, curandState *states);

// Kernel para mutar individuos en la GPU
__global__ void mutar_individuos_kernel(individuo_gpu *individuos, double *distancias, double *ventanas_de_tiempo,
                                        double prob_mutacion, int tamano_poblacion,
                                        int longitud_genotipo, curandState *states);

// Kernel para actualizar a los hijos como la nueva poblacion
__global__ void actualizar_poblacion_kernel(individuo_gpu *destino,
                                            individuo_gpu *origen,
                                            int tamano_poblacion,
                                            int longitud_genotipo);

// Función auxiliar para obtener el mejor individuo (índice + fitness) en GPU
// Sus funciones auxiliares son reduce_find_minphase1 y phase2
void buscarMejorIndividuoEnGPU(individuo_gpu *d_poblacion,
                               int tamano_poblacion,
                               int blockSize,
                               MinData *d_result);

// Kernel para encontrar los valores minimos de un bloque y su indice
__global__ void reduce_find_min_phase1(const individuo_gpu *d_poblacion,
                                       int tamano_poblacion,
                                       MinData *d_parciales);

// Kernel para encontrar los valores minimos de un conjunto de bloques y su indice
__global__ void reduce_find_min_phase2(const MinData *d_parciales_in,
                                       int n,
                                       MinData *d_result);

// ----------------------------------------------------
// Funciones device/host auxiliares
// ----------------------------------------------------

// Función device para evaluar un individuo en la GPU
__device__ double evaluar_individuo_gpu(int *ruta, double *distancias, double *ventanas_de_tiempo, int num_ciudades);

// Función device para realizar el cruzamiento de ciclo en la GPU
__device__ void cycle_crossover_device(const int *p1, const int *p2, int *child, int *visitado, int num_ciudades);

// Función device para aplicar la heurística de abruptos en la GPU
__device__ void heuristica_abruptos_gpu(int *ruta, int num_ciudades, int m, double *distancias, double *ventanas_de_tiempo, int *ruta_temp, DistanciaOrdenadaGPU *dist_ordenadas);

// Función device para comparar distancias en la GPU
__device__ int comparar_distancias_gpu(DistanciaOrdenadaGPU a, DistanciaOrdenadaGPU b);

// Función device para insertar un elemento en una posición específica del arreglo en la GPU
__device__ void insertar_en_posicion_gpu(int* array, int longitud, int elemento, int posicion);

// Función device para eliminar un elemento de una posición específica del arreglo en la GPU
__device__ void eliminar_de_posicion_gpu(int* array, int longitud, int posicion);

// ----------------------------------------------------
// Prototipos de funciones para manejo de poblaciones CPU
// ----------------------------------------------------

// Función para crear una población en la CPU
poblacion *crear_poblacion(int tamano, int longitud_genotipo);

// Función para generar permutaciones aleatorias para cada individuo en la población
void crear_permutaciones(poblacion *poblacion, int longitud_genotipo);

// Función para liberar la memoria asignada a una población en la CPU
void liberar_poblacion(poblacion *poblacion);

// ----------------------------------------------------
// Prototipos de funciones auxiliares para copiar
// poblaciones entre CPU y GPU
// ----------------------------------------------------

// Función para copiar una población desde la CPU a la GPU
void copiarPoblacionCPUaGPU(const poblacion *pobCPU, 
                            individuo_gpu *pobGPU, 
                            int *genotiposGPU,
                            int tamPobl, 
                            int longGen);

// Función para copiar una población desde la GPU a la CPU
void copiarPoblacionGPUaCPU(poblacion *pobCPU, 
                            const individuo_gpu *pobGPU, 
                            const int *genotiposGPU,
                            int tamPobl, 
                            int longGen);

#endif // BIBLIOTECA_CUDA_H