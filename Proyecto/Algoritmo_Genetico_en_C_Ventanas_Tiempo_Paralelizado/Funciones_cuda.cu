#include "Biblioteca_cuda.h"
#include <math.h> // para log2, etc.

// ----------------------------------------------------
// Implementación de la macro para manejo de errores
// ----------------------------------------------------
// Definición de la función
void gpuAssert(cudaError_t code, const char *file, int line, bool abort) {
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// ----------------------------------------------------
// Kernels y funciones CUDA
// ----------------------------------------------------
__global__ void setup_curand_kernel(curandState *states, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, idx, 0, &states[idx]);
}

void obtenerConfiguracionCUDA(int *blockSize, int *minGridSize, int *gridSize, int N) {
    cudaOccupancyMaxPotentialBlockSize(minGridSize, blockSize, evaluar_poblacion_kernel, 0, N);
    *gridSize = (N + *blockSize - 1) / *blockSize;
}

__global__ void evaluar_poblacion_kernel(individuo_gpu *poblacion, double *distancias, double *ventanas_tiempo,
                                         int tamano_poblacion, int longitud_genotipo) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= tamano_poblacion) return;

    double total_cost = 0.0;      // Costo total del recorrido (en horas)
    double tiempo_acumulado = 0.0; // Tiempo transcurrido desde el inicio del recorrido

    int *genotipo = poblacion[idx].genotipo;

    // Iteramos sobre las ciudades en el genotipo de forma circular
    for (int i = 0; i < longitud_genotipo; i++) {
        int origen = genotipo[i];
        int destino = genotipo[(i + 1) % longitud_genotipo];

        // Añadimos el tiempo de viaje entre ciudades
        double tiempo_viaje = distancias[origen * longitud_genotipo + destino];
        tiempo_acumulado += tiempo_viaje;

        // Calculamos la hora de llegada al destino (ajustado al formato de 24 horas)
        double hora_llegada = fmod(tiempo_acumulado, 24.0);

        // Ventanas de tiempo de la ciudad de destino
        double ventana_inicio = ventanas_tiempo[destino * 2];
        double ventana_fin = ventanas_tiempo[destino * 2 + 1];

        // Ajustamos el tiempo acumulado si se llega fuera de la ventana permitida
        if (hora_llegada < ventana_inicio) {
            // Esperamos hasta el inicio de la ventana
            tiempo_acumulado += (ventana_inicio - hora_llegada);
        } else if (ventana_fin < ventana_inicio) {
            // Caso especial: La ventana cruza medianoche (ej. 22:00 a 02:00)
            if (hora_llegada > ventana_fin && hora_llegada < ventana_inicio) {
                tiempo_acumulado += (24.0 - hora_llegada + ventana_inicio); // Esperamos al siguiente día
            }
        } else if (hora_llegada > ventana_fin) {
            // Esperamos al siguiente día si llegamos después del cierre
            tiempo_acumulado += (24.0 - hora_llegada + ventana_inicio);
        }

        // Añadimos al costo total
        total_cost += tiempo_viaje;
    }

    // Guardamos el fitness en la población
    poblacion[idx].fitness = total_cost;
}

__global__ void seleccionar_padres_kernel(individuo_gpu *poblacion, individuo_gpu *padres,
                                          int num_competidores, int tamano_poblacion,
                                          int longitud_genotipo, curandState *states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= tamano_poblacion) return;

    int mejor_idx = -1;
    double mejor_fitness = 1e9;

    for (int i = 0; i < num_competidores; i++) {
        int rand_idx = curand(&states[idx]) % tamano_poblacion;
        if (poblacion[rand_idx].fitness < mejor_fitness) {
            mejor_fitness = poblacion[rand_idx].fitness;
            mejor_idx = rand_idx;
        }
    }

    // Copiar el mejor individuo al arreglo de padres
    for (int j = 0; j < longitud_genotipo; j++) {
        padres[idx].genotipo[j] = poblacion[mejor_idx].genotipo[j];
    }
    padres[idx].fitness = poblacion[mejor_idx].fitness;
}

__global__ void cruzar_individuos_kernel(individuo_gpu *padres, individuo_gpu *hijos,
                                         double *distancias, double *ventanas_de_tiempo, double prob_cruce,
                                         int tamano_poblacion, int longitud_genotipo,
                                         int m, curandState *states)
{
    // 1) Cada bloque tiene "blockDim.x" hilos. 
    //    Usamos "extern __shared__ int sMem[]" para la memoria compartida dinámica.
    extern __shared__ unsigned char sMem[]; 


    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= tamano_poblacion / 2) return;

    // blockSize = blockDim.x
    // Cada hilo "i" dentro del bloque usará un trozo de sMem

    size_t espacioCrossover = 3UL * longitud_genotipo * sizeof(int);  // hijo1,hijo2,visitado
    size_t espacioHeurRuta  = (size_t)longitud_genotipo * sizeof(int);
    size_t espacioHeurDist  = (size_t)longitud_genotipo * sizeof(DistanciaOrdenadaGPU);

    size_t totalPorHilo = espacioCrossover + espacioHeurRuta + espacioHeurDist;

    // offset para ESTE hilo (threadIdx.x)
    size_t offset = (size_t)threadIdx.x * totalPorHilo; 
    unsigned char* ptrBase = &sMem[offset];

    // 1) Crossover
    // a) hijo1: int[longitud_genotipo]
    int* hijo1 = reinterpret_cast<int*>(ptrBase);
    ptrBase += longitud_genotipo * sizeof(int);

    // b) hijo2: int[longitud_genotipo]
    int* hijo2 = reinterpret_cast<int*>(ptrBase);
    ptrBase += longitud_genotipo * sizeof(int);

    // c) visitado: int[longitud_genotipo]
    int* visitado = reinterpret_cast<int*>(ptrBase);
    ptrBase += longitud_genotipo * sizeof(int);

    // 2) Heurística
    // a) ruta_temp: int[longitud_genotipo]
    int* ruta_temp = reinterpret_cast<int*>(ptrBase);
    ptrBase += longitud_genotipo * sizeof(int);

    // b) dist_ordenadas: DistanciaOrdenadaGPU[longitud_genotipo]
    DistanciaOrdenadaGPU* dist_ordenadas = reinterpret_cast<DistanciaOrdenadaGPU*>(ptrBase);
    ptrBase += longitud_genotipo * sizeof(DistanciaOrdenadaGPU);

    // idx2 indica qué par (padre1, padre2) estamos trabajando
    int idx2 = idx * 2;

    // Decidimos si hacemos cruce
    if (curand_uniform(&states[idx]) < prob_cruce)
    {
        // 1) Generar hijo1 con cycle_crossover_device(padre1, padre2)
        cycle_crossover_device(
            padres[idx2].genotipo,
            padres[idx2 + 1].genotipo,
            hijo1,         // => en shared memory
            visitado,      // => en shared memory
            longitud_genotipo
        );

        // 2) Generar hijo2 con cycle_crossover_device(padre2, padre1)
        //    *Pero* hay que "reiniciar" "visitado" antes de reusar. Lo más fácil:
        //    reusar la misma "visitado[]" si deseas, o usar un trozo distinto.
        //    Aquí, por simplicidad, volvemos a poner en 0:
        for (int i = 0; i < longitud_genotipo; i++) {
            visitado[i] = 0;
        }
        cycle_crossover_device(
            padres[idx2 + 1].genotipo,
            padres[idx2].genotipo,
            hijo2,
            visitado,
            longitud_genotipo
        );

        // (Opcional) Llamar heurística:
        heuristica_abruptos_gpu(hijo1, longitud_genotipo, m, distancias, ventanas_de_tiempo,ruta_temp, dist_ordenadas);

        heuristica_abruptos_gpu(hijo2, longitud_genotipo, m, distancias, ventanas_de_tiempo,ruta_temp, dist_ordenadas);

        // 3) Evaluar padres e hijos
        double fit_p1 = evaluar_individuo_gpu(padres[idx2].genotipo, distancias, ventanas_de_tiempo, longitud_genotipo);
        double fit_p2 = evaluar_individuo_gpu(padres[idx2+1].genotipo, distancias, ventanas_de_tiempo, longitud_genotipo);
        double fit_h1 = evaluar_individuo_gpu(hijo1, distancias, ventanas_de_tiempo, longitud_genotipo);
        double fit_h2 = evaluar_individuo_gpu(hijo2, distancias, ventanas_de_tiempo, longitud_genotipo);

        // Seleccionamos 2 mejores
        double fitness_array[4] = { fit_p1, fit_p2, fit_h1, fit_h2 };
        int *genotipos[4]       = { padres[idx2].genotipo,
                                    padres[idx2+1].genotipo,
                                    hijo1,
                                    hijo2 };

        int mejores[2] = {0, 1};
        for (int j = 2; j < 4; j++) {
            if (fitness_array[j] < fitness_array[mejores[0]]) {
                mejores[1] = mejores[0];
                mejores[0] = j;
            }
            else if (fitness_array[j] < fitness_array[mejores[1]]) {
                mejores[1] = j;
            }
        }

        // Copiar a hijos finales
        for(int j = 0; j < longitud_genotipo; j++) {
            hijos[idx2].genotipo[j]   = genotipos[mejores[0]][j];
            hijos[idx2+1].genotipo[j] = genotipos[mejores[1]][j];
        }
        hijos[idx2].fitness   = fitness_array[mejores[0]];
        hijos[idx2+1].fitness = fitness_array[mejores[1]];
    }
    else
    {
        // Sin cruce, copiamos padres
        for (int i = 0; i < longitud_genotipo; i++) {
            hijos[idx2].genotipo[i]   = padres[idx2].genotipo[i];
            hijos[idx2+1].genotipo[i] = padres[idx2+1].genotipo[i];
        }
        hijos[idx2].fitness   = padres[idx2].fitness;
        hijos[idx2+1].fitness = padres[idx2+1].fitness;
    }
}

__global__ void mutar_individuos_kernel(individuo_gpu *individuos, double *distancias, double *ventanas_de_tiempo,
                                        double prob_mutacion, int tamano_poblacion,
                                        int longitud_genotipo, curandState *states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= tamano_poblacion) return;

    if (curand_uniform(&states[idx]) < prob_mutacion) {
        int idx1 = (int)(curand_uniform(&states[idx]) * longitud_genotipo);
        int idx2 = (int)(curand_uniform(&states[idx]) * longitud_genotipo);
        while (idx2 == idx1) {
            idx2 = (int)(curand_uniform(&states[idx]) * longitud_genotipo);
        }
        int temp = individuos[idx].genotipo[idx1];
        individuos[idx].genotipo[idx1] = individuos[idx].genotipo[idx2];
        individuos[idx].genotipo[idx2] = temp;

        // recalcular fitness local
        double total_cost = 0.0;
        for (int i = 0; i < longitud_genotipo - 1; i++) {
            total_cost += distancias[individuos[idx].genotipo[i] * longitud_genotipo +
                                     individuos[idx].genotipo[i+1]];
        }
        total_cost += distancias[individuos[idx].genotipo[longitud_genotipo - 1] * 
                                 longitud_genotipo + individuos[idx].genotipo[0]];
        individuos[idx].fitness = total_cost;
    }
}

// ----------------------------------------------------
// Funciones device auxiliares
// ----------------------------------------------------

/// --------------------------------------------------------------------------
/// cycle_crossover_device:
///   Versión "device" de tu ciclo de cruce en CPU.
///   Genera 1 hijo en el array "child" (size = num_ciudades)
///   usando padre1 "p1" y padre2 "p2" (también arrays de size=num_ciudades).
/// --------------------------------------------------------------------------

// ---------------------------------------------------------------------
// cycle_crossover_device: Sin "new[]".
// Usa "child[]" y "visitado[]" que fueron asignados en shared memory.
// ---------------------------------------------------------------------
__device__ void cycle_crossover_device(const int *p1, const int *p2,
                                       int *child, int *visitado,
                                       int num_ciudades)
{
    // 1) Inicializa child con -1, visitado = 0
    for (int i = 0; i < num_ciudades; i++) {
        child[i]   = -1;
        visitado[i] = 0;  // 0 => no visitado
    }

    int ciclo = 0;
    int restantes = num_ciudades;

    // 2) Mientras queden posiciones sin visitar
    while (restantes > 0) {
        // encontrar primera posición no visitada
        int inicio = -1;
        for (int i = 0; i < num_ciudades; i++) {
            if (visitado[i] == 0) {
                inicio = i;
                break;
            }
        }

        ciclo++;
        int actual = inicio;

        // 3) Recorremos el ciclo
        while (true) {
            visitado[actual] = 1;
            restantes--;

            // En ciclos impares copiamos de p1, en pares de p2
            if (ciclo % 2 == 1) {
                child[actual] = p1[actual];
            } else {
                child[actual] = p2[actual];
            }

            // "valor_buscar" = p2[actual], lo buscamos en p1
            int valor_buscar = p2[actual];
            int siguiente = -1;
            for (int j = 0; j < num_ciudades; j++) {
                if (p1[j] == valor_buscar) {
                    siguiente = j;
                    break;
                }
            }
            if (siguiente == -1 || visitado[siguiente] == 1) {
                break;
            }
            actual = siguiente;
        }
    }
}

__device__ double evaluar_individuo_gpu(int *ruta, double *distancias, double *ventanas_de_tiempo, int num_ciudades) {
    double total_cost = 0.0;      // Costo total del recorrido (en horas)
    double tiempo_acumulado = 0.0; // Tiempo transcurrido desde el inicio del recorrido

    // Iteramos sobre las ciudades en la ruta de forma circular
    for (int i = 0; i < num_ciudades; i++) {
        int origen = ruta[i];
        int destino = ruta[(i + 1) % num_ciudades];

        // Añadimos el tiempo de viaje entre ciudades
        double tiempo_viaje = distancias[origen * num_ciudades + destino];
        tiempo_acumulado += tiempo_viaje;

        // Calculamos la hora de llegada al destino (ajustado al formato de 24 horas)
        double hora_llegada = fmod(tiempo_acumulado, 24.0);

        // Ventanas de tiempo de la ciudad de destino
        double ventana_inicio = ventanas_de_tiempo[destino * 2];
        double ventana_fin = ventanas_de_tiempo[destino * 2 + 1];

        // Ajustamos el tiempo acumulado si se llega fuera de la ventana permitida
        if (hora_llegada < ventana_inicio) {
            // Esperamos hasta el inicio de la ventana
            tiempo_acumulado += (ventana_inicio - hora_llegada);
        } else if (ventana_fin < ventana_inicio) {
            // Caso especial: La ventana cruza medianoche (ej. 22:00 a 02:00)
            if (hora_llegada > ventana_fin && hora_llegada < ventana_inicio) {
                tiempo_acumulado += (24.0 - hora_llegada + ventana_inicio); // Esperamos al siguiente día
            }
        } else if (hora_llegada > ventana_fin) {
            // Esperamos al siguiente día si llegamos después del cierre
            tiempo_acumulado += (24.0 - hora_llegada + ventana_inicio);
        }

        // Añadimos al costo total
        total_cost += tiempo_viaje;
    }

    return total_cost;
}

__device__ void heuristica_abruptos_gpu(int *ruta,
                                        int num_ciudades,
                                        int m,
                                        double *distancias,
                                        double *ventanas_de_tiempo,
                                        int *ruta_temp,
                                        DistanciaOrdenadaGPU *dist_ordenadas) {

    for (int i = 0; i < num_ciudades; i++) {
        int ciudad_actual = ruta[i];
        
        // Ordenar ciudades por distancia
        for (int j = 0; j < num_ciudades; j++) {
            dist_ordenadas[j].distancia = distancias[ciudad_actual * num_ciudades + j];
            dist_ordenadas[j].indice = j;
        }
        
        // Ordenamiento simple para GPU
        for (int j = 0; j < m; j++) {
            for (int k = j + 1; k < num_ciudades; k++) {
                if (comparar_distancias_gpu(dist_ordenadas[k], dist_ordenadas[j])) {
                    DistanciaOrdenadaGPU temp = dist_ordenadas[j];
                    dist_ordenadas[j] = dist_ordenadas[k];
                    dist_ordenadas[k] = temp;
                }
            }
        }

        int pos_actual = -1;
        for (int j = 0; j < num_ciudades; j++) {
            if (ruta[j] == ciudad_actual) {
                pos_actual = j;
                break;
            }
        }

        double mejor_costo = evaluar_individuo_gpu(ruta, distancias, ventanas_de_tiempo, num_ciudades);
        int mejor_posicion = pos_actual;
        int mejor_vecino = -1;

        for (int j = 1; j <= m && j < num_ciudades; j++) {
            int ciudad_cercana = dist_ordenadas[j].indice;
            
            int pos_cercana = -1;
            for (int k = 0; k < num_ciudades; k++) {
                if (ruta[k] == ciudad_cercana) {
                    pos_cercana = k;
                    break;
                }
            }

            if (pos_cercana != -1) {
                for (int posicion_antes_o_despues = 0; posicion_antes_o_despues <= 1; posicion_antes_o_despues++) {
                    // Copiar ruta actual
                    for (int k = 0; k < num_ciudades; k++) {
                        ruta_temp[k] = ruta[k];
                    }
                    
                    eliminar_de_posicion_gpu(ruta_temp, num_ciudades, pos_actual);
                    
                    int nueva_pos = pos_cercana + posicion_antes_o_despues;
                    if (nueva_pos > pos_actual) nueva_pos--;
                    if (nueva_pos >= num_ciudades) nueva_pos = num_ciudades - 1;
                    
                    insertar_en_posicion_gpu(ruta_temp, num_ciudades, ciudad_actual, nueva_pos);
                    
                    double nuevo_costo = evaluar_individuo_gpu(ruta_temp, distancias, ventanas_de_tiempo, num_ciudades);
                    
                    if (nuevo_costo < mejor_costo) {
                        mejor_costo = nuevo_costo;
                        mejor_posicion = nueva_pos;
                        mejor_vecino = ciudad_cercana;
                    }
                }
            }
        }

        if (mejor_vecino != -1 && mejor_posicion != pos_actual) {
            for (int k = 0; k < num_ciudades; k++) {
                ruta_temp[k] = ruta[k];
            }
            eliminar_de_posicion_gpu(ruta_temp, num_ciudades, pos_actual);
            insertar_en_posicion_gpu(ruta_temp, num_ciudades, ciudad_actual, mejor_posicion);
            for (int k = 0; k < num_ciudades; k++) {
                ruta[k] = ruta_temp[k];
            }
        }
    }

}

__device__ int comparar_distancias_gpu(DistanciaOrdenadaGPU a, DistanciaOrdenadaGPU b) {
    return (a.distancia < b.distancia);
}

__device__ void insertar_en_posicion_gpu(int* array, int longitud, int elemento, int posicion) {
    for (int i = longitud-1; i > posicion; i--) {
        array[i] = array[i-1];
    }
    array[posicion] = elemento;
}

__device__ void eliminar_de_posicion_gpu(int* array, int longitud, int posicion) {
    int valor = array[posicion];
    for (int i = posicion; i < longitud-1; i++) {
        array[i] = array[i+1];
    }
    array[longitud-1] = valor;
}

// ----------------------------------------------------
// Funciones para copiar poblaciones CPU <-> GPU
// ----------------------------------------------------
void copiarPoblacionCPUaGPU(const poblacion *pobCPU, 
                            individuo_gpu *pobGPU, 
                            int *genotiposGPU,
                            int tamPobl, 
                            int longGen)
{
    // Array temporal en CPU
    individuo_gpu *temp = (individuo_gpu*)malloc(tamPobl * sizeof(individuo_gpu));

    for(int i = 0; i < tamPobl; i++) {
        // Copiar genotipo
        gpuErrchk(cudaMemcpy(genotiposGPU + i*longGen,
                             pobCPU->individuos[i].genotipo,
                             longGen*sizeof(int),
                             cudaMemcpyHostToDevice));
        // Ajustamos puntero
        temp[i].genotipo = genotiposGPU + i*longGen;
        // Fitness
        temp[i].fitness = pobCPU->individuos[i].fitness;
    }

    // Copiar a pobGPU
    gpuErrchk(cudaMemcpy(pobGPU,
                         temp,
                         tamPobl*sizeof(individuo_gpu),
                         cudaMemcpyHostToDevice));

    free(temp);
}

void copiarPoblacionGPUaCPU(poblacion *pobCPU, 
                            const individuo_gpu *pobGPU, 
                            const int *genotiposGPU,
                            int tamPobl, 
                            int longGen)
{
    individuo_gpu *temp = (individuo_gpu*)malloc(tamPobl * sizeof(individuo_gpu));

    gpuErrchk(cudaMemcpy(temp,
                         pobGPU,
                         tamPobl*sizeof(individuo_gpu),
                         cudaMemcpyDeviceToHost));

    for(int i = 0; i < tamPobl; i++) {
        gpuErrchk(cudaMemcpy(pobCPU->individuos[i].genotipo,
                             genotiposGPU + i*longGen,
                             longGen*sizeof(int),
                             cudaMemcpyDeviceToHost));
        pobCPU->individuos[i].fitness = temp[i].fitness;
    }
    free(temp);
}

// ----------------------------------------------------
// Funciones para crear y manejar poblaciones en CPU
// ----------------------------------------------------
poblacion *crear_poblacion(int tamano, int longitud_genotipo) {
    poblacion *Poblacion = (poblacion *)malloc(sizeof(poblacion));
    if(!Poblacion) {
        fprintf(stderr, "Error al asignar memoria para Poblacion\n");
        exit(EXIT_FAILURE);
    }
    Poblacion->tamano = tamano;
    Poblacion->individuos = (individuo *)malloc(tamano * sizeof(individuo));
    if(!Poblacion->individuos) {
        fprintf(stderr, "Error al asignar memoria para individuos\n");
        free(Poblacion);
        exit(EXIT_FAILURE);
    }
    for(int i=0; i<tamano; i++) {
        Poblacion->individuos[i].genotipo = (int*)malloc(longitud_genotipo*sizeof(int));
        if(!Poblacion->individuos[i].genotipo) {
            fprintf(stderr, "Error al asignar memoria para genotipo\n");
            for(int j=0; j<i; j++) {
                free(Poblacion->individuos[j].genotipo);
            }
            free(Poblacion->individuos);
            free(Poblacion);
            exit(EXIT_FAILURE);
        }
    }
    return Poblacion;
}

void crear_permutaciones(poblacion *poblacion, int longitud_genotipo) {
    for(int i=0; i< poblacion->tamano; i++) {
        // inicializa
        for(int j=0; j<longitud_genotipo; j++) {
            poblacion->individuos[i].genotipo[j] = j;
        }
        // fisher-yates
        for(int j = longitud_genotipo-1; j>0; j--) {
            int k = rand()%(j+1);
            int tmp = poblacion->individuos[i].genotipo[j];
            poblacion->individuos[i].genotipo[j] = poblacion->individuos[i].genotipo[k];
            poblacion->individuos[i].genotipo[k] = tmp;
        }
    }
}

void ordenar_poblacion(poblacion *poblacion) {
    int n = poblacion->tamano;
    if(n<=1) return;
    int profundidad_max = 2 * log2_suelo(n);
    introsort_util(poblacion->individuos, &profundidad_max, 0, n);
}

void actualizar_poblacion(poblacion **destino, poblacion *origen, int longitud_genotipo) {
    poblacion *nueva = crear_poblacion(origen->tamano, longitud_genotipo);
    for(int i=0; i<origen->tamano; i++) {
        for(int j=0; j<longitud_genotipo; j++) {
            nueva->individuos[i].genotipo[j] = origen->individuos[i].genotipo[j];
        }
        nueva->individuos[i].fitness = origen->individuos[i].fitness;
    }
    if(*destino!=NULL) {
        for(int i=0; i<(*destino)->tamano; i++) {
            free((*destino)->individuos[i].genotipo);
        }
        free((*destino)->individuos);
        free(*destino);
    }
    *destino = nueva;
}

void liberar_poblacion(poblacion *pob) {
    if(!pob) return;
    if(pob->individuos) {
        for(int i=0; i<pob->tamano; i++) {
            free(pob->individuos[i].genotipo);
        }
        free(pob->individuos);
    }
    free(pob);
}

// ----------------------------------------------------
// Funciones de ordenamiento (introsort, etc.)
// ----------------------------------------------------
int log2_suelo(int n) {
    int log = 0;
    while(n>1) {
        n >>= 1;
        log++;
    }
    return log;
}

void introsort_util(individuo *arr, int *profundidad_max, int inicio, int fin) {
    int tamano = fin - inicio;
    if(tamano<16) {
        insertion_sort(arr, inicio, fin-1);
        return;
    }
    if(*profundidad_max == 0) {
        heapsort(arr+inicio, tamano);
        return;
    }
    (*profundidad_max)--;
    int piv = particion(arr, inicio, fin-1);
    introsort_util(arr, profundidad_max, inicio,   piv);
    introsort_util(arr, profundidad_max, piv+1,    fin);
}

int particion(individuo *arr, int bajo, int alto) {
    int medio = bajo + (alto-bajo)/2;
    int indice_pivote = mediana_de_tres(arr, bajo, medio, alto);
    intercambiar_individuos(&arr[indice_pivote], &arr[alto]);

    individuo pivote = arr[alto];
    int i = bajo-1;
    for(int j=bajo; j<alto; j++) {
        if(arr[j].fitness <= pivote.fitness) {
            i++;
            intercambiar_individuos(&arr[i], &arr[j]);
        }
    }
    intercambiar_individuos(&arr[i+1], &arr[alto]);
    return i+1;
}

int mediana_de_tres(individuo *arr, int a, int b, int c) {
    if(arr[a].fitness <= arr[b].fitness) {
        if(arr[b].fitness <= arr[c].fitness) return b;
        else if(arr[a].fitness <= arr[c].fitness) return c;
        else return a;
    } else {
        if(arr[a].fitness <= arr[c].fitness) return a;
        else if(arr[b].fitness <= arr[c].fitness) return c;
        else return b;
    }
}

void intercambiar_individuos(individuo *a, individuo *b) {
    individuo temp = *a;
    *a = *b;
    *b = temp;
}

void insertion_sort(individuo *arr, int izquierda, int derecha) {
    for(int i=izquierda+1; i<=derecha; i++) {
        individuo clave = arr[i];
        int j = i-1;
        while(j>=izquierda && arr[j].fitness > clave.fitness) {
            arr[j+1] = arr[j];
            j--;
        }
        arr[j+1] = clave;
    }
}

void heapsort(individuo *arr, int n) {
    for(int i = n/2 -1; i>=0; i--)
        heapify(arr, n, i);
    for(int i=n-1; i>0; i--) {
        intercambiar_individuos(&arr[0], &arr[i]);
        heapify(arr, i, 0);
    }
}

void heapify(individuo *arr, int n, int i) {
    int mayor = i;
    int izquierda = 2*i + 1;
    int derecha   = 2*i + 2;
    if(izquierda<n && arr[izquierda].fitness > arr[mayor].fitness) {
        mayor = izquierda;
    }
    if(derecha<n && arr[derecha].fitness > arr[mayor].fitness) {
        mayor = derecha;
    }
    if(mayor!=i) {
        intercambiar_individuos(&arr[i], &arr[mayor]);
        heapify(arr, n, mayor);
    }
}
