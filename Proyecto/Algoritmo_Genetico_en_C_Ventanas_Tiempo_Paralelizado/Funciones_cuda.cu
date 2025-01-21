#include "Biblioteca_cuda.h"

// ----------------------------------------------------
// Implementación de la macro para manejo de errores
// ----------------------------------------------------

// Función para verificar y manejar errores de CUDA
// Recibe el código de error, el nombre del archivo y la línea donde ocurrió el error
// No devuelve nada, pero puede abortar la ejecución si se detecta un error
void gpuAssert(cudaError_t code, const char *file, int line, bool abort) {
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// ----------------------------------------------------
// Kernels y funciones CUDA
// ----------------------------------------------------

// Kernel para inicializar los estados de curand en la GPU
// Recibe un puntero a los estados de curand y una semilla para la generación de números aleatorios
// No devuelve nada, pero inicializa el estado de curand para cada hilo
__global__ void setup_curand_kernel(curandState *states, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, idx, 0, &states[idx]);
}

// Función para obtener la configuración óptima de bloques y grids en CUDA
// Recibe punteros a blockSize, minGridSize, gridSize y el tamaño N del problema
// No devuelve nada, pero actualiza los valores de blockSize, minGridSize y gridSize
void obtenerConfiguracionCUDA(int *blockSize, int *minGridSize, int *gridSize, int N) {
    cudaOccupancyMaxPotentialBlockSize(minGridSize, blockSize, evaluar_poblacion_kernel, 0, N);
    *gridSize = (N + *blockSize - 1) / *blockSize;
}

// Kernel para evaluar la población en la GPU
// Recibe un puntero a la población en la GPU, matrices de distancias y ventanas de tiempo, y los tamaños de la población y el genotipo
// No devuelve nada, pero actualiza el fitness de cada individuo en la población
__global__ void evaluar_poblacion_kernel(individuo_gpu *poblacion, double *distancias, double *ventanas_tiempo,
                                         int tamano_poblacion, int longitud_genotipo) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= tamano_poblacion) return;

    double total_cost = 0.0;         // Costo total del recorrido (en horas)
    double tiempo_acumulado = 0.0;   // Tiempo transcurrido desde el inicio del recorrido

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

// Kernel para seleccionar padres mediante torneo en la GPU
// Recibe punteros a la población y a los padres en la GPU, el número de competidores, el tamaño de la población, la longitud del genotipo y los estados de curand
// No devuelve nada, pero selecciona los mejores individuos para ser padres
__global__ void seleccionar_padres_kernel(individuo_gpu *poblacion, individuo_gpu *padres,
                                          int num_competidores, int tamano_poblacion,
                                          int longitud_genotipo, curandState *states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= tamano_poblacion) return;

    int mejor_idx = -1;
    double mejor_fitness = 1e9;

    // Realiza un torneo entre num_competidores individuos aleatorios
    for (int i = 0; i < num_competidores; i++) {
        int rand_idx = curand(&states[idx]) % tamano_poblacion;
        if (poblacion[rand_idx].fitness < mejor_fitness) {
            mejor_fitness = poblacion[rand_idx].fitness;
            mejor_idx = rand_idx;
        }
    }

    // Copia el mejor individuo seleccionado al arreglo de padres
    for (int j = 0; j < longitud_genotipo; j++) {
        padres[idx].genotipo[j] = poblacion[mejor_idx].genotipo[j];
    }
    padres[idx].fitness = poblacion[mejor_idx].fitness;
}

// Kernel para cruzar individuos y generar hijos en la GPU
// Recibe punteros a los padres y a los hijos en la GPU, matrices de distancias y ventanas de tiempo, la probabilidad de cruce, el tamaño de la población, la longitud del genotipo, el parámetro m y los estados de curand
// No devuelve nada, pero genera nuevos hijos a partir de los padres seleccionados
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

    size_t espacioCrossover = 3UL * longitud_genotipo * sizeof(int);  // hijo1, hijo2, visitado
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

        // Llamar heurística:
        heuristica_abruptos_gpu(hijo1, longitud_genotipo, m, distancias, ventanas_de_tiempo, ruta_temp, dist_ordenadas);
        heuristica_abruptos_gpu(hijo2, longitud_genotipo, m, distancias, ventanas_de_tiempo, ruta_temp, dist_ordenadas);

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

// Kernel para mutar individuos en la GPU
// Recibe punteros a los individuos en la GPU, matrices de distancias y ventanas de tiempo, la probabilidad de mutación, el tamaño de la población, la longitud del genotipo y los estados de curand
// No devuelve nada, pero aplica mutaciones a los individuos seleccionados y actualiza su fitness
__global__ void mutar_individuos_kernel(individuo_gpu *individuos, double *distancias, double *ventanas_de_tiempo,
                                        double prob_mutacion, int tamano_poblacion,
                                        int longitud_genotipo, curandState *states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= tamano_poblacion) return;

    // Determina si el individuo debe mutar basado en la probabilidad de mutación
    if (curand_uniform(&states[idx]) < prob_mutacion) {
        // Selecciona dos índices aleatorios para intercambiar en el genotipo
        int idx1 = (int)(curand_uniform(&states[idx]) * longitud_genotipo);
        int idx2 = (int)(curand_uniform(&states[idx]) * longitud_genotipo);
        while (idx2 == idx1) {
            idx2 = (int)(curand_uniform(&states[idx]) * longitud_genotipo);
        }

        // Intercambia los genes en las posiciones seleccionadas
        int temp = individuos[idx].genotipo[idx1];
        individuos[idx].genotipo[idx1] = individuos[idx].genotipo[idx2];
        individuos[idx].genotipo[idx2] = temp;

        // Recalcula el fitness local del individuo tras la mutación
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

// Actualiza la población destino con la población origen
// Recibe:
//   - individuo_gpu *destino: Doble puntero a la población destino en la GPU
//   - individuo_gpu *origen: Puntero a la población origen en la GPU
//   - int longitud_genotipo: Longitud del genotipo de cada individuo
// No devuelve nada, pero reemplaza la población destino con la población origen
__global__ void actualizar_poblacion_kernel(individuo_gpu *destino,
                                            individuo_gpu *origen,
                                            int tamano_poblacion,
                                            int longitud_genotipo)
{
    // Identificador global del hilo
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Copiamos solo si idx está dentro del rango
    if (idx < tamano_poblacion) {
        // Copiamos todos los genes
        for(int j = 0; j < longitud_genotipo; j++) {
            destino[idx].genotipo[j] = origen[idx].genotipo[j];
        }
        // Copiamos el fitness
        destino[idx].fitness = origen[idx].fitness;
    }
}

// Función auxiliar para obtener el mejor individuo (índice + fitness) en GPU
// Recibe punteros a los individuos en la poblacion, el tamaño de la poblacion, el tamaño del bloque de memoria
// y un puntero con los indices y el genotipo del mejor individuo en GPU
// No devuelve nada, todo se hace por referencia
void buscarMejorIndividuoEnGPU(individuo_gpu *d_poblacion,
                               int tamano_poblacion,
                               int blockSize,
                               MinData *d_result)
{
    // 1) Fase 1: reduce en muchos bloques
    int gridSize = (tamano_poblacion + blockSize - 1)/blockSize;
    size_t sharedMemSize = blockSize * sizeof(MinData);

    // Reservamos array d_parciales (gridSize entradas)
    MinData *d_parciales = nullptr;
    cudaMalloc(&d_parciales, gridSize*sizeof(MinData));

    reduce_find_min_phase1<<<gridSize, blockSize, sharedMemSize>>>(
        d_poblacion,
        tamano_poblacion,
        d_parciales
    );
    cudaDeviceSynchronize();

    // 2) Fase 2: reduce de los parciales (gridSize) a un solo resultado
    //            Podemos lanzar 1 bloque con blockSize >= gridSize
    int blockSize2 = 256;
    size_t sharedMemSize2 = blockSize2*sizeof(MinData);

    reduce_find_min_phase2<<<1, blockSize2, sharedMemSize2>>>(
        d_parciales,
        gridSize,
        d_result
    );
    cudaDeviceSynchronize();

    cudaFree(d_parciales);
}

// Implementación de los kernels de reduce (búsqueda del mínimo)
__global__ void reduce_find_min_phase1(
    const individuo_gpu *d_poblacion,
    int tamano_poblacion,
    MinData *d_parciales // una entrada por bloque
)
{
    extern __shared__ MinData sdata[]; // Memoria compartida dinámica

    int tidGlobal = blockIdx.x * blockDim.x + threadIdx.x;
    int tidLocal  = threadIdx.x;

    // Inicializamos con "valores grandes"
    sdata[tidLocal].fitness = 1e30;
    sdata[tidLocal].idx     = -1;

    // Cargamos desde memoria global
    if(tidGlobal < tamano_poblacion){
        sdata[tidLocal].fitness = d_poblacion[tidGlobal].fitness;
        sdata[tidLocal].idx     = tidGlobal;
    }
    __syncthreads();

    // Reducción (min) en la memoria compartida
    for(int stride = blockDim.x/2; stride>0; stride >>= 1){
        if(tidLocal < stride){
            if(sdata[tidLocal + stride].fitness < sdata[tidLocal].fitness){
                sdata[tidLocal].fitness = sdata[tidLocal + stride].fitness;
                sdata[tidLocal].idx     = sdata[tidLocal + stride].idx;
            }
        }
        __syncthreads();
    }

    // El hilo 0 del bloque escribe su resultado parcial
    if(tidLocal == 0){
        d_parciales[blockIdx.x] = sdata[0];
    }
}

__global__ void reduce_find_min_phase2(
    const MinData *d_parciales_in, // tamaño = gridSize de la fase1
    int n,
    MinData *d_result // 1 entrada final
)
{
    extern __shared__ MinData sdata[];
    int tidLocal  = threadIdx.x;
    int tidGlobal = blockIdx.x * blockDim.x + threadIdx.x;

    // Inicializamos en valores grandes
    sdata[tidLocal].fitness = 1e30;
    sdata[tidLocal].idx     = -1;

    // Cargamos
    if(tidGlobal < n){
        sdata[tidLocal] = d_parciales_in[tidGlobal];
    }
    __syncthreads();

    // Reducción en shared
    for(int stride = blockDim.x/2; stride>0; stride >>= 1){
        if(tidLocal < stride){
            if(sdata[tidLocal + stride].fitness < sdata[tidLocal].fitness){
                sdata[tidLocal].fitness = sdata[tidLocal + stride].fitness;
                sdata[tidLocal].idx     = sdata[tidLocal + stride].idx;
            }
        }
        __syncthreads();
    }

    // Hilo 0 escribe el resultado
    if(tidLocal == 0){
        d_result[0] = sdata[0];
    }
}

// ----------------------------------------------------
// Funciones device auxiliares
// ----------------------------------------------------

// Función device para realizar el cruce de ciclos entre dos padres
// Recibe:
//   - const int *p1: Genotipo del primer padre
//   - const int *p2: Genotipo del segundo padre
//   - int *child: Genotipo del hijo a generar
//   - int *visitado: Arreglo para rastrear ciudades visitadas
//   - int num_ciudades: Número de ciudades en el genotipo
// No devuelve nada, pero modifica el genotipo del hijo
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
        // Encontrar primera posición no visitada
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

// Función device para evaluar el fitness de un individuo en la GPU
// Recibe:
//   - int *ruta: Genotipo del individuo (ruta de ciudades)
//   - double *distancias: Matriz de distancias entre ciudades
//   - double *ventanas_de_tiempo: Matriz de ventanas de tiempo para cada ciudad
//   - int num_ciudades: Número de ciudades en el genotipo
// Devuelve:
//   - double: Costo total del recorrido (fitness del individuo)
__device__ double evaluar_individuo_gpu(int *ruta, double *distancias, double *ventanas_de_tiempo, int num_ciudades) {
    double total_cost = 0.0;          // Costo total del recorrido (en horas)
    double tiempo_acumulado = 0.0;    // Tiempo transcurrido desde el inicio del recorrido

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

// Función device para aplicar la heurística de eliminación de abruptos en la GPU
// Recibe:
//   - int *ruta: Ruta actual del individuo
//   - int num_ciudades: Número de ciudades en la ruta
//   - int m: Número de vecinos a considerar
//   - double *distancias: Matriz de distancias entre ciudades
//   - double *ventanas_de_tiempo: Matriz de ventanas de tiempo para cada ciudad
//   - int *ruta_temp: Ruta temporal para modificaciones
//   - DistanciaOrdenadaGPU *dist_ordenadas: Arreglo auxiliar para ordenar distancias
// No devuelve nada, pero modifica la ruta para eliminar abruptos
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

// Función device para comparar dos distancias en la GPU
// Recibe:
//   - DistanciaOrdenadaGPU a: Primera distancia a comparar
//   - DistanciaOrdenadaGPU b: Segunda distancia a comparar
// Devuelve:
//   - int: 1 si a < b, 0 en caso contrario
__device__ int comparar_distancias_gpu(DistanciaOrdenadaGPU a, DistanciaOrdenadaGPU b) {
    return (a.distancia < b.distancia);
}

// Función device para insertar un elemento en una posición específica de un arreglo en la GPU
// Recibe:
//   - int* array: Arreglo donde se insertará el elemento
//   - int longitud: Longitud del arreglo
//   - int elemento: Elemento a insertar
//   - int posicion: Posición donde se insertará el elemento
// No devuelve nada, pero modifica el arreglo
__device__ void insertar_en_posicion_gpu(int* array, int longitud, int elemento, int posicion) {
    for (int i = longitud-1; i > posicion; i--) {
        array[i] = array[i-1];
    }
    array[posicion] = elemento;
}

// Función device para eliminar un elemento de una posición específica de un arreglo en la GPU
// Recibe:
//   - int* array: Arreglo del que se eliminará el elemento
//   - int longitud: Longitud del arreglo
//   - int posicion: Posición del elemento a eliminar
// No devuelve nada, pero modifica el arreglo
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

// Copia una población desde la CPU a la GPU
// Recibe:
//   - const poblacion *pobCPU: Puntero a la población en la CPU
//   - individuo_gpu *pobGPU: Puntero a la población en la GPU
//   - int *genotiposGPU: Puntero a los genotipos en la GPU
//   - int tamPobl: Tamaño de la población
//   - int longGen: Longitud del genotipo
// No devuelve nada, pero transfiere los datos de la CPU a la GPU
void copiarPoblacionCPUaGPU(const poblacion *pobCPU, 
                            individuo_gpu *pobGPU, 
                            int *genotiposGPU,
                            int tamPobl, 
                            int longGen)
{
    // Array temporal en CPU para almacenar los individuos GPU
    individuo_gpu *temp = (individuo_gpu*)malloc(tamPobl * sizeof(individuo_gpu));

    for(int i = 0; i < tamPobl; i++) {
        // Copiar genotipo desde la CPU a la GPU
        gpuErrchk(cudaMemcpy(genotiposGPU + i*longGen,
                             pobCPU->individuos[i].genotipo,
                             longGen*sizeof(int),
                             cudaMemcpyHostToDevice));
        // Ajustar el puntero del genotipo en la GPU
        temp[i].genotipo = genotiposGPU + i*longGen;
        // Copiar el fitness del individuo
        temp[i].fitness = pobCPU->individuos[i].fitness;
    }

    // Copiar toda la población GPU desde el array temporal
    gpuErrchk(cudaMemcpy(pobGPU,
                         temp,
                         tamPobl*sizeof(individuo_gpu),
                         cudaMemcpyHostToDevice));

    // Liberar la memoria temporal
    free(temp);
}

// Copia una población desde la GPU a la CPU
// Recibe:
//   - poblacion *pobCPU: Puntero a la población en la CPU
//   - const individuo_gpu *pobGPU: Puntero a la población en la GPU
//   - const int *genotiposGPU: Puntero a los genotipos en la GPU
//   - int tamPobl: Tamaño de la población
//   - int longGen: Longitud del genotipo
// No devuelve nada, pero transfiere los datos de la GPU a la CPU
void copiarPoblacionGPUaCPU(poblacion *pobCPU, 
                            const individuo_gpu *pobGPU, 
                            const int *genotiposGPU,
                            int tamPobl, 
                            int longGen)
{
    // Array temporal en CPU para almacenar los individuos GPU
    individuo_gpu *temp = (individuo_gpu*)malloc(tamPobl * sizeof(individuo_gpu));

    // Copiar la población GPU al array temporal en la CPU
    gpuErrchk(cudaMemcpy(temp,
                         pobGPU,
                         tamPobl*sizeof(individuo_gpu),
                         cudaMemcpyDeviceToHost));

    for(int i = 0; i < tamPobl; i++) {
        // Copiar el genotipo desde la GPU a la CPU
        gpuErrchk(cudaMemcpy(pobCPU->individuos[i].genotipo,
                             genotiposGPU + i*longGen,
                             longGen*sizeof(int),
                             cudaMemcpyDeviceToHost));
        // Copiar el fitness del individuo
        pobCPU->individuos[i].fitness = temp[i].fitness;
    }
    // Liberar la memoria temporal
    free(temp);
}

// ----------------------------------------------------
// Funciones para crear y manejar poblaciones en CPU
// ----------------------------------------------------

// Crea una población en la CPU
// Recibe:
//   - int tamano: Tamaño de la población
//   - int longitud_genotipo: Longitud del genotipo de cada individuo
// Devuelve:
//   - poblacion*: Puntero a la población creada
poblacion *crear_poblacion(int tamano, int longitud_genotipo) {
    // Asigna memoria para la estructura de la población
    poblacion *Poblacion = (poblacion *)malloc(sizeof(poblacion));
    if(!Poblacion) {
        fprintf(stderr, "Error al asignar memoria para Poblacion\n");
        exit(EXIT_FAILURE);
    }
    Poblacion->tamano = tamano;

    // Asigna memoria para los individuos de la población
    Poblacion->individuos = (individuo *)malloc(tamano * sizeof(individuo));
    if(!Poblacion->individuos) {
        fprintf(stderr, "Error al asignar memoria para individuos\n");
        free(Poblacion);
        exit(EXIT_FAILURE);
    }

    // Asigna memoria para los genotipos de cada individuo
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

// Crea permutaciones aleatorias para cada individuo de la población
// Recibe:
//   - poblacion *poblacion: Puntero a la población en la CPU
//   - int longitud_genotipo: Longitud del genotipo de cada individuo
// No devuelve nada, pero inicializa los genotipos con permutaciones aleatorias
void crear_permutaciones(poblacion *poblacion, int longitud_genotipo) {
    for(int i=0; i< poblacion->tamano; i++) {
        // Inicializa el genotipo con valores ordenados
        for(int j=0; j<longitud_genotipo; j++) {
            poblacion->individuos[i].genotipo[j] = j;
        }
        // Mezcla el genotipo utilizando el algoritmo de Fisher-Yates
        for(int j = longitud_genotipo-1; j>0; j--) {
            int k = rand()%(j+1);
            int tmp = poblacion->individuos[i].genotipo[j];
            poblacion->individuos[i].genotipo[j] = poblacion->individuos[i].genotipo[k];
            poblacion->individuos[i].genotipo[k] = tmp;
        }
    }
}

// Libera la memoria asignada a una población en la CPU
// Recibe:
//   - poblacion *pob: Puntero a la población en la CPU
// No devuelve nada, pero libera toda la memoria asociada a la población
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