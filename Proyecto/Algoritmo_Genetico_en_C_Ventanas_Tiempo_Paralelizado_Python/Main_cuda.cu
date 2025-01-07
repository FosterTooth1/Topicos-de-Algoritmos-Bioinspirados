#include "Biblioteca_cuda.h"
#include <math.h>  // para log2 etc.

// Estructura para devolver resultados a Python
// Esta estructura almacena la información del resultado del algoritmo genético
typedef struct {
    int* recorrido;               // Arreglo con el mejor recorrido encontrado
    double aptitud;               // Aptitud del mejor individuo
    double tiempo_ejecucion;      // Tiempo de ejecución del algoritmo
    char (*nombres_ciudades)[50]; // Arreglo con los nombres de las ciudades correspondientes al mejor recorrido
    int longitud_recorrido;       // Longitud del recorrido
} ResultadoGenetico;

// Definimos el atributo de exportación según el sistema operativo
#ifdef _WIN32
    #define EXPORT __declspec(dllexport)
#else
    #define EXPORT __attribute__((visibility("default")))
#endif

extern "C" {
// Implementación del algoritmo genético
EXPORT ResultadoGenetico* ejecutar_algoritmo_genetico_ventanas_tiempo_paralelizado(
    int tamano_poblacion,         // Tamaño de la población inicial
    int longitud_genotipo,        // Longitud del genotipo (número de ciudades)
    int num_generaciones,         // Número de generaciones a ejecutar
    int num_competidores,         // Número de competidores en el torneo de selección
    int m,                        // Valor para considerar las m ciudades más cercanas en la heurística
    double probabilidad_mutacion, // Probabilidad de mutación
    double probabilidad_cruce,    // Probabilidad de cruce
    const char* nombre_archivo,    // Nombre del archivo con la matriz de distancias
    int km_hr                     // Velocidad en km/hr
) {

    // Iniciamos la medición del tiempo
    time_t inicio = time(NULL);
    // Parámetros del algoritmo genético
    srand(time(NULL));

    // Cargar archivo de distancias
    double **distancias = (double **)malloc(longitud_genotipo * sizeof(double*));
    for(int i=0; i<longitud_genotipo; i++) {
        distancias[i] = (double*)malloc(longitud_genotipo*sizeof(double));
    }
    FILE *archivo = fopen(nombre_archivo, "r");
    if(!archivo) {
        perror("Error al abrir el archivo");
        return NULL;
    }
    char linea[8192];
    int fila=0;
    while(fgets(linea,sizeof(linea), archivo) && fila<longitud_genotipo) {
        char *token = strtok(linea,",");
        int columna=0;
        while(token && columna<longitud_genotipo) {
            // Convertir a double y convertir de km a hr
            distancias[fila][columna] = (atof(token))/km_hr;
            token = strtok(NULL,",");
            columna++;
        }
        fila++;
    }
    fclose(archivo);

    // Nombres de las ciudades (solo para impresión)
    const char nombres_ciudades[32][20] = {
        "Aguascalientes","Baja California","Baja California Sur","Campeche","Chiapas",
        "Chihuahua","Coahuila","Colima","Durango","Guanajuato","Guerrero","Hidalgo",
        "Jalisco","EdoMex","Michoacan","Morelos","Nayarit","NuevoLeon","Oaxaca","Puebla",
        "Queretaro","QuintanaRoo","SanLuisPotosi","Sinaloa","Sonora","Tabasco","Tamaulipas",
        "Tlaxcala","Veracruz","Yucatan","Zacatecas","CDMX"
    };

    // Cereamos las ventanas de tiempo para cada ciudad
    double ventanas_tiempo[32][2] = {
        {8, 12}, {9, 13}, {10, 14}, {11, 15}, {12, 16}, {13, 17}, {14, 18},
        {15, 19}, {16, 20}, {17, 21}, {18, 22}, {19, 23}, {20, 0}, {21, 1},
        {22, 2}, {23, 3}, {0, 4}, {1, 5}, {2, 6}, {3, 7}, {4, 8}, {5, 9},
        {6, 10}, {7, 11}, {8, 12}, {9, 13}, {10, 14}, {11, 15}, {12, 16},
        {13, 17}, {14, 18}, {15, 19}
    };

    // Aplanar ventanas de tiempo
    double *h_ventanas_tiempo_flat = (double*)malloc(longitud_genotipo * 2 * sizeof(double));
    for (int i = 0; i < longitud_genotipo; i++) {
        h_ventanas_tiempo_flat[i * 2] = ventanas_tiempo[i][0];
        h_ventanas_tiempo_flat[i * 2 + 1] = ventanas_tiempo[i][1];
    }

    // Aplanar distancias
    double *h_distancias_flat = (double*)malloc(longitud_genotipo*longitud_genotipo*sizeof(double));
    for(int i=0; i<longitud_genotipo; i++) {
        for(int j=0; j<longitud_genotipo; j++) {
            h_distancias_flat[i*longitud_genotipo + j] = distancias[i][j];
        }
    }
    
    // Reserva en GPU para distancias y RNG states
    double *d_distancias;
    curandState *d_states;
    gpuErrchk(cudaMalloc(&d_distancias, longitud_genotipo*longitud_genotipo*sizeof(double)));
    gpuErrchk(cudaMalloc(&d_states, tamano_poblacion*sizeof(curandState)));

    gpuErrchk(cudaMemcpy(d_distancias, h_distancias_flat,
                         longitud_genotipo*longitud_genotipo*sizeof(double),
                         cudaMemcpyHostToDevice));

    double *d_ventanas_tiempo;
    gpuErrchk(cudaMalloc(&d_ventanas_tiempo, longitud_genotipo * 2 * sizeof(double)));
    gpuErrchk(cudaMemcpy(d_ventanas_tiempo, h_ventanas_tiempo_flat,
                        longitud_genotipo * 2 * sizeof(double),
                        cudaMemcpyHostToDevice));

    // Configuración de CUDA
    int blockSize, minGridSize, gridSize;
    obtenerConfiguracionCUDA(&blockSize, &minGridSize, &gridSize, tamano_poblacion);
    //int blockSize = 64; // o 128
    //int gridSize = ( (tamano_poblacion/2) + blockSize - 1 ) / blockSize;

    setup_curand_kernel<<<gridSize, blockSize>>>(d_states, time(NULL));
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    // Crear poblaciones en CPU
    poblacion *Poblacion = crear_poblacion(tamano_poblacion, longitud_genotipo);
    poblacion *padres    = crear_poblacion(tamano_poblacion, longitud_genotipo);
    poblacion *hijos     = crear_poblacion(tamano_poblacion, longitud_genotipo);

    // Permutaciones iniciales
    crear_permutaciones(Poblacion, longitud_genotipo);

    // ---------------------------------------------------------
    // Reservar estructuras (individuo_gpu) y genotipos en GPU
    // ---------------------------------------------------------
    individuo_gpu *d_poblacion, *d_padres, *d_hijos;
    gpuErrchk(cudaMalloc(&d_poblacion, tamano_poblacion*sizeof(individuo_gpu)));
    gpuErrchk(cudaMalloc(&d_padres,    tamano_poblacion*sizeof(individuo_gpu)));
    gpuErrchk(cudaMalloc(&d_hijos,     tamano_poblacion*sizeof(individuo_gpu)));

    int *d_genotipos_poblacion, *d_genotipos_padres, *d_genotipos_hijos;
    gpuErrchk(cudaMalloc(&d_genotipos_poblacion, tamano_poblacion*longitud_genotipo*sizeof(int)));
    gpuErrchk(cudaMalloc(&d_genotipos_padres,    tamano_poblacion*longitud_genotipo*sizeof(int)));
    gpuErrchk(cudaMalloc(&d_genotipos_hijos,     tamano_poblacion*longitud_genotipo*sizeof(int)));

    // ---------------------------------------------------------
    // (SOLUCIÓN A) Configurar punteros para "padres" en GPU
    // ---------------------------------------------------------
    {
        individuo_gpu *tempPadres = (individuo_gpu*)malloc(tamano_poblacion*sizeof(individuo_gpu));
        for(int i=0; i<tamano_poblacion; i++){
            // cada d_padres[i] apuntará a su trozo en d_genotipos_padres
            tempPadres[i].genotipo = d_genotipos_padres + i*longitud_genotipo;
            tempPadres[i].fitness  = 0.0;  // se actualizará en kernels
        }
        gpuErrchk(cudaMemcpy(d_padres, tempPadres,
                             tamano_poblacion*sizeof(individuo_gpu),
                             cudaMemcpyHostToDevice));
        free(tempPadres);
    }

    // ---------------------------------------------------------
    // (SOLUCIÓN A) Configurar punteros para "hijos" en GPU
    // ---------------------------------------------------------
    {
        individuo_gpu *tempHijos = (individuo_gpu*)malloc(tamano_poblacion*sizeof(individuo_gpu));
        for(int i=0; i<tamano_poblacion; i++){
            // cada d_hijos[i] apuntará a su trozo en d_genotipos_hijos
            tempHijos[i].genotipo = d_genotipos_hijos + i*longitud_genotipo;
            tempHijos[i].fitness  = 0.0;  // se actualizará en kernels
        }
        gpuErrchk(cudaMemcpy(d_hijos, tempHijos,
                             tamano_poblacion*sizeof(individuo_gpu),
                             cudaMemcpyHostToDevice));
        free(tempHijos);
    }
    
    // Ejemplo de blockSize=128, o el que hayas elegido
    int blockSize_1 = 32;
    int gridSize_1  = ( (tamano_poblacion/2) + blockSize_1 - 1 ) / blockSize_1;

    // Cálculo de shared memory:
    size_t espacioCrossover = 3UL * longitud_genotipo * sizeof(int);
    size_t espacioHeurRuta  = (size_t)longitud_genotipo * sizeof(int);
    size_t espacioHeurDist  = (size_t)longitud_genotipo * sizeof(DistanciaOrdenadaGPU);

    size_t totalPorHilo = espacioCrossover + espacioHeurRuta + espacioHeurDist;
    size_t sharedMemSize_1 = blockSize_1 * totalPorHilo;

    // ---------------------------------------------------------
    // Copiar la población inicial (CPU->GPU) para d_poblacion
    // ---------------------------------------------------------
    copiarPoblacionCPUaGPU(Poblacion, d_poblacion, d_genotipos_poblacion, 
                           tamano_poblacion, longitud_genotipo);

    // Evaluar población inicial
    evaluar_poblacion_kernel<<<gridSize, blockSize>>>(d_poblacion, d_distancias, d_ventanas_tiempo,
                                                      tamano_poblacion, longitud_genotipo);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    // Bajar resultados, ordenar
    copiarPoblacionGPUaCPU(Poblacion, d_poblacion, d_genotipos_poblacion,
                           tamano_poblacion, longitud_genotipo);
    ordenar_poblacion(Poblacion);

    // Mejor individuo
    individuo *Mejor_Individuo = (individuo*)malloc(sizeof(individuo));
    Mejor_Individuo->genotipo = (int*)malloc(longitud_genotipo*sizeof(int));
    memcpy(Mejor_Individuo->genotipo, 
           Poblacion->individuos[0].genotipo, 
           longitud_genotipo*sizeof(int));
    Mejor_Individuo->fitness = Poblacion->individuos[0].fitness;

    // Bucle principal
    for(int gen=0; gen<num_generaciones; gen++) {
        // Subir Poblacion (CPU->GPU) para la selección
        copiarPoblacionCPUaGPU(Poblacion, d_poblacion, d_genotipos_poblacion,
                               tamano_poblacion, longitud_genotipo);

        // Selección
        seleccionar_padres_kernel<<<gridSize, blockSize>>>(d_poblacion, d_padres,
                                                           num_competidores,
                                                           tamano_poblacion,
                                                           longitud_genotipo,
                                                           d_states);
        cudaDeviceSynchronize();
        gpuErrchk(cudaGetLastError());

        // Cruzamiento
        cruzar_individuos_kernel<<<gridSize_1, blockSize_1, sharedMemSize_1>>>(
            d_padres, d_hijos,
            d_distancias, d_ventanas_tiempo, probabilidad_cruce,
            tamano_poblacion, longitud_genotipo,
            m, d_states
        );
        cudaDeviceSynchronize();
        gpuErrchk(cudaGetLastError());

        obtenerConfiguracionCUDA(&blockSize, &minGridSize, &gridSize, tamano_poblacion);

        // Mutación
        mutar_individuos_kernel<<<gridSize, blockSize>>>(d_hijos, d_distancias, d_ventanas_tiempo,
                                                         probabilidad_mutacion,
                                                         tamano_poblacion,
                                                         longitud_genotipo,
                                                         d_states);
        cudaDeviceSynchronize();
        gpuErrchk(cudaGetLastError());

        // Bajar hijos a CPU
        copiarPoblacionGPUaCPU(hijos, d_hijos, d_genotipos_hijos,
                               tamano_poblacion, longitud_genotipo);

        // Actualizar Poblacion en CPU
        actualizar_poblacion(&Poblacion, hijos, longitud_genotipo);

        // Evaluar con kernel (nueva Población)
        copiarPoblacionCPUaGPU(Poblacion, d_poblacion, d_genotipos_poblacion,
                               tamano_poblacion, longitud_genotipo);

        evaluar_poblacion_kernel<<<gridSize, blockSize>>>(d_poblacion, d_distancias, d_ventanas_tiempo,
                                                          tamano_poblacion, longitud_genotipo);
        cudaDeviceSynchronize();
        gpuErrchk(cudaGetLastError());

        copiarPoblacionGPUaCPU(Poblacion, d_poblacion, d_genotipos_poblacion,
                               tamano_poblacion, longitud_genotipo);
        ordenar_poblacion(Poblacion);

        // Mejor individuo
        if(Poblacion->individuos[0].fitness < Mejor_Individuo->fitness) {
            memcpy(Mejor_Individuo->genotipo,
                   Poblacion->individuos[0].genotipo,
                   longitud_genotipo*sizeof(int));
            Mejor_Individuo->fitness = Poblacion->individuos[0].fitness;
        }
    }

    // Tiempo final
    time_t fin = time(NULL);
    double tiempo = difftime(fin, inicio);

    // Preparamos el resultado para devolverlo a Python
    ResultadoGenetico* resultado = (ResultadoGenetico*)malloc(sizeof(ResultadoGenetico));
    resultado->recorrido = (int*)malloc(longitud_genotipo * sizeof(int));
    resultado->nombres_ciudades = (char(*)[50])malloc(longitud_genotipo * sizeof(char[50]));
    resultado->longitud_recorrido = longitud_genotipo;

    // Rellenamos la estructura con el mejor recorrido y su información
    for (int i = 0; i < longitud_genotipo; i++) {
        resultado->recorrido[i] = Mejor_Individuo->genotipo[i];
        strncpy(resultado->nombres_ciudades[i], nombres_ciudades[Mejor_Individuo->genotipo[i]], 49);
        resultado->nombres_ciudades[i][49] = '\0';
    }
    resultado->aptitud = Mejor_Individuo->fitness;
    resultado->tiempo_ejecucion = tiempo;

    // Liberar memoria CPU
    liberar_poblacion(Poblacion);
    liberar_poblacion(padres);
    liberar_poblacion(hijos);
    for(int i=0; i<longitud_genotipo; i++) {
        free(distancias[i]);
    }
    free(distancias);
    free(h_distancias_flat);
    free(h_ventanas_tiempo_flat);

    free(Mejor_Individuo->genotipo);
    free(Mejor_Individuo);

    // Liberar memoria GPU
    gpuErrchk(cudaFree(d_distancias));
    gpuErrchk(cudaFree(d_states));
    gpuErrchk(cudaFree(d_ventanas_tiempo));
    gpuErrchk(cudaFree(d_poblacion));
    gpuErrchk(cudaFree(d_padres));
    gpuErrchk(cudaFree(d_hijos));
    gpuErrchk(cudaFree(d_genotipos_poblacion));
    gpuErrchk(cudaFree(d_genotipos_padres));
    gpuErrchk(cudaFree(d_genotipos_hijos));

    return resultado;
}
}

extern "C" {
// Función para liberar la memoria del resultado en Python
EXPORT void liberar_resultado(ResultadoGenetico* resultado) {
    if (resultado) {
        free(resultado->recorrido);
        free(resultado->nombres_ciudades);
        free(resultado);
    }
}
}