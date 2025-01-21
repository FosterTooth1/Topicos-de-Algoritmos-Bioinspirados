#include "Biblioteca_cuda.h"

int main(int argc, char** argv) {
    // Iniciamos la medición del tiempo
    time_t inicio = time(NULL);

    // Definimos los parámetros del algoritmo genético
    srand(time(NULL));
    int tamano_poblacion = 100000;
    int longitud_genotipo = 32;
    int num_generaciones  = 100;
    int num_competidores  = 2;
    int m = 3;
    double prob_mutacion = 0.3;
    double prob_cruce    = 0.9;
    int km_hr = 80;

    // Cargamos el archivo de distancias
    const char *nombre_archivo = "Distancias_no_head.csv";
    double **distancias = (double **)malloc(longitud_genotipo * sizeof(double*));
    for(int i = 0; i < longitud_genotipo; i++) {
        distancias[i] = (double*)malloc(longitud_genotipo * sizeof(double));
    }
    FILE *archivo = fopen(nombre_archivo, "r");
    if(!archivo) {
        perror("Error al abrir el archivo");
        return 1;
    }
    char linea[8192];
    int fila = 0;
    while(fgets(linea, sizeof(linea), archivo) && fila < longitud_genotipo) {
        char *token = strtok(linea, ",");
        int columna = 0;
        while(token && columna < longitud_genotipo) {
            // Convertimos a double y convertimos de km a hr
            distancias[fila][columna] = (atof(token)) / km_hr;
            token = strtok(NULL, ",");
            columna++;
        }
        fila++;
    }
    fclose(archivo);

    // Definimos los nombres de las ciudades (solo para impresión)
    const char nombres_ciudades[32][20] = {
        "Aguascalientes","Baja California","Baja California Sur","Campeche","Chiapas",
        "Chihuahua","Coahuila","Colima","Durango","Guanajuato","Guerrero","Hidalgo",
        "Jalisco","EdoMex","Michoacan","Morelos","Nayarit","NuevoLeon","Oaxaca","Puebla",
        "Queretaro","QuintanaRoo","SanLuisPotosi","Sinaloa","Sonora","Tabasco","Tamaulipas",
        "Tlaxcala","Veracruz","Yucatan","Zacatecas","CDMX"
    };

    // Creamos las ventanas de tiempo para cada ciudad
    double ventanas_tiempo[32][2] = {
        {8, 12}, {9, 13}, {10, 14}, {11, 15}, {12, 16}, {13, 17}, {14, 18},
        {15, 19}, {16, 20}, {17, 21}, {18, 22}, {19, 23}, {20, 0}, {21, 1},
        {22, 2}, {23, 3}, {0, 4}, {1, 5}, {2, 6}, {3, 7}, {4, 8}, {5, 9},
        {6, 10}, {7, 11}, {8, 12}, {9, 13}, {10, 14}, {11, 15}, {12, 16},
        {13, 17}, {14, 18}, {15, 19}
    };

    // Aplanamos las ventanas de tiempo
    double *h_ventanas_tiempo_flat = (double*)malloc(longitud_genotipo * 2 * sizeof(double));
    for (int i = 0; i < longitud_genotipo; i++) {
        h_ventanas_tiempo_flat[i * 2]     = ventanas_tiempo[i][0];
        h_ventanas_tiempo_flat[i * 2 + 1] = ventanas_tiempo[i][1];
    }

    // Aplanamos las distancias
    double *h_distancias_flat = (double*)malloc(longitud_genotipo * longitud_genotipo * sizeof(double));
    for(int i = 0; i < longitud_genotipo; i++) {
        for(int j = 0; j < longitud_genotipo; j++) {
            h_distancias_flat[i * longitud_genotipo + j] = distancias[i][j];
        }
    }
    
    // Reservamos en GPU para distancias y estados RNG
    double *d_distancias;
    curandState *d_states;
    gpuErrchk(cudaMalloc(&d_distancias, longitud_genotipo * longitud_genotipo * sizeof(double)));
    gpuErrchk(cudaMalloc(&d_states, tamano_poblacion * sizeof(curandState)));

    gpuErrchk(cudaMemcpy(d_distancias, h_distancias_flat,
                         longitud_genotipo * longitud_genotipo * sizeof(double),
                         cudaMemcpyHostToDevice));

    double *d_ventanas_tiempo;
    gpuErrchk(cudaMalloc(&d_ventanas_tiempo, longitud_genotipo * 2 * sizeof(double)));
    gpuErrchk(cudaMemcpy(d_ventanas_tiempo, h_ventanas_tiempo_flat,
                        longitud_genotipo * 2 * sizeof(double),
                        cudaMemcpyHostToDevice));

    // Configuramos CUDA
    int blockSize, minGridSize, gridSize;
    obtenerConfiguracionCUDA(&blockSize, &minGridSize, &gridSize, tamano_poblacion);
    //int blockSize = 64; 
    //int gridSize = ( (tamano_poblacion/2) + blockSize - 1 ) / blockSize;

    setup_curand_kernel<<<gridSize, blockSize>>>(d_states, time(NULL));
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    // Creamos poblaciones en CPU
    poblacion *Poblacion = crear_poblacion(tamano_poblacion, longitud_genotipo);
    poblacion *padres    = crear_poblacion(tamano_poblacion, longitud_genotipo);
    poblacion *hijos     = crear_poblacion(tamano_poblacion, longitud_genotipo);

    // Realizamos permutaciones iniciales
    crear_permutaciones(Poblacion, longitud_genotipo);

    // ---------------------------------------------------------
    // Reservamos estructuras (individuo_gpu) y genotipos en GPU
    // ---------------------------------------------------------
    individuo_gpu *d_poblacion, *d_padres, *d_hijos;
    gpuErrchk(cudaMalloc(&d_poblacion, tamano_poblacion * sizeof(individuo_gpu)));
    gpuErrchk(cudaMalloc(&d_padres,    tamano_poblacion * sizeof(individuo_gpu)));
    gpuErrchk(cudaMalloc(&d_hijos,     tamano_poblacion * sizeof(individuo_gpu)));

    int *d_genotipos_poblacion, *d_genotipos_padres, *d_genotipos_hijos;
    gpuErrchk(cudaMalloc(&d_genotipos_poblacion, tamano_poblacion * longitud_genotipo * sizeof(int)));
    gpuErrchk(cudaMalloc(&d_genotipos_padres,    tamano_poblacion * longitud_genotipo * sizeof(int)));
    gpuErrchk(cudaMalloc(&d_genotipos_hijos,     tamano_poblacion * longitud_genotipo * sizeof(int)));


    // ---------------------------------------------------------
    // Configuramos punteros para "padres" en GPU
    // ---------------------------------------------------------
    {
        individuo_gpu *tempPadres = (individuo_gpu*)malloc(tamano_poblacion * sizeof(individuo_gpu));
        for(int i = 0; i < tamano_poblacion; i++){
            // cada d_padres[i] apuntará a su trozo en d_genotipos_padres
            tempPadres[i].genotipo = d_genotipos_padres + i * longitud_genotipo;
            tempPadres[i].fitness  = 0.0;  // se actualizará en kernels
        }
        gpuErrchk(cudaMemcpy(d_padres, tempPadres,
                             tamano_poblacion * sizeof(individuo_gpu),
                             cudaMemcpyHostToDevice));
        free(tempPadres);
    }

    // ---------------------------------------------------------
    // Configuramos punteros para "hijos" en GPU
    // ---------------------------------------------------------
    {
        individuo_gpu *tempHijos = (individuo_gpu*)malloc(tamano_poblacion * sizeof(individuo_gpu));
        for(int i = 0; i < tamano_poblacion; i++){
            // cada d_hijos[i] apuntará a su trozo en d_genotipos_hijos
            tempHijos[i].genotipo = d_genotipos_hijos + i * longitud_genotipo;
            tempHijos[i].fitness  = 0.0;  // se actualizará en kernels
        }
        gpuErrchk(cudaMemcpy(d_hijos, tempHijos,
                             tamano_poblacion * sizeof(individuo_gpu),
                             cudaMemcpyHostToDevice));
        free(tempHijos);
    }
    
    // Definimos blockSize_1 y gridSize_1
    // Estos tamaños los usamos para el kernel de Cruzamiento
    int blockSize_1 = 32;
    int gridSize_1  = ( (tamano_poblacion / 2) + blockSize_1 - 1 ) / blockSize_1;

    // Calculamos la memoria compartida:
    size_t espacioCrossover = 3UL * longitud_genotipo * sizeof(int);
    size_t espacioHeurRuta  = (size_t)longitud_genotipo * sizeof(int);
    size_t espacioHeurDist  = (size_t)longitud_genotipo * sizeof(DistanciaOrdenadaGPU);

    size_t totalPorHilo = espacioCrossover + espacioHeurRuta + espacioHeurDist;
    size_t sharedMemSize_1 = blockSize_1 * totalPorHilo;

    // ---------------------------------------------------------
    // Copiamos la población inicial (CPU->GPU) para d_poblacion
    // ---------------------------------------------------------
    copiarPoblacionCPUaGPU(Poblacion, d_poblacion, d_genotipos_poblacion, 
                           tamano_poblacion, longitud_genotipo);

    // Evaluamos la población inicial
    evaluar_poblacion_kernel<<<gridSize, blockSize>>>(d_poblacion, d_distancias, d_ventanas_tiempo,
                                                      tamano_poblacion, longitud_genotipo);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    // Hallamos el mejor individuo con reduce en GPU
    MinData *d_result;
    cudaMalloc(&d_result, sizeof(MinData));
    buscarMejorIndividuoEnGPU(d_poblacion, tamano_poblacion, blockSize, d_result);

    // Copiamos el mejor (fitness, idx)
    MinData host_min_data;
    cudaMemcpy(&host_min_data, d_result, sizeof(MinData), cudaMemcpyDeviceToHost);

    int *tempGenotipo = (int*)malloc(longitud_genotipo * sizeof(int));
    if (!tempGenotipo) {
        fprintf(stderr, "Error al asignar memoria para tempGenotipo.\n");
        exit(1);
    }

    // Copiar el genotipo del mejor individuo
    individuo *Mejor_Individuo = (individuo*)malloc(sizeof(individuo));
    Mejor_Individuo->genotipo = (int*)malloc(longitud_genotipo*sizeof(int));
    Mejor_Individuo->fitness  = host_min_data.fitness;

    int bestIdx = host_min_data.idx;
    cudaMemcpy(Mejor_Individuo->genotipo,
               d_genotipos_poblacion + bestIdx*longitud_genotipo,
               longitud_genotipo*sizeof(int),
               cudaMemcpyDeviceToHost);

    // Ejecutamos el bucle principal
    for(int gen = 0; gen < num_generaciones; gen++) {

        // Seleccionamos los padres
        seleccionar_padres_kernel<<<gridSize, blockSize>>>(d_poblacion, d_padres,
                                                           num_competidores,
                                                           tamano_poblacion,
                                                           longitud_genotipo,
                                                           d_states);
        cudaDeviceSynchronize();
        gpuErrchk(cudaGetLastError());

        // Realizamos el cruzamiento
        cruzar_individuos_kernel<<<gridSize_1, blockSize_1, sharedMemSize_1>>>(
            d_padres, d_hijos,
            d_distancias, d_ventanas_tiempo, prob_cruce,
            tamano_poblacion, longitud_genotipo,
            m, d_states
        );
        cudaDeviceSynchronize();
        gpuErrchk(cudaGetLastError());

        // Aplicamos la mutación
        mutar_individuos_kernel<<<gridSize, blockSize>>>(d_hijos, d_distancias, d_ventanas_tiempo,
                                                         prob_mutacion,
                                                         tamano_poblacion,
                                                         longitud_genotipo,
                                                         d_states);
        cudaDeviceSynchronize();
        gpuErrchk(cudaGetLastError());

        // Reconfiguramos CUDA si es necesario
        obtenerConfiguracionCUDA(&blockSize, &minGridSize, &gridSize, tamano_poblacion);

        // Actualizamos poblacion de padres a hijos
        actualizar_poblacion_kernel<<<gridSize, blockSize>>>(d_poblacion, d_hijos,
                                                            tamano_poblacion, longitud_genotipo);
        cudaDeviceSynchronize();
        gpuErrchk(cudaGetLastError());

        // Evaluamos la poblacion
        evaluar_poblacion_kernel<<<gridSize, blockSize>>>(d_poblacion, d_distancias, d_ventanas_tiempo,
                                                          tamano_poblacion, longitud_genotipo);
        cudaDeviceSynchronize();
        gpuErrchk(cudaGetLastError());

        // Hallamos el mejor individuo en GPU (reduce)
        buscarMejorIndividuoEnGPU(d_poblacion, tamano_poblacion, blockSize, d_result);
        cudaMemcpy(&host_min_data, d_result, sizeof(MinData), cudaMemcpyDeviceToHost);

        // Copiamos el genotipo del mejor individuo (host_min_data.idx)
        cudaMemcpy(
            tempGenotipo,
            d_genotipos_poblacion + host_min_data.idx * longitud_genotipo,
            longitud_genotipo * sizeof(int),
            cudaMemcpyDeviceToHost
        );

        // Actualizamos el Mejor_Individuo global si es necesario (para guardar el mejor histórico):
        if (host_min_data.fitness < Mejor_Individuo->fitness) {
            Mejor_Individuo->fitness = host_min_data.fitness;
            memcpy(Mejor_Individuo->genotipo, tempGenotipo, 
                longitud_genotipo * sizeof(int));
        }
    }

    // Imprimimos el mejor recorrido encontrado
    printf("\nMejor recorrido encontrado:\n");
    for(int i = 0; i < longitud_genotipo; i++) {
        printf("%s -> ", nombres_ciudades[Mejor_Individuo->genotipo[i]]);
    }
    printf("%s\n", nombres_ciudades[Mejor_Individuo->genotipo[0]]);
    printf("Distancia total: %f\n", Mejor_Individuo->fitness);

    // Liberamos la memoria en la CPU
    liberar_poblacion(Poblacion);
    liberar_poblacion(padres);
    liberar_poblacion(hijos);
    for(int i = 0; i < longitud_genotipo; i++) {
        free(distancias[i]);
    }
    free(distancias);
    free(h_distancias_flat);
    free(h_ventanas_tiempo_flat);

    free(Mejor_Individuo->genotipo);
    free(Mejor_Individuo);
    free(tempGenotipo);

    // Liberamos la memoria en la GPU
    gpuErrchk(cudaFree(d_distancias));
    gpuErrchk(cudaFree(d_states));
    gpuErrchk(cudaFree(d_ventanas_tiempo));
    gpuErrchk(cudaFree(d_poblacion));
    gpuErrchk(cudaFree(d_padres));
    gpuErrchk(cudaFree(d_hijos));
    gpuErrchk(cudaFree(d_genotipos_poblacion));
    gpuErrchk(cudaFree(d_genotipos_padres));
    gpuErrchk(cudaFree(d_genotipos_hijos));

    // Finalizamos la medición del tiempo
    time_t fin = time(NULL);
    double tiempo = difftime(fin, inicio);
    printf("Tiempo de ejecución: %.2f segundos\n", tiempo);

    return 0;
}