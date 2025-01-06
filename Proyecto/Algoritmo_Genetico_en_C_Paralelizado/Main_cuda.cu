#include "Biblioteca_cuda.h"
#include <math.h>  // para log2 etc.

int main(int argc, char** argv) {
    // Iniciamos la medición del tiempo
    time_t inicio = time(NULL);

    // Parámetros del algoritmo genético
    srand(time(NULL));
    int tamano_poblacion = 100000;
    int longitud_genotipo = 32;
    int num_generaciones  = 200;
    int num_competidores  = 2;
    int m = 6;
    double prob_mutacion = 0.15;
    double prob_cruce    = 0.9;

    // Cargar archivo de distancias
    const char *nombre_archivo = "Distancias_no_head.csv";
    double **distancias = (double **)malloc(longitud_genotipo * sizeof(double*));
    for(int i=0; i<longitud_genotipo; i++) {
        distancias[i] = (double*)malloc(longitud_genotipo*sizeof(double));
    }
    FILE *archivo = fopen(nombre_archivo, "r");
    if(!archivo) {
        perror("Error al abrir el archivo");
        return 1;
    }
    char linea[8192];
    int fila=0;
    while(fgets(linea,sizeof(linea), archivo) && fila<longitud_genotipo) {
        char *token = strtok(linea,",");
        int columna=0;
        while(token && columna<longitud_genotipo) {
            distancias[fila][columna] = atof(token);
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
    evaluar_poblacion_kernel<<<gridSize, blockSize>>>(d_poblacion, d_distancias,
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
            d_distancias, prob_cruce,
            tamano_poblacion, longitud_genotipo,
            m, d_states
        );
        cudaDeviceSynchronize();
        gpuErrchk(cudaGetLastError());

        obtenerConfiguracionCUDA(&blockSize, &minGridSize, &gridSize, tamano_poblacion);

        // Mutación
        mutar_individuos_kernel<<<gridSize, blockSize>>>(d_hijos, d_distancias,
                                                         prob_mutacion,
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

        evaluar_poblacion_kernel<<<gridSize, blockSize>>>(d_poblacion, d_distancias,
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

    // Imprimir mejor
    printf("\nMejor recorrido encontrado:\n");
    for(int i=0; i<longitud_genotipo; i++) {
        printf("%s -> ", nombres_ciudades[Mejor_Individuo->genotipo[i]]);
    }
    printf("%s\n", nombres_ciudades[Mejor_Individuo->genotipo[0]]);
    printf("Distancia total: %f\n", Mejor_Individuo->fitness);

    // Liberar memoria CPU
    liberar_poblacion(Poblacion);
    liberar_poblacion(padres);
    liberar_poblacion(hijos);
    for(int i=0; i<longitud_genotipo; i++) {
        free(distancias[i]);
    }
    free(distancias);
    free(h_distancias_flat);

    free(Mejor_Individuo->genotipo);
    free(Mejor_Individuo);

    // Liberar memoria GPU
    gpuErrchk(cudaFree(d_distancias));
    gpuErrchk(cudaFree(d_states));
    gpuErrchk(cudaFree(d_poblacion));
    gpuErrchk(cudaFree(d_padres));
    gpuErrchk(cudaFree(d_hijos));
    gpuErrchk(cudaFree(d_genotipos_poblacion));
    gpuErrchk(cudaFree(d_genotipos_padres));
    gpuErrchk(cudaFree(d_genotipos_hijos));

    // Tiempo final
    time_t fin = time(NULL);
    double tiempo = difftime(fin, inicio);
    printf("Tiempo de ejecucion: %.2f segundos\n", tiempo);

    return 0;
}
