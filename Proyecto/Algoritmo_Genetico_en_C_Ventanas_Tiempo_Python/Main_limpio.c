#include "Biblioteca_c_limpio.h"
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

// Implementación del algoritmo genético
EXPORT ResultadoGenetico* ejecutar_algoritmo_genetico_ventanas_tiempo(
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
    srand(time(NULL));

    // Reservamos memoria para la matriz que almacena las distancias
    double **distancias = malloc(longitud_genotipo * sizeof(double *));
    for (int i = 0; i < longitud_genotipo; i++) {
        distancias[i] = malloc(longitud_genotipo * sizeof(double));
    }

    // Abrimos el archivo
    FILE *archivo = fopen(nombre_archivo, "r");
    if (!archivo) {
        perror("Error al abrir el archivo");
        return NULL;
    }

    // Leemos el archivo y llenamos la matriz
    char linea[8192];
    int fila = 0;
    while (fgets(linea, sizeof(linea), archivo) && fila < longitud_genotipo) {
        char *token = strtok(linea, ",");
        int columna = 0;
        while (token && columna < longitud_genotipo) {
            // Convertimos a double y pasamos las distancias a horas 
            distancias[fila][columna] = (atof(token)) / km_hr;
            token = strtok(NULL, ",");
            columna++;
        }
        fila++;
        free(token);
    }
    fclose(archivo);

    // Creamos un arreglo con los nombres de las ciudades
    char nombres_ciudades[32][19] = {
        "Aguascalientes", "Baja California", "Baja California Sur",
        "Campeche", "Chiapas", "Chihuahua", "Coahuila", "Colima", "Durango",
        "Guanajuato", "Guerrero", "Hidalgo", "Jalisco", "Estado de Mexico",
        "Michoacan", "Morelos", "Nayarit", "Nuevo Leon", "Oaxaca", "Puebla",
        "Queretaro", "Quintana Roo", "San Luis Potosi", "Sinaloa", "Sonora",
        "Tabasco", "Tamaulipas", "Tlaxcala", "Veracruz", "Yucatan",
        "Zacatecas", "CDMX"
    };

    // Cereamos las ventanas de tiempo para cada ciudad
    double ventanas_tiempo[32][2] = {
        {8, 12}, {9, 13}, {10, 14}, {11, 15}, {12, 16}, {13, 17}, {14, 18},
        {15, 19}, {16, 20}, {17, 21}, {18, 22}, {19, 23}, {20, 0}, {21, 1},
        {22, 2}, {23, 3}, {0, 4}, {1, 5}, {2, 6}, {3, 7}, {4, 8}, {5, 9},
        {6, 10}, {7, 11}, {8, 12}, {9, 13}, {10, 14}, {11, 15}, {12, 16},
        {13, 17}, {14, 18}, {15, 19}
    };

    // Inicializamos la población
    poblacion *Poblacion = crear_poblacion(tamano_poblacion, longitud_genotipo);
    poblacion *padres = crear_poblacion(tamano_poblacion, longitud_genotipo);    
    poblacion *hijos = crear_poblacion(tamano_poblacion, longitud_genotipo);

    // Creamos permutaciones aleatorias para cada individuo de la población
    crear_permutaciones(Poblacion, longitud_genotipo);

    // Evaluamos la población
    evaluar_poblacion(Poblacion, distancias, longitud_genotipo, ventanas_tiempo);

    // Ordenamos la población
    ordenar_poblacion(Poblacion);

    // Inicializamos el mejor individuo
    individuo *Mejor_Individuo = (individuo *)malloc(sizeof(individuo));
    Mejor_Individuo->genotipo = (int *)malloc(longitud_genotipo * sizeof(int));

    // Copiamos el mejor individuo de la población a Mejor_Individuo
    for (int i = 0; i < longitud_genotipo; i++) {
        Mejor_Individuo->genotipo[i] = Poblacion->individuos[0].genotipo[i];
    }
    // Copiamos el fitness del mejor individuo
    Mejor_Individuo->fitness = Poblacion->individuos[0].fitness;
    
    // Ejecutamos el algoritmo genético
    for(int generacion=0; generacion<num_generaciones; generacion++){
        // Seleccionamos a los padres
        seleccionar_padres_torneo(Poblacion, padres, num_competidores, longitud_genotipo);

        // Cruzamos a los padres
        cruzar_individuos(padres, hijos, tamano_poblacion, longitud_genotipo, m, distancias, probabilidad_cruce, ventanas_tiempo);

        // Mutamos a los hijos
        for (int i = 0; i < tamano_poblacion; i++) {
            mutar_individuo(&hijos->individuos[i], distancias, probabilidad_mutacion, longitud_genotipo, ventanas_tiempo);
        }

        //Reemplazamos la población actual con los hijos
        actualizar_poblacion(&Poblacion, hijos, longitud_genotipo);

        // Evaluamos a los hijos
        evaluar_poblacion(Poblacion, distancias, longitud_genotipo, ventanas_tiempo);
        ordenar_poblacion(Poblacion);

        // Actualizamos al mejor individuo si es necesario
        if (Poblacion->individuos[0].fitness < Mejor_Individuo->fitness) {
        for (int i = 0; i < longitud_genotipo; i++) {
            Mejor_Individuo->genotipo[i] = Poblacion->individuos[0].genotipo[i];
        }
        Mejor_Individuo->fitness = Poblacion->individuos[0].fitness;
        }
    }

    // Finalizamos la medición del tiempo
    time_t fin = time(NULL);
    double tiempo_ejecucion = difftime(fin, inicio);

    // Preparamos el resultado para devolverlo a Python
    ResultadoGenetico* resultado = (ResultadoGenetico*)malloc(sizeof(ResultadoGenetico));
    resultado->recorrido = (int*)malloc(longitud_genotipo * sizeof(int));
    resultado->nombres_ciudades = malloc(longitud_genotipo * sizeof(char[50]));
    resultado->longitud_recorrido = longitud_genotipo;

    // Rellenamos la estructura con el mejor recorrido y su información
    for (int i = 0; i < longitud_genotipo; i++) {
        resultado->recorrido[i] = Mejor_Individuo->genotipo[i];
        strncpy(resultado->nombres_ciudades[i], nombres_ciudades[Mejor_Individuo->genotipo[i]], 49);
        resultado->nombres_ciudades[i][49] = '\0';
    }
    resultado->aptitud = Mejor_Individuo->fitness;
    resultado->tiempo_ejecucion = tiempo_ejecucion;

    // Liberamos la memoria de todos los elementos
    liberar_poblacion(Poblacion);
    liberar_poblacion(padres);
    liberar_poblacion(hijos);
    for (int i = 0; i < longitud_genotipo; i++) {
        free(distancias[i]);
    }
    free(distancias);
    free(Mejor_Individuo->genotipo);
    Mejor_Individuo->genotipo = NULL;
    free(Mejor_Individuo);

    return resultado;  // Devolvemos el resultado a Python
}

// Función para liberar la memoria del resultado en Python
EXPORT void liberar_resultado(ResultadoGenetico* resultado) {
    if (resultado) {
        free(resultado->recorrido);
        free(resultado->nombres_ciudades);
        free(resultado);
    }
}