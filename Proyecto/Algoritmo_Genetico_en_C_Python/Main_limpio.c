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
EXPORT ResultadoGenetico* ejecutar_algoritmo_genetico(
    int tamano_poblacion,         // Tamaño de la población inicial
    int longitud_genotipo,        // Longitud del genotipo (número de ciudades)
    int num_generaciones,         // Número de generaciones a ejecutar
    int num_competidores,         // Número de competidores en el torneo de selección
    int m,                        // Valor para considerar las m ciudades más cercanas en la heurística
    double probabilidad_mutacion, // Probabilidad de mutación
    double probabilidad_cruce,    // Probabilidad de cruce
    const char* nombre_archivo    // Nombre del archivo con la matriz de distancias
) {
    // Iniciamos la medición del tiempo de ejecución
    time_t inicio = time(NULL);

    // Inicializamos el generador de números aleatorios con la hora actual
    srand(time(NULL));

    // Reservamos memoria para la matriz de distancias entre las ciudades
    double **distancias = malloc(longitud_genotipo * sizeof(double *));
    for (int i = 0; i < longitud_genotipo; i++) {
        distancias[i] = malloc(longitud_genotipo * sizeof(double));
    }

    // Abrimos el archivo que contiene la matriz de distancias
    FILE *archivo = fopen(nombre_archivo, "r");
    if (!archivo) {
        printf("Error al abrir el archivo %s\n", nombre_archivo);
        return NULL;
    }

    // Leemos el archivo y llenamos la matriz de distancias
    char linea[8192];
    int fila = 0;
    while (fgets(linea, sizeof(linea), archivo) && fila < longitud_genotipo) {
        char *token = strtok(linea, ",");  // Separamos los valores por comas
        int columna = 0;
        while (token && columna < longitud_genotipo) {
            distancias[fila][columna] = atof(token);  // Convertimos los valores a double
            token = strtok(NULL, ",");
            columna++;
        }
        fila++;
    }
    fclose(archivo);

    // Definimos los nombres de las ciudades
    const char nombres_ciudades[32][50] = {
        "Aguascalientes", "Baja California", "Baja California Sur",
        "Campeche", "Chiapas", "Chihuahua", "Coahuila", "Colima", "Durango",
        "Guanajuato", "Guerrero", "Hidalgo", "Jalisco", "Estado de Mexico",
        "Michoacan", "Morelos", "Nayarit", "Nuevo Leon", "Oaxaca", "Puebla",
        "Queretaro", "Quintana Roo", "San Luis Potosi", "Sinaloa", "Sonora",
        "Tabasco", "Tamaulipas", "Tlaxcala", "Veracruz", "Yucatan",
        "Zacatecas", "CDMX"
    };

    // Inicializamos las poblaciones para el algoritmo genético
    poblacion *Poblacion = crear_poblacion(tamano_poblacion, longitud_genotipo);
    poblacion *padres = crear_poblacion(tamano_poblacion, longitud_genotipo);
    poblacion *hijos = crear_poblacion(tamano_poblacion, longitud_genotipo);

    // Creamos la población inicial con permutaciones aleatorias
    crear_permutaciones(Poblacion, longitud_genotipo);

    // Evaluamos la población inicial y la ordenamos por aptitud
    evaluar_poblacion(Poblacion, distancias, longitud_genotipo);
    ordenar_poblacion(Poblacion);

    // Inicializamos el mejor individuo como el mejor de la población inicial
    individuo *MejorIndividuo = (individuo *)malloc(sizeof(individuo));
    MejorIndividuo->genotipo = (int *)malloc(longitud_genotipo * sizeof(int));
    for (int i = 0; i < longitud_genotipo; i++) {
        MejorIndividuo->genotipo[i] = Poblacion->individuos[0].genotipo[i];
    }
    MejorIndividuo->fitness = Poblacion->individuos[0].fitness;

    // Bucle principal del algoritmo genético
    for (int generacion = 0; generacion < num_generaciones; generacion++) {
        // Selección de padres mediante torneo
        seleccionar_padres_torneo(Poblacion, padres, num_competidores, longitud_genotipo);

        // Cruce entre los padres para generar hijos
        cruzar_individuos(padres, hijos, tamano_poblacion, longitud_genotipo, m, distancias, probabilidad_cruce);

        // Aplicación de mutación en los hijos
        for (int i = 0; i < tamano_poblacion; i++) {
            mutar_individuo(&hijos->individuos[i], distancias, probabilidad_mutacion, longitud_genotipo);
        }

        // Actualización de la población con los hijos generados
        actualizar_poblacion(&Poblacion, hijos, longitud_genotipo);

        // Evaluación y ordenamiento de la nueva población
        evaluar_poblacion(Poblacion, distancias, longitud_genotipo);
        ordenar_poblacion(Poblacion);

        // Actualización del mejor individuo si se encuentra uno con mejor aptitud
        if (Poblacion->individuos[0].fitness < MejorIndividuo->fitness) {
            for (int i = 0; i < longitud_genotipo; i++) {
                MejorIndividuo->genotipo[i] = Poblacion->individuos[0].genotipo[i];
            }
            MejorIndividuo->fitness = Poblacion->individuos[0].fitness;
        }
    }

    // Calculamos el tiempo total de ejecución del algoritmo
    time_t fin = time(NULL);
    double tiempo_ejecucion = difftime(fin, inicio);

    // Preparamos el resultado para devolverlo a Python
    ResultadoGenetico* resultado = (ResultadoGenetico*)malloc(sizeof(ResultadoGenetico));
    resultado->recorrido = (int*)malloc(longitud_genotipo * sizeof(int));
    resultado->nombres_ciudades = malloc(longitud_genotipo * sizeof(char[50]));
    resultado->longitud_recorrido = longitud_genotipo;

    // Rellenamos la estructura con el mejor recorrido y su información
    for (int i = 0; i < longitud_genotipo; i++) {
        resultado->recorrido[i] = MejorIndividuo->genotipo[i];
        strncpy(resultado->nombres_ciudades[i], nombres_ciudades[MejorIndividuo->genotipo[i]], 49);
        resultado->nombres_ciudades[i][49] = '\0';
    }
    resultado->aptitud = MejorIndividuo->fitness;
    resultado->tiempo_ejecucion = tiempo_ejecucion;

    // Liberamos toda la memoria dinámica usada
    liberar_poblacion(Poblacion);
    liberar_poblacion(padres);
    liberar_poblacion(hijos);
    for (int i = 0; i < longitud_genotipo; i++) {
        free(distancias[i]);
    }
    free(distancias);
    free(MejorIndividuo->genotipo);
    free(MejorIndividuo);

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
