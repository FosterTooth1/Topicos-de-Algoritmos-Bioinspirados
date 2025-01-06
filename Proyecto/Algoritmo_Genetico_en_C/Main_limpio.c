#include "Biblioteca_c_limpio.h"

int main(int argc, char** argv){
    // Iniciamos la medición del tiempo
    time_t inicio = time(NULL);

    // Parámetros del algoritmo genético
    srand(time(NULL));
    int tamano_poblacion = 100000;
    int longitud_genotipo = 32;
    int num_generaciones  = 100;
    int num_competidores  = 2;
    int m = 3;
    double probabilidad_mutacion = 0.15;
    double probabilidad_cruce = 0.99;

    // Nombre del archivo con las distancias
    char *nombre_archivo = "Distancias_no_head.csv";

    // Reservamos memoria para la matriz que almacena las distancias
    double **distancias = malloc(longitud_genotipo * sizeof(double *));
    for (int i = 0; i < longitud_genotipo; i++) {
        distancias[i] = malloc(longitud_genotipo * sizeof(double));
    }

    // Abrimos el archivo
    FILE *archivo = fopen(nombre_archivo, "r");
    if (!archivo) {
        perror("Error al abrir el archivo");
        return 1;
    }

    // Leemos el archivo y llenamos la matriz
    char linea[8192];
    int fila = 0;
    while (fgets(linea, sizeof(linea), archivo) && fila < longitud_genotipo) {
        char *token = strtok(linea, ",");
        int columna = 0;
        while (token && columna < longitud_genotipo) {
            distancias[fila][columna] = atof(token);
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
        "Guanajuato", "Guerrero", "Hidalgo", "Jalisco", "Estado de México",
        "Michoacán", "Morelos", "Nayarit", "Nuevo León", "Oaxaca", "Puebla",
        "Querétaro", "Quintana Roo", "San Luis Potosí", "Sinaloa", "Sonora",
        "Tabasco", "Tamaulipas", "Tlaxcala", "Veracruz", "Yucatán",
        "Zacatecas", "CDMX"
    };

    // Inicializamos la población
    poblacion *Poblacion = crear_poblacion(tamano_poblacion, longitud_genotipo);
    poblacion *padres = crear_poblacion(tamano_poblacion, longitud_genotipo);    
    poblacion *hijos = crear_poblacion(tamano_poblacion, longitud_genotipo);

    // Creamos permutaciones aleatorias para cada individuo de la población
    crear_permutaciones(Poblacion, longitud_genotipo);

    // Evaluamos la población
    evaluar_poblacion(Poblacion, distancias, longitud_genotipo);

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
        cruzar_individuos(padres, hijos, tamano_poblacion, longitud_genotipo, m, distancias, probabilidad_cruce);

        // Mutamos a los hijos
        for (int i = 0; i < tamano_poblacion; i++) {
            mutar_individuo(&hijos->individuos[i], distancias, probabilidad_mutacion, longitud_genotipo);
        }

        //Reemplazamos la población actual con los hijos
        actualizar_poblacion(&Poblacion, hijos, longitud_genotipo);

        // Evaluamos a los hijos
        evaluar_poblacion(Poblacion, distancias, longitud_genotipo);
        ordenar_poblacion(Poblacion);

        // Actualizamos al mejor individuo si es necesario
        if (Poblacion->individuos[0].fitness < Mejor_Individuo->fitness) {
        for (int i = 0; i < longitud_genotipo; i++) {
            Mejor_Individuo->genotipo[i] = Poblacion->individuos[0].genotipo[i];
        }
        Mejor_Individuo->fitness = Poblacion->individuos[0].fitness;
        }
    }

    // Imprimimos al mejor individuo
    printf("Mejor Individuo: ");
    for (int i = 0; i < longitud_genotipo; i++) {
        printf("%d  ", Mejor_Individuo->genotipo[i]);
    }
    printf(" Fitness: %f\n", Mejor_Individuo->fitness);

    // Imprimimos el recorrido
    printf("Recorrido: ");
    for (int i = 0; i < longitud_genotipo; i++) {
        printf("%s -> ", nombres_ciudades[Mejor_Individuo->genotipo[i]]);
    }

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

    // Finalizamos la medición del tiempo
    time_t fin = time(NULL);

    // Imprimimos el tiempo de ejecución
    double tiempo_ejecucion = difftime(fin, inicio);
    printf("Tiempo de ejecución: %.2f segundos\n", tiempo_ejecucion);

    return 0;
}