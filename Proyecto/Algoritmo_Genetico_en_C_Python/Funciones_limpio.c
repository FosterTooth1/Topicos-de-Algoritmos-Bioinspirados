#include "Biblioteca_c_limpio.h"

//Funciones principales del algoritmo genético

//Asigna memoria para una población
//Recibe el tamaño de la población y la longitud del genotipo
//Devuelve un puntero a la población creada
poblacion *crear_poblacion(int tamano, int longitud_genotipo) {
    // Asigna memoria para la estructura de la población
    poblacion *Poblacion = malloc(sizeof(poblacion));

    // Asigna memoria para los individuos
    Poblacion->tamano = tamano;
    Poblacion->individuos = malloc(tamano * sizeof(individuo));

    // Asigna memoria para los genotipos de cada individuo
    for(int i = 0; i < tamano; i++) {
        Poblacion->individuos[i].genotipo = malloc(longitud_genotipo * sizeof(int));

        // Inicializa el fitness en 0
        Poblacion->individuos[i].fitness = 0;
    }
    return Poblacion;
}

// Crea permutaciones aleatorias para cada individuo de la población
// Recibe un puntero a la población y la longitud del genotipo
// No devuelve nada (todo se hace por referencia)
void crear_permutaciones(poblacion *poblacion, int longitud_genotipo) {
    for (int i = 0; i < poblacion->tamano; i++) {
        
        // Inicializa el genotipo con valores ordenados
        for (int j = 0; j < longitud_genotipo; j++) {
            poblacion->individuos[i].genotipo[j] = j;
        }
        
        // Mezcla el genotipo utilizando el algoritmo de Fisher-Yates
        for (int j = longitud_genotipo - 1; j > 0; j--) {
            int k = rand() % (j + 1);
            int temp = poblacion->individuos[i].genotipo[j];
            poblacion->individuos[i].genotipo[j] = poblacion->individuos[i].genotipo[k];
            poblacion->individuos[i].genotipo[k] = temp;
        }
    }
}

// Evalua la población basándose en las distancias entre las ciudades (fitness)
// Recibe un puntero a la población, una matriz de distancias y la longitud del genotipo
// No devuelve nada (todo se hace por referencia)
void evaluar_poblacion(poblacion *poblacion, double **distancias, int longitud_genotipo) {
    // Evalua cada individuo de la población
    for (int i = 0; i < poblacion->tamano; i++) {
        poblacion->individuos[i].fitness = evaluar_individuo(
            poblacion->individuos[i].genotipo, distancias, longitud_genotipo);
    }
}

// Función para calcular la distancia total recorrida por el individuo (fitness)
// Recibe un genotipo, una matriz de distancias y la longitud del genotipo
// Devuelve el fitness del individuo
double evaluar_individuo(int *genotipo, double **distancias, int longitud_genotipo) {
    double total_cost = 0.0;
    for (int i = 0; i < longitud_genotipo - 1; i++) {
        total_cost += distancias[genotipo[i]][genotipo[i + 1]];
    }
    total_cost += distancias[genotipo[longitud_genotipo - 1]][genotipo[0]];
    return total_cost;
}

// Función principal de ordenamiento para la población
// Recibe un puntero a la población
// No devuelve nada (todo se hace por referencia)
void ordenar_poblacion(poblacion *poblacion) {
    // Obtenemos el tamaño de la población
    int n = poblacion->tamano;
    
    // Si la población igual o menor a 1, no se hace nada
    if (n <= 1) return;
    
    // Calculamos la profundidad máxima de recursión
    int profundidad_max = 2 * log2_suelo(n);
    
    // Llamamos a la función auxiliar de ordenamiento introspectivo
    introsort_util(poblacion->individuos, &profundidad_max, 0, n);
}

// Selección de padres mediante un torneo de fitness
// Recibe un puntero a la población, un puntero a la población de padres, el número de competidores y la longitud del genotipo
// No devuelve nada (todo se hace por referencia)
void seleccionar_padres_torneo(poblacion *Poblacion, poblacion *padres, int num_competidores, int longitud_genotipo) {
    // Inicializamos un array para los índices de los competidores
    int tamano_poblacion = Poblacion->tamano;
    int *indices_torneo = malloc(num_competidores * sizeof(int));

    // Realizamos un torneo para seleccionar a un padre
    for (int i = 0; i < tamano_poblacion; i++) {

        // Seleccionamos al azar los competidores del torneo
        for (int j = 0; j < num_competidores; j++) {
            indices_torneo[j] = rand() % tamano_poblacion;
        }

        // Encontramos el ganador del torneo evaluando su fitness
        int indice_ganador = indices_torneo[0];
        double mejor_fitness = Poblacion->individuos[indices_torneo[0]].fitness;
        for (int j = 1; j < num_competidores; j++) {
            int indice_actual = indices_torneo[j];
            double fitness_actual = Poblacion->individuos[indice_actual].fitness;

            if (fitness_actual < mejor_fitness) {
                mejor_fitness = fitness_actual;
                indice_ganador = indice_actual;
            }
        }

        // Copiamos el individuo ganador a la población de padres
        for (int j = 0; j < longitud_genotipo; j++) {
            padres->individuos[i].genotipo[j] = Poblacion->individuos[indice_ganador].genotipo[j];
        }

        // Copiamos el fitness del ganador
        padres->individuos[i].fitness = Poblacion->individuos[indice_ganador].fitness;
    }

    // Liberamos la memoria usada para los índices
    free(indices_torneo);
}

// Cruza a los padres para generar a los hijos dependiendo de una probabilidad de cruce
// Recibe un puntero a la población destino, un puntero a la población origen y la longitud del genotipo
// No devuelve nada (todo se hace por referencia)
void cruzar_individuos(poblacion *padres, poblacion *hijos, int num_pob, int longitud_genotipo, int m, double **distancias, double probabilidad_cruce) {
    for (int i = 0; i < num_pob; i += 2) {
        // Genera un número aleatorio y determina si se realiza la cruza
        if ((double)rand() / RAND_MAX < probabilidad_cruce) {

            // Se asigna memoria para los hijos
            int *hijo1 = malloc(longitud_genotipo * sizeof(int));
            int *hijo2 = malloc(longitud_genotipo * sizeof(int));

            // Cruza los padres para generar a dos hijos
            cycle_crossover(padres->individuos[i].genotipo, padres->individuos[i + 1].genotipo, hijo1, longitud_genotipo);
            cycle_crossover(padres->individuos[i + 1].genotipo, padres->individuos[i].genotipo, hijo2, longitud_genotipo);

            heuristica_abruptos(hijo1, longitud_genotipo, m, distancias);
            heuristica_abruptos(hijo2, longitud_genotipo, m, distancias);

            // Crea un array temporal para almacenar los individuos
            int **individuos = malloc(4 * sizeof(int *));
            individuos[0] = padres->individuos[i].genotipo;
            individuos[1] = padres->individuos[i + 1].genotipo;
            individuos[2] = hijo1;
            individuos[3] = hijo2;

            // Evalua a los individuos
            individuo temp_hijos[4];
            for (int j = 0; j < 4; j++) {
                temp_hijos[j].genotipo = individuos[j];
                temp_hijos[j].fitness = evaluar_individuo(individuos[j], distancias, longitud_genotipo);
            }

            // Selecciona los mejores dos individuos
            int mejores_indices[2] = {0, 1};
            for (int j = 2; j < 4; j++) {
                if (temp_hijos[j].fitness < temp_hijos[mejores_indices[0]].fitness) {
                    mejores_indices[1] = mejores_indices[0];
                    mejores_indices[0] = j;
                } else if (temp_hijos[j].fitness < temp_hijos[mejores_indices[1]].fitness) {
                    mejores_indices[1] = j;
                }
            }

            // Asigna los mejores individuos a los hijos
            for (int j = 0; j < longitud_genotipo; j++) {
                hijos->individuos[i].genotipo[j] = individuos[mejores_indices[0]][j];
                hijos->individuos[i + 1].genotipo[j] = individuos[mejores_indices[1]][j];
            }
            hijos->individuos[i].fitness = temp_hijos[mejores_indices[0]].fitness;
            hijos->individuos[i + 1].fitness = temp_hijos[mejores_indices[1]].fitness;

            // Libera memoria de los hijos temporales
            free(hijo1);
            free(hijo2);
            free(individuos);
            
        } else {
            // Si no hay cruce, copia los padres directamente a los hijos
            for (int j = 0; j < longitud_genotipo; j++) {
                hijos->individuos[i].genotipo[j] = padres->individuos[i].genotipo[j];
                hijos->individuos[i + 1].genotipo[j] = padres->individuos[i + 1].genotipo[j];
            }
            hijos->individuos[i].fitness = padres->individuos[i].fitness;
            hijos->individuos[i + 1].fitness = padres->individuos[i + 1].fitness;
        }
    }
}

// Muta a un individuo basado en una probabilidad de mutación
// Recibe un puntero al individuo, una matriz de distancias, la probabilidad de mutación y la longitud del genotipo
// No devuelve nada (todo se hace por referencia)
void mutar_individuo(individuo *individuo, double **distancias, double probabilidad_mutacion, int longitud_genotipo) {
    // Genera un número aleatorio y determina si se realiza la mutación
    if ((double)rand() / RAND_MAX < probabilidad_mutacion) {

        // Genera dos índices aleatorios distintos
        int idx1 = rand() % longitud_genotipo;
        int idx2 = rand() % longitud_genotipo;
        while (idx1 == idx2) {
            idx2 = rand() % longitud_genotipo;
        }

        // Intercambia los genes en las posiciones idx1 e idx2
        int temp = individuo->genotipo[idx1];
        individuo->genotipo[idx1] = individuo->genotipo[idx2];
        individuo->genotipo[idx2] = temp;

        // Recalcula el fitness del individuo usando la nueva evaluar_individuo
        individuo->fitness = evaluar_individuo(individuo->genotipo, distancias, longitud_genotipo);
    }
}

// Actualiza la población destino copiando los datos de la población origen
// Recibe un puntero a la población destino, un puntero a la población origen y la longitud del genotipo
// No devuelve nada (todo se hace por referencia)
void actualizar_poblacion(poblacion **destino, poblacion *origen, int longitud_genotipo) {
    // Crea una población temporal nueva
    poblacion *nueva = crear_poblacion(origen->tamano, longitud_genotipo);

    // Copia los datos
    for (int i = 0; i < origen->tamano; i++) {
        for (int j = 0; j < longitud_genotipo; j++) {
            nueva->individuos[i].genotipo[j] = origen->individuos[i].genotipo[j];
        }
        nueva->individuos[i].fitness = origen->individuos[i].fitness;
    }

    // Libera la población antigua si existe
    if (*destino != NULL) {
        for (int i = 0; i < (*destino)->tamano; i++) {
            if ((*destino)->individuos[i].genotipo != NULL) {
                free((*destino)->individuos[i].genotipo);
            }
        }
        if ((*destino)->individuos != NULL) {
            free((*destino)->individuos);
        }
        free(*destino);
    }

    // Asigna la nueva población
    *destino = nueva;
}

// Libera la memoria usada por una población
// Recibe un puntero a la población
// No devuelve nada (todo se hace por referencia)
void liberar_poblacion(poblacion *pob) {
    // Verifica si la población es nula
    if (pob == NULL) return;

    // Libera la memoria de los genotipos de cada individuo
    if (pob->individuos != NULL) {
        for (int i = 0; i < pob->tamano; i++) {
            if (pob->individuos[i].genotipo != NULL) {
                free(pob->individuos[i].genotipo);
                pob->individuos[i].genotipo = NULL;
            }
        }
        free(pob->individuos);
        pob->individuos = NULL;
    }

    // Libera la memoria de la población
    free(pob);
}

// Funciones auxiliares del cruzamiento

// Heurística para remover abruptos en la ruta intercambiando ciudades mal posicionadas
// Recibe un puntero a la ruta, el número de ciudades total (longitud del genotipo), el número de ciudades más cercanas a considerar y la matriz de distancias
// No devuelve nada (todo se hace por referencia)
void heuristica_abruptos(int* ruta, int num_ciudades, int m, double** distancias) {
    // Inicializamos memoria para un arreglo temporal para la manipulación de rutas
    int* ruta_temp = malloc(num_ciudades * sizeof(int));

    // Inicializamos meemoria para la estructura que sirve para ordenar distancias
    DistanciaOrdenada* dist_ordenadas = malloc(num_ciudades * sizeof(DistanciaOrdenada));

    // Para cada ciudad en la ruta
    for (int i = 0; i < num_ciudades; i++) {
        int ciudad_actual = ruta[i];
        
        // Se obtiene y ordenan las m ciudades más cercanas
        for (int j = 0; j < num_ciudades; j++) {
            dist_ordenadas[j].distancia = distancias[ciudad_actual][j];
            dist_ordenadas[j].indice = j;
        }
        qsort(dist_ordenadas, num_ciudades, sizeof(DistanciaOrdenada), comparar_distancias);

        // Encontramos la posición actual de la ciudad en la ruta
        int pos_actual = -1;
        for (int j = 0; j < num_ciudades; j++) {
            if (ruta[j] == ciudad_actual) {
                pos_actual = j;
                break;
            }
        }

        // Inicializamos el mejor costo con el costo actual
        double mejor_costo = evaluar_individuo(ruta, distancias, num_ciudades);
        int mejor_posicion = pos_actual;
        int mejor_vecino = -1;

        // Probamos la inserción con las m ciudades más cercanas
        for (int j = 1; j <= m && j < num_ciudades; j++) {
            int ciudad_cercana = dist_ordenadas[j].indice;
            
            // Encontramos la posición de la ciudad cercana
            int pos_cercana = -1;
            for (int k = 0; k < num_ciudades; k++) {
                if (ruta[k] == ciudad_cercana) {
                    pos_cercana = k;
                    break;
                }
            }

            if (pos_cercana != -1) {
                // Probar inserción antes y después de la ciudad cercana
                for (int posicion_antes_o_despues = 0; posicion_antes_o_despues <= 1; posicion_antes_o_despues++) {
                    memcpy(ruta_temp, ruta, num_ciudades * sizeof(int));
                    
                    // Eliminar de posición actual
                    eliminar_de_posicion(ruta_temp, num_ciudades, pos_actual);
                    
                    // Insertar en nueva posición (antes o después de la ciudad cercana)
                    int nueva_pos = pos_cercana + posicion_antes_o_despues;
                    if (nueva_pos > pos_actual) nueva_pos--;
                    if (nueva_pos >= num_ciudades) nueva_pos = num_ciudades - 1;
                    insertar_en_posicion(ruta_temp, num_ciudades, ciudad_actual, nueva_pos);

                    // Evaluar el nuevo costo
                    double nuevo_costo = evaluar_individuo(ruta_temp, distancias, num_ciudades);
                    
                    // Actualizar el mejor costo y posición de la ciudad actual si es necesario
                    if (nuevo_costo < mejor_costo) {
                        mejor_costo = nuevo_costo;
                        mejor_posicion = nueva_pos;
                        mejor_vecino = ciudad_cercana;
                    }
                }
            }
        }

        // Si se encontró un mejor vecino, actualizar la ruta
        if (mejor_vecino != -1 && mejor_posicion != pos_actual) {
            memcpy(ruta_temp, ruta, num_ciudades * sizeof(int));
            eliminar_de_posicion(ruta_temp, num_ciudades, pos_actual);
            insertar_en_posicion(ruta_temp, num_ciudades, ciudad_actual, mejor_posicion);
            memcpy(ruta, ruta_temp, num_ciudades * sizeof(int));
        }
    }

    // Liberamos memoria
    free(ruta_temp);
    free(dist_ordenadas);
}

// Cruza a dos padres para generar a dos hijos mediante el operador de ciclo
// Recibe los genotipos de los padres, el genotipo del hijo y el numero de ciudades total (longitud del genotipo)
// No devuelve nada (todo se hace por referencia)
void cycle_crossover(int *padre1, int *padre2, int *hijo, int num_ciudades) {
    // Inicializamos el hijo con -1 en todo su genotipo (marca de no visitado)
    for (int i = 0; i < num_ciudades; i++) {
        hijo[i] = -1;
    }

    // Inicializamos un array para marcar las posiciones ya visitadas
    int *visitado = calloc(num_ciudades, sizeof(int));

    // Inicializamos un ciclo para seguir el ciclo de los padres
    int ciclo = 0;
    int posiciones_restantes = num_ciudades;

    // Realizamos el cambio de ciclo mientras queden posiciones por visitar 
    while (posiciones_restantes > 0) {
        // Encontramos la primera posición no visitada
        int inicio = -1;
        for (int i = 0; i < num_ciudades; i++) {
            if (!visitado[i]) {
                inicio = i;
                break;
            }
        }

        ciclo++;
        int actual = inicio;

        // Seguimos un ciclo hasta que se cierre
        while (1) {
            // Marcamos la posición actual como visitada
            visitado[actual] = 1;
            posiciones_restantes--;

            // Asignamos el valor del padre correspondiente al hijo (dependiendo del ciclo)
            hijo[actual] = (ciclo % 2 == 1) ? padre1[actual] : padre2[actual];

            // Encontramos la siguiente posición en el ciclo
            int valor_buscar = padre2[actual];
            int siguiente = -1;
            for (int i = 0; i < num_ciudades; i++) {
                if (padre1[i] == valor_buscar) {
                    siguiente = i;
                    break;
                }
            }

            // Si la siguiente posición ya fue visitada, terminamos este ciclo
            if (visitado[siguiente]) {
                break;
            }

            // Actualizamos la posición actual
            actual = siguiente;
        }
    }

    // Liberamos la memoria usada para el array de visitados
    free(visitado);
}

// Funciones auxiliares de ordenamiento

// Implementación de ordenamiento introspectivo
// Recibe un array de individuos, la profundidad máxima de recursión, el índice de inicio y fin
// No devuelve nada (todo se hace por referencia)
void introsort_util(individuo *arr, int *profundidad_max, int inicio, int fin) {
    // Calculamos el tamaño de la partición
    int tamano = fin - inicio;
    
    // Si el tamaño de la partición es pequeño, usamos el ordenamiento por inserción
    if (tamano < 16) {
        insertion_sort(arr, inicio, fin - 1);
        return;
    }
    
    // Si la profundidad máxima es cero, cambiamos a heapsort (para evitar peor caso de quicksort)
    if (*profundidad_max == 0) {
        heapsort(arr + inicio, tamano);
        return;
    }
    
    // En caso contrario, usamos quicksort
    (*profundidad_max)--;
    int pivote = particion(arr, inicio, fin - 1);
    introsort_util(arr, profundidad_max, inicio, pivote);
    introsort_util(arr, profundidad_max, pivote + 1, fin);
}

// Función para calcular el logaritmo en base 2 de un número entero (parte entera)
// Recibe un número entero
// Devuelve el logaritmo en base 2 (parte entera)
int log2_suelo(int n) {
    int log = 0;
    while (n > 1) {
        n >>= 1;
        log++;
    }
    return log;
}

// Partición de quicksort usando la mediana de tres como pivote
// Recibe un array de individuos, los índices bajo y alto
// Devuelve el índice del pivote
int particion(individuo *arr, int bajo, int alto) {
    // Encontramos el índice del pivote usando la mediana de tres
    int medio = bajo + (alto - bajo) / 2;
    int indice_pivote = mediana_de_tres(arr, bajo, medio, alto);
    
    // Movemos el pivote seleccionado al final del rango para facilitar partición
    intercambiar_individuos(&arr[indice_pivote], &arr[alto]);

    // Guardamos el elemento del pivote para comparación
    individuo pivote = arr[alto];

    // i indica la última posición donde los elementos son menores o iguales al pivote
    int i = bajo - 1;

    // Recorremos el rango desde `bajo` hasta `alto - 1` (excluyendo el pivote)
    for (int j = bajo; j < alto; j++) {
        // Si el elemento actual es menor o igual al pivote
        if (arr[j].fitness <= pivote.fitness) {
            i++; // Avanzamos `i` para marcar la posición de intercambio
            intercambiar_individuos(&arr[i], &arr[j]); // Intercambiamos el elemento menor al pivote
        }
    }

    // Finalmente, colocamos el pivote en su posición correcta
    intercambiar_individuos(&arr[i + 1], &arr[alto]);

    // Retornamos la posición del pivote
    return i + 1;
}

// Función para encontrar la mediana de tres elementos (usado en quicksort para mejorar el balanceo)
// Recibe un array de individuos y tres índices
// Devuelve el índice de la mediana
int mediana_de_tres(individuo *arr, int a, int b, int c) {
    // Se realizan comparaciones lógicas para encontrar la mediana
    if (arr[a].fitness <= arr[b].fitness) {
        if (arr[b].fitness <= arr[c].fitness)
            return b;
        else if (arr[a].fitness <= arr[c].fitness)
            return c;
        else
            return a;
    } else {
        if (arr[a].fitness <= arr[c].fitness)
            return a;
        else if (arr[b].fitness <= arr[c].fitness)
            return c;
        else
            return b;
    }
}

// Función para intercambiar dos elementos
// Recibe dos punteros a individuos
// No devuelve nada (todo se hace por referencia)
void intercambiar_individuos(individuo *a, individuo *b) {
    individuo temp = *a;
    *a = *b;
    *b = temp;
}

// Ordenamiento por inserción para arreglos pequeños
// Recibe un array de individuos, el índice izquierdo y derecho
// No devuelve nada (todo se hace por referencia)
void insertion_sort(individuo *arr, int izquierda, int derecha) {
    // Recorremos el array de izquierda a derecha
    for (int i = izquierda + 1; i <= derecha; i++) {
        // Insertamos el elemento actual en la posición correcta
        individuo clave = arr[i];
        int j = i - 1;
        
        // Movemos los elementos mayores que la clave a una posición adelante
        while (j >= izquierda && arr[j].fitness > clave.fitness) {
            arr[j + 1] = arr[j];
            j--;
        }

        // Insertamos la clave en la posición correcta
        arr[j + 1] = clave;
    }
}

// Heapsort para ordenar a los individuos por fitness
// Recibe un array de individuos y el tamaño del array
// No devuelve nada (todo se hace por referencia)
void heapsort(individuo *arr, int n) {
    // Construimos el montón (heapify)
    for (int i = n / 2 - 1; i >= 0; i--)
        heapify(arr, n, i);

    // Extraemos los elementos del montón uno por uno
    for (int i = n - 1; i > 0; i--) {
        individuo temp = arr[0];
        arr[0] = arr[i];
        arr[i] = temp;
        heapify(arr, i, 0);
    }
}

// Función auxiliar para heapsort
// Recibe un array de individuos, el tamaño del array y un índice
// No devuelve nada (todo se hace por referencia)
void heapify(individuo *arr, int n, int i) {
    // Inicializamos el mayor como el indice actual
    int mayor = i;

    // Calculamos los indices de los hijos izquierdo y derecho
    int izquierda = 2 * i + 1;
    int derecha = 2 * i + 2;

    // Si el hijo izquierdo es mayor que el padre actualizamos el mayor
    if (izquierda < n && arr[izquierda].fitness > arr[mayor].fitness)
        mayor = izquierda;

    // Si el hijo derecho es mayor que el padre actualizamos el mayor
    if (derecha < n && arr[derecha].fitness > arr[mayor].fitness)
        mayor = derecha;

    // Si el mayor no es el padre, intercambiamos y aplicamos heapify al subárbol
    if (mayor != i) {
        individuo temp = arr[i];
        arr[i] = arr[mayor];
        arr[mayor] = temp;
        heapify(arr, n, mayor);
    }
}

//Funciones auxiliares de manipulación de arreglos (Usadas en la heurística de remoción de abruptos)

// Función de comparación para qsort
// Recibe dos punteros a distancia ordenada
// Devuelve un entero que indica la relación entre las distancias
int comparar_distancias(const void* a, const void* b) {
    DistanciaOrdenada* da = (DistanciaOrdenada*)a;
    DistanciaOrdenada* db = (DistanciaOrdenada*)b;
    if (da->distancia < db->distancia) return -1;
    if (da->distancia > db->distancia) return 1;
    return 0;
}

// Función para insertar un elemento en una posición específica del array
// Recibe un puntero al array, la longitud del array, el elemento a insertar y la posición
// No devuelve nada (todo se hace por referencia)
void insertar_en_posicion(int* array, int longitud, int elemento, int posicion) {
    for (int i = longitud - 1; i > posicion; i--) {
        array[i] = array[i - 1];
    }
    array[posicion] = elemento;
}

// Función para eliminar un elemento de una posición específica
// Recibe un puntero al array, la longitud del array y la posición
// No devuelve nada (todo se hace por referencia)
void eliminar_de_posicion(int* array, int longitud, int posicion) {
    for (int i = posicion; i < longitud - 1; i++) {
        array[i] = array[i + 1];
    }
}