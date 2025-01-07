import numpy as np
import time
import pandas as pd
from numba import njit
import time
import random
import ctypes
from ctypes import c_int, c_double, c_char_p, c_char, POINTER, Structure
import os
import matplotlib.pyplot as plt  

############################################################
# Algoritmo Genetico con Ventanas de Tiempo en CUDA Python #
############################################################

@njit
def costo_cu(individuo, distancias, ventanas_tiempo):
    total_cost = 0.0
    tiempo_acumulado = 0.0

    for i in range(len(individuo)):
        origen = individuo[i]
        destino = individuo[(i + 1) % len(individuo)]

        tiempo_viaje = distancias[origen][destino]
        tiempo_acumulado += tiempo_viaje

        hora_llegada = tiempo_acumulado % 24
        ventana_inicio, ventana_fin = ventanas_tiempo[destino]

        if hora_llegada < ventana_inicio:
            tiempo_acumulado += (ventana_inicio - hora_llegada)
        elif ventana_fin < ventana_inicio:
            if hora_llegada > ventana_fin and hora_llegada < ventana_inicio:
                tiempo_acumulado += (24 - hora_llegada + ventana_inicio)
        elif hora_llegada > ventana_fin:
            tiempo_acumulado += (24 - hora_llegada + ventana_inicio)

        total_cost += tiempo_viaje

    return total_cost

@njit
def cycle_crossover_cu(padre1, padre2):
    n = len(padre1)
    hijo = np.full(n, -1, dtype=np.int64)
    visitado = np.zeros(n, dtype=np.bool_)

    ciclo = 0
    for inicio in range(n):
        if not visitado[inicio]:
            ciclo += 1
            actual = inicio
            while not visitado[actual]:
                visitado[actual] = True
                hijo[actual] = padre1[actual] if ciclo % 2 else padre2[actual]
                actual = np.where(padre1 == padre2[actual])[0][0]
                if visitado[actual]:
                    break
    return hijo

@njit
def eliminar_de_posicion_cu(ruta, num_ciudades, pos_actual):
    # Elimina la ciudad de la posición actual, desplazando elementos a la izquierda
    for i in range(pos_actual, num_ciudades - 1):
        ruta[i] = ruta[i + 1]
    ruta[num_ciudades - 1] = -1  # Marcamos como vacío

@njit
def insertar_en_posicion_cu(ruta, num_ciudades, ciudad, nueva_pos):
    # Inserta la ciudad en la nueva posición, desplazando elementos a la derecha
    for i in range(num_ciudades - 1, nueva_pos, -1):
        ruta[i] = ruta[i - 1]
    ruta[nueva_pos] = ciudad

@njit
def heuristica_abruptos_cu(hijo, num_ciudades, m, distancias, ventanas_tiempo):
    ruta = hijo.copy()
    ruta_temp = np.zeros_like(ruta)  # Ruta temporal para modificaciones

    for i in range(num_ciudades):
        ciudad_actual = ruta[i]

        # Ordenar las distancias a otras ciudades
        dist_ordenadas = [(distancias[ciudad_actual, j], j) for j in range(num_ciudades)]
        dist_ordenadas.sort()

        # Evaluar la posición actual
        pos_actual = -1
        for j in range(num_ciudades):
            if ruta[j] == ciudad_actual:
                pos_actual = j
                break

        mejor_costo = costo_cu(ruta, distancias, ventanas_tiempo)
        mejor_posicion = pos_actual
        mejor_vecino = -1

        # Evaluar las m ciudades más cercanas
        for j in range(1, min(m + 1, num_ciudades)):
            ciudad_cercana = dist_ordenadas[j][1]

            pos_cercana = -1
            for k in range(num_ciudades):
                if ruta[k] == ciudad_cercana:
                    pos_cercana = k
                    break

            if pos_cercana != -1:
                for posicion_antes_o_despues in range(2):  # Antes o después
                    ruta_temp[:] = ruta
                    eliminar_de_posicion_cu(ruta_temp, num_ciudades, pos_actual)

                    nueva_pos = pos_cercana + posicion_antes_o_despues
                    if nueva_pos > pos_actual:
                        nueva_pos -= 1
                    if nueva_pos >= num_ciudades:
                        nueva_pos = num_ciudades - 1

                    insertar_en_posicion_cu(ruta_temp, num_ciudades, ciudad_actual, nueva_pos)
                    nuevo_costo = costo_cu(ruta_temp, distancias, ventanas_tiempo)

                    if nuevo_costo < mejor_costo:
                        mejor_costo = nuevo_costo
                        mejor_posicion = nueva_pos
                        mejor_vecino = ciudad_cercana

        if mejor_vecino != -1 and mejor_posicion != pos_actual:
            eliminar_de_posicion_cu(ruta, num_ciudades, pos_actual)
            insertar_en_posicion_cu(ruta, num_ciudades, ciudad_actual, mejor_posicion)

    return ruta

@njit
def cruzamiento_cu(num_pob, padres, m, distancias, ventanas_tiempo, pc):
    hijos = np.empty_like(padres)
    aptitudes_hijos = np.zeros(num_pob, dtype=np.float64)
    n = padres.shape[1]  # número de ciudades

    for i in range(0, num_pob, 2):
        if np.random.random() < pc:  # Probabilidad de cruzamiento
            # Realizar cruzamiento
            hijo1 = heuristica_abruptos_cu(
                cycle_crossover_cu(padres[i], padres[i + 1]),
                n,
                m,
                distancias,
                ventanas_tiempo
            )

            hijo2 = heuristica_abruptos_cu(
                cycle_crossover_cu(padres[i + 1], padres[i]),
                n,
                m,
                distancias,
                ventanas_tiempo
            )
        else:
            # No se realiza cruzamiento, los hijos son copias de los padres
            hijo1 = padres[i].copy()
            hijo2 = padres[i + 1].copy()

        # [padre1, padre2, hijo1, hijo2]
        individuos = np.empty((4, n), dtype=np.int64)
        individuos[0, :] = padres[i]
        individuos[1, :] = padres[i + 1]
        individuos[2, :] = hijo1
        individuos[3, :] = hijo2

        aptitudes_individuos = np.array([
            costo_cu(padres[i], distancias, ventanas_tiempo),
            costo_cu(padres[i + 1], distancias, ventanas_tiempo),
            costo_cu(hijo1, distancias, ventanas_tiempo),
            costo_cu(hijo2, distancias, ventanas_tiempo)
        ], dtype=np.float64)

        indices_mejores = np.argsort(aptitudes_individuos)[:2]
        hijos[i] = individuos[indices_mejores[0]]
        hijos[i + 1] = individuos[indices_mejores[1]]
        aptitudes_hijos[i] = aptitudes_individuos[indices_mejores[0]]
        aptitudes_hijos[i + 1] = aptitudes_individuos[indices_mejores[1]]

    return hijos, aptitudes_hijos

@njit
def seleccion_torneo_cu(poblacion, aptitudes, num_competidores):
    n, num_var = poblacion.shape
    padres = np.empty_like(poblacion)
    for i in range(n):
        # Selección de competidores aleatoria
        competidores = np.random.choice(n, num_competidores, replace=False)
        # Evaluación del torneo
        aptitudes_torneo = aptitudes[competidores]
        ganador = competidores[np.argmin(aptitudes_torneo)]
        padres[i] = poblacion[ganador]
    return padres

@njit
def mutacion_cu(poblacion, aptitudes, distancias, ventanas_tiempo, pm):
    n = len(poblacion)
    long_ruta = len(poblacion[0])
    
    for j in range(n):
        if np.random.random() < pm:  # Probabilidad de mutar
            # Seleccionar dos índices aleatorios para intercambiar
            idx1, idx2 = np.random.choice(long_ruta, 2, replace=False)
            # Intercambiar los índices seleccionados
            poblacion[j][idx1], poblacion[j][idx2] = poblacion[j][idx2], poblacion[j][idx1]
            # Recalcular la aptitud del individuo mutado
            aptitudes[j] = costo_cu(poblacion[j], distancias, ventanas_tiempo)
    
    return poblacion, aptitudes

def ejecutar_algoritmo_python_paralelizado(num_pob, num_gen, pm, pc, m, num_competidores, distancias, ventanas_tiempo):
    rng = np.random.default_rng(42)
    num_var = distancias.shape[0]

    # Población inicial
    poblacion = np.array([random_permutation(num_var, rng) for _ in range(num_pob)])
    aptitudes = np.array([costo_cu(ind, distancias, ventanas_tiempo) for ind in poblacion])

    mejor_aptitud_historico = np.min(aptitudes)
    mejor_individuo_historico = poblacion[np.argmin(aptitudes)]

    for _ in range(num_gen):
        # Selección
        padres = seleccion_torneo_cu(poblacion, aptitudes, num_competidores)

        # Cruzamiento_cu (determinístico)
        poblacion, aptitudes = cruzamiento_cu(num_pob, padres, m, distancias, ventanas_tiempo, pc)

        # Mutación: generamos la info aleatoria fuera de njit y luego llamamos a la función determinista
        poblacion, aptitudes = mutacion_cu(poblacion, aptitudes, distancias, ventanas_tiempo, pm)

        # Actualizar mejor historial
        mejor_aptitud_generacion = np.min(aptitudes)
        mejor_individuo_generacion = poblacion[np.argmin(aptitudes)]
        if mejor_aptitud_generacion < mejor_aptitud_historico:
            mejor_aptitud_historico = mejor_aptitud_generacion
            mejor_individuo_historico = mejor_individuo_generacion

    return mejor_individuo_historico, mejor_aptitud_historico

############################################################
# Algoritmo Genetico con Ventanas de Tiempo Python (normal)#
############################################################

def costo(individuo, distancias, ventanas_tiempo):
    total_cost = 0.0
    tiempo_acumulado = 0.0

    num_ciudades = len(individuo)

    for i in range(num_ciudades):
        origen = individuo[i]
        destino = individuo[(i + 1) % num_ciudades]

        tiempo_viaje = distancias[origen][destino]
        tiempo_acumulado += tiempo_viaje

        hora_llegada = tiempo_acumulado % 24
        ventana_inicio, ventana_fin = ventanas_tiempo[destino]

        if ventana_fin < ventana_inicio:
            # Ventana que abarca la medianoche
            if hora_llegada < ventana_inicio and hora_llegada > ventana_fin:
                tiempo_acumulado += (24 - hora_llegada + ventana_inicio)
        else:
            if hora_llegada < ventana_inicio:
                tiempo_acumulado += (ventana_inicio - hora_llegada)
            elif hora_llegada > ventana_fin:
                tiempo_acumulado += (24 - hora_llegada + ventana_inicio)

        total_cost += tiempo_viaje

    return total_cost

def cycle_crossover(padre1, padre2):
    n = len(padre1)
    hijo = [-1] * n
    visitado = [False] * n

    ciclo = 0
    for inicio in range(n):
        if not visitado[inicio]:
            ciclo += 1
            actual = inicio
            while not visitado[actual]:
                visitado[actual] = True
                if ciclo % 2 == 1:
                    hijo[actual] = padre1[actual]
                else:
                    hijo[actual] = padre2[actual]
                try:
                    actual = padre1.index(padre2[actual])
                except ValueError:
                    break  # En caso de que no se encuentre, se rompe el ciclo
    return hijo

def eliminar_de_posicion(ruta, num_ciudades, pos_actual):
    # Elimina la ciudad de la posición actual, desplazando elementos a la izquierda
    for i in range(pos_actual, num_ciudades - 1):
        ruta[i] = ruta[i + 1]
    ruta[num_ciudades - 1] = -1  # Marcamos como vacío

def insertar_en_posicion(ruta, num_ciudades, ciudad, nueva_pos):
    # Inserta la ciudad en la nueva posición, desplazando elementos a la derecha
    for i in range(num_ciudades - 1, nueva_pos, -1):
        ruta[i] = ruta[i - 1]
    ruta[nueva_pos] = ciudad

def heuristica_abruptos(hijo, num_ciudades, m, distancias, ventanas_tiempo):
    ruta = hijo.copy()
    ruta_temp = [0] * len(ruta)  # Ruta temporal para modificaciones

    for i in range(num_ciudades):
        ciudad_actual = ruta[i]

        # Ordenar las distancias a otras ciudades
        dist_ordenadas = sorted([(distancias[ciudad_actual][j], j) for j in range(num_ciudades)], key=lambda x: x[0])

        # Evaluar la posición actual
        try:
            pos_actual = ruta.index(ciudad_actual)
        except ValueError:
            pos_actual = -1

        mejor_costo = costo(ruta, distancias, ventanas_tiempo)
        mejor_posicion = pos_actual
        mejor_vecino = -1

        # Evaluar las m ciudades más cercanas
        for j in range(1, min(m + 1, num_ciudades)):
            ciudad_cercana = dist_ordenadas[j][1]

            try:
                pos_cercana = ruta.index(ciudad_cercana)
            except ValueError:
                pos_cercana = -1

            if pos_cercana != -1:
                for posicion_antes_o_despues in range(2):  # Antes o después
                    ruta_temp = ruta.copy()
                    eliminar_de_posicion(ruta_temp, num_ciudades, pos_actual)

                    nueva_pos = pos_cercana + posicion_antes_o_despues
                    if nueva_pos > pos_actual:
                        nueva_pos -= 1
                    if nueva_pos >= num_ciudades:
                        nueva_pos = num_ciudades - 1

                    insertar_en_posicion(ruta_temp, num_ciudades, ciudad_actual, nueva_pos)
                    nuevo_costo = costo(ruta_temp, distancias, ventanas_tiempo)

                    if nuevo_costo < mejor_costo:
                        mejor_costo = nuevo_costo
                        mejor_posicion = nueva_pos
                        mejor_vecino = ciudad_cercana

        if mejor_vecino != -1 and mejor_posicion != pos_actual:
            eliminar_de_posicion(ruta, num_ciudades, pos_actual)
            insertar_en_posicion(ruta, num_ciudades, ciudad_actual, mejor_posicion)

    return ruta

def cruzamiento(num_pob, padres, m, distancias, ventanas_tiempo, pc):
    hijos = [None] * num_pob
    aptitudes_hijos = [0.0] * num_pob
    n = len(padres[0])  # número de ciudades

    for i in range(0, num_pob, 2):
        if random.random() < pc:  # Probabilidad de cruzamiento
            # Realizar cruzamiento
            hijo1 = heuristica_abruptos(
                cycle_crossover(padres[i], padres[i + 1]),
                n,
                m,
                distancias,
                ventanas_tiempo
            )

            hijo2 = heuristica_abruptos(
                cycle_crossover(padres[i + 1], padres[i]),
                n,
                m,
                distancias,
                ventanas_tiempo
            )
        else:
            # No se realiza cruzamiento, los hijos son copias de los padres
            hijo1 = padres[i].copy()
            hijo2 = padres[i + 1].copy()

        # [padre1, padre2, hijo1, hijo2]
        individuos = [padres[i], padres[i + 1], hijo1, hijo2]

        aptitudes_individuos = [
            costo(padres[i], distancias, ventanas_tiempo),
            costo(padres[i + 1], distancias, ventanas_tiempo),
            costo(hijo1, distancias, ventanas_tiempo),
            costo(hijo2, distancias, ventanas_tiempo)
        ]

        # Obtener índices de los dos mejores individuos
        sorted_indices = sorted(range(4), key=lambda x: aptitudes_individuos[x])
        indices_mejores = sorted_indices[:2]

        hijos[i] = individuos[indices_mejores[0]]
        hijos[i + 1] = individuos[indices_mejores[1]]
        aptitudes_hijos[i] = aptitudes_individuos[indices_mejores[0]]
        aptitudes_hijos[i + 1] = aptitudes_individuos[indices_mejores[1]]

    return hijos, aptitudes_hijos

def seleccion_torneo(poblacion, aptitudes, num_competidores):
    n = len(poblacion)
    padres = [None] * n
    for i in range(n):
        # Selección de competidores aleatoria
        competidores = random.sample(range(n), num_competidores)
        # Evaluación del torneo
        aptitudes_torneo = [aptitudes[j] for j in competidores]
        ganador = competidores[aptitudes_torneo.index(min(aptitudes_torneo))]
        padres[i] = poblacion[ganador]
    return padres

def mutacion(poblacion, aptitudes, distancias, ventanas_tiempo, pm):
    n = len(poblacion)
    long_ruta = len(poblacion[0])

    for j in range(n):
        if random.random() < pm:  # Probabilidad de mutar
            # Seleccionar dos índices aleatorios para intercambiar
            idx1, idx2 = random.sample(range(long_ruta), 2)
            # Intercambiar los índices seleccionados
            poblacion[j][idx1], poblacion[j][idx2] = poblacion[j][idx2], poblacion[j][idx1]
            # Recalcular la aptitud del individuo mutado
            aptitudes[j] = costo(poblacion[j], distancias, ventanas_tiempo)

    return poblacion, aptitudes

def ejecutar_algoritmo_python(num_pob, num_gen, pm, pc, m, num_competidores, distancias, ventanas_tiempo):
    random.seed(42)
    num_var = len(distancias)

    # Población inicial
    poblacion = [random_permutation_normal(num_var) for _ in range(num_pob)]
    aptitudes = [costo(ind, distancias, ventanas_tiempo) for ind in poblacion]

    mejor_aptitud_historico = min(aptitudes)
    mejor_individuo_historico = poblacion[aptitudes.index(mejor_aptitud_historico)]

    for _ in range(num_gen):
        # Selección
        padres = seleccion_torneo(poblacion, aptitudes, num_competidores)

        # Cruzamiento (determinístico)
        poblacion, aptitudes = cruzamiento(num_pob, padres, m, distancias, ventanas_tiempo, pc)

        # Mutación
        poblacion, aptitudes = mutacion(poblacion, aptitudes, distancias, ventanas_tiempo, pm)

        # Actualizar mejor historial
        mejor_aptitud_generacion = min(aptitudes)
        mejor_individuo_generacion = poblacion[aptitudes.index(mejor_aptitud_generacion)]
        if mejor_aptitud_generacion < mejor_aptitud_historico:
            mejor_aptitud_historico = mejor_aptitud_generacion
            mejor_individuo_historico = mejor_individuo_generacion

    return mejor_individuo_historico, mejor_aptitud_historico

############################################################
#                   Funciones en Común                     #
############################################################
def random_permutation(n, rng):
    arr = np.arange(n, dtype=np.int64)
    for i in range(n - 1, 0, -1):
        j = rng.integers(0, i + 1)
        arr[i], arr[j] = arr[j], arr[i]
    return arr

def random_permutation_normal(n):
    arr = list(range(n))
    random.shuffle(arr)
    return arr

############################################################
# Algoritmo Genetico con Ventanas de Tiempo en C Python    #
############################################################

# Definimos una estructura que mapea la estructura `ResultadoGenetico` en C
class ResultadoGenetico(Structure):
    _fields_ = [
        ("recorrido", POINTER(c_int)),         # Puntero al arreglo de la mejor ruta
        ("fitness", c_double),                # Fitness del mejor individuo
        ("tiempo_ejecucion", c_double),       # Tiempo de ejecución del algoritmo
        ("nombres_ciudades", POINTER(c_char * 50 * 32)),  # Puntero a los nombres de las ciudades
        ("longitud_recorrido", c_int)         # Longitud de la ruta
    ]

# Clase para la biblioteca compartida del algoritmo genético
class AlgoritmoGeneticoNormal:
    def __init__(self, ruta_biblioteca_normal):
        # Cargamos la biblioteca compartida desde la ruta proporcionada
        self.biblioteca = ctypes.CDLL(ruta_biblioteca_normal)
        
        # Configuramos el tipo de retorno de la función `ejecutar_algoritmo_genetico_ventanas_tiempo`
        self.biblioteca.ejecutar_algoritmo_genetico_ventanas_tiempo.restype = POINTER(ResultadoGenetico)
        
        # Especificamos los tipos de argumentos que espera `ejecutar_algoritmo_genetico_ventanas_tiempo`
        self.biblioteca.ejecutar_algoritmo_genetico_ventanas_tiempo.argtypes = [
            c_int,      # tamano_poblacion
            c_int,      # longitud_genotipo
            c_int,      # num_generaciones
            c_int,      # num_competidores
            c_int,      # m parametro de heurística
            c_double,   # probabilidad_mutacion
            c_double,   # probabilidad_cruce
            c_char_p,    # nombre_archivo (ruta al archivo con matriz de distancias)
            c_int       # km_hr
        ]
        
        # Configuramos los argumentos de la función para liberar resultados
        self.biblioteca.liberar_resultado.argtypes = [POINTER(ResultadoGenetico)]

    def ejecutar(self, tamano_poblacion, longitud_genotipo, num_generaciones,
                 num_competidores, m, probabilidad_mutacion, 
                 probabilidad_cruce, nombre_archivo, km_hr):
        try:
            # Convertimos el nombre del archivo a una cadena de bytes
            nombre_archivo_bytes = nombre_archivo.encode('utf-8')
            
            # Llamamos a la función `ejecutar_algoritmo_genetico_ventanas_tiempo` de la biblioteca C
            resultado = self.biblioteca.ejecutar_algoritmo_genetico_ventanas_tiempo(
                tamano_poblacion,
                longitud_genotipo,
                num_generaciones,
                num_competidores,
                m,
                probabilidad_mutacion,
                probabilidad_cruce,
                nombre_archivo_bytes,
                km_hr
            )
            
            # Verificamos si la función devolvió un resultado válido
            if not resultado:
                raise RuntimeError("Error al ejecutar el algoritmo genético")
            
            # Convertimos el recorrido (índices de las ciudades) a una lista de Python
            recorrido = [resultado.contents.recorrido[i] for i in range(resultado.contents.longitud_recorrido)]
            
            # Convertimos los nombres de las ciudades a una lista de Python
            nombres_ciudades = []
            for i in range(resultado.contents.longitud_recorrido):
                # Cada ciudad es un array de caracteres en C que convertimos a cadena de Python
                nombre_ciudad = bytes(resultado.contents.nombres_ciudades.contents[i]).decode('utf-8')
                nombre_ciudad = nombre_ciudad.split('\0')[0]  # Eliminamos los caracteres nulos
                nombres_ciudades.append(nombre_ciudad)
            
            # Creamos un diccionario con los resultados
            salida = {
                'recorrido': recorrido,                 # Ruta como lista de índices
                'nombres_ciudades': nombres_ciudades,   # Lista de nombres de las ciudades
                'fitness': resultado.contents.fitness,  # Fitness del mejor individuo
                'tiempo_ejecucion': resultado.contents.tiempo_ejecucion  # Tiempo de ejecución
            }
            
            # Liberamos la memoria reservada por la biblioteca C
            self.biblioteca.liberar_resultado(resultado)
            
            return salida  # Devolvemos los resultados como un diccionario
            
        except Exception as e:
            raise RuntimeError(f"Error al ejecutar el algoritmo genético: {str(e)}")
        
#########################################################################
# Algoritmo Genetico con Ventanas de Tiempo paralelizado en C Python    #
#########################################################################
# Clase para la biblioteca compartida del algoritmo genético
class AlgoritmoGeneticoParalelizado:
    def __init__(self, ruta_biblioteca_normal):
        # Cargamos la biblioteca compartida desde la ruta proporcionada
        self.biblioteca = ctypes.CDLL(ruta_biblioteca_normal)
        
        # Configuramos el tipo de retorno de la función `ejecutar_algoritmo_genetico_ventanas_tiempo`
        self.biblioteca.ejecutar_algoritmo_genetico_ventanas_tiempo_paralelizado.restype = POINTER(ResultadoGenetico)
        
        # Especificamos los tipos de argumentos que espera `ejecutar_algoritmo_genetico_ventanas_tiempo`
        self.biblioteca.ejecutar_algoritmo_genetico_ventanas_tiempo_paralelizado.argtypes = [
            c_int,      # tamano_poblacion
            c_int,      # longitud_genotipo
            c_int,      # num_generaciones
            c_int,      # num_competidores
            c_int,      # m parametro de heurística
            c_double,   # probabilidad_mutacion
            c_double,   # probabilidad_cruce
            c_char_p,    # nombre_archivo (ruta al archivo con matriz de distancias)
            c_int       # km_hr
        ]
        
        # Configuramos los argumentos de la función para liberar resultados
        self.biblioteca.liberar_resultado.argtypes = [POINTER(ResultadoGenetico)]

    def ejecutar(self, tamano_poblacion, longitud_genotipo, num_generaciones,
                 num_competidores, m, probabilidad_mutacion, 
                 probabilidad_cruce, nombre_archivo, km_hr):
        try:
            # Convertimos el nombre del archivo a una cadena de bytes
            nombre_archivo_bytes = nombre_archivo.encode('utf-8')
            
            # Llamamos a la función `ejecutar_algoritmo_genetico_ventanas_tiempo_paralelizado` de la biblioteca C
            resultado = self.biblioteca.ejecutar_algoritmo_genetico_ventanas_tiempo_paralelizado(
                tamano_poblacion,
                longitud_genotipo,
                num_generaciones,
                num_competidores,
                m,
                probabilidad_mutacion,
                probabilidad_cruce,
                nombre_archivo_bytes,
                km_hr
            )
            
            # Verificamos si la función devolvió un resultado válido
            if not resultado:
                raise RuntimeError("Error al ejecutar el algoritmo genético")
            
            # Convertimos el recorrido (índices de las ciudades) a una lista de Python
            recorrido = [resultado.contents.recorrido[i] for i in range(resultado.contents.longitud_recorrido)]
            
            # Convertimos los nombres de las ciudades a una lista de Python
            nombres_ciudades = []
            for i in range(resultado.contents.longitud_recorrido):
                # Cada ciudad es un array de caracteres en C que convertimos a cadena de Python
                nombre_ciudad = bytes(resultado.contents.nombres_ciudades.contents[i]).decode('utf-8')
                nombre_ciudad = nombre_ciudad.split('\0')[0]  # Eliminamos los caracteres nulos
                nombres_ciudades.append(nombre_ciudad)
            
            # Creamos un diccionario con los resultados
            salida = {
                'recorrido': recorrido,                 # Ruta como lista de índices
                'nombres_ciudades': nombres_ciudades,   # Lista de nombres de las ciudades
                'fitness': resultado.contents.fitness,  # Fitness del mejor individuo
                'tiempo_ejecucion': resultado.contents.tiempo_ejecucion  # Tiempo de ejecución
            }
            
            # Liberamos la memoria reservada por la biblioteca C
            self.biblioteca.liberar_resultado(resultado)
            
            return salida  # Devolvemos los resultados como un diccionario
            
        except Exception as e:
            raise RuntimeError(f"Error al ejecutar el algoritmo genético: {str(e)}")


def main():

    # Parámetros constantes
    tamano_poblacion = 100
    longitud_genotipo = 32
    num_generaciones = 100
    num_competidores = 2
    m = 3
    probabilidad_mutacion = 0.3
    probabilidad_cruce = 0.9
    nombre_archivo = "Distancias_no_head.csv"
    km_hr = 80

    # Carga de datos
    try:
        distancias_array = np.loadtxt(nombre_archivo, delimiter=',')
    except Exception as e:
        print(f"Error al cargar los datos: {e}")
        return

    # Convertir distancias a tiempo
    distancias_array = distancias_array / km_hr

    # Carga de datos para el algoritmo Python (Normal)
    try:
        with open(nombre_archivo, 'r') as f:
            distancias = [list(map(float, line.strip().split(','))) for line in f]
    except Exception as e:
        print(f"Error al cargar los datos: {e}")
        return

    distancias = [[d / km_hr for d in row] for row in distancias]

    nombres_ciudades = [
        "Aguascalientes", "Baja California", "Baja California Sur",
        "Campeche", "Chiapas", "Chihuahua", "Coahuila", "Colima", "Durango",
        "Guanajuato", "Guerrero", "Hidalgo", "Jalisco", "Estado de México",
        "Michoacán", "Morelos", "Nayarit", "Nuevo León", "Oaxaca", "Puebla",
        "Querétaro", "Quintana Roo", "San Luis Potosí", "Sinaloa", "Sonora",
        "Tabasco", "Tamaulipas", "Tlaxcala", "Veracruz", "Yucatán",
        "Zacatecas", "CDMX"
    ]

    ventanas_tiempo_array = np.array([
        [8, 12], [9, 13], [10, 14], [11, 15], [12, 16], [13, 17], [14, 18],
        [15, 19], [16, 20], [17, 21], [18, 22], [19, 23], [20, 0], [21, 1],
        [22, 2], [23, 3], [0, 4], [1, 5], [2, 6], [3, 7], [4, 8], [5, 9],
        [6, 10], [7, 11], [8, 12], [9, 13], [10, 14], [11, 15], [12, 16],
        [13, 17], [14, 18], [15, 19]
    ])

    ventanas_tiempo = [
        [8, 12], [9, 13], [10, 14], [11, 15], [12, 16], [13, 17], [14, 18],
        [15, 19], [16, 20], [17, 21], [18, 22], [19, 23], [20, 0], [21, 1],
        [22, 2], [23, 3], [0, 4], [1, 5], [2, 6], [3, 7], [4, 8], [5, 9],
        [6, 10], [7, 11], [8, 12], [9, 13], [10, 14], [11, 15], [12, 16],
        [13, 17], [14, 18], [15, 19]
    ]
    

    # Inicializar diccionario para almacenar tiempos
    tiempos_ejecucion = {}

    # 1. Ejecutar Algoritmo Python Paralelizado (Numba)
    print("Ejecutando Algoritmo Python Paralelizado (Numba)...")
    start_time = time.time()
    mejor_individuo_python_cuda, aptitud_python_cuda = ejecutar_algoritmo_python_paralelizado(
        tamano_poblacion, num_generaciones, probabilidad_mutacion, probabilidad_cruce, 
        m, num_competidores, distancias_array, ventanas_tiempo_array
    )
    end_time = time.time()
    tiempo_python_cuda = end_time - start_time
    tiempos_ejecucion['Python Paralelizado (Numba)'] = tiempo_python_cuda
    print(f"Tiempo de ejecución: {tiempo_python_cuda:.4f} segundos\n")

    # 2. Ejecutar Algoritmo Python (Normal)
    print("Ejecutando Algoritmo Python (Normal)...")
    start_time = time.time()
    mejor_individuo_python, aptitud_python = ejecutar_algoritmo_python(
        tamano_poblacion, num_generaciones, probabilidad_mutacion, probabilidad_cruce, 
        m, num_competidores, distancias, ventanas_tiempo
    )
    end_time = time.time()
    tiempo_python = end_time - start_time
    tiempos_ejecucion['Python (Normal)'] = tiempo_python
    print(f"Tiempo de ejecución: {tiempo_python:.4f} segundos\n")

    # Obtener la ruta absoluta del archivo actual
    directorio_actual = os.path.dirname(os.path.abspath(__file__))

    # 3. Ejecutar Algoritmo Genético en C (Normal)
    print("Ejecutando Algoritmo Genético en C (Normal)...")
    nombre_biblioteca_normal = "genetic_algo_vent.dll" if os.name == 'nt' else "libgenetic_algo_vent.so"
    ruta_biblioteca_normal = os.path.join(directorio_actual, nombre_biblioteca_normal)

    # Verificamos que el archivo de la biblioteca existe en la ruta especificada
    if not os.path.exists(ruta_biblioteca_normal):
        raise RuntimeError(f"No se encuentra la biblioteca en {ruta_biblioteca_normal}")

    # Crear instancia del wrapper para la biblioteca
    ag_normal = AlgoritmoGeneticoNormal(ruta_biblioteca_normal)

    start_time = time.time()
    resultado_normal = ag_normal.ejecutar(
        tamano_poblacion=tamano_poblacion,
        longitud_genotipo=longitud_genotipo,
        num_generaciones=num_generaciones,
        num_competidores=num_competidores,
        m=m,
        probabilidad_mutacion=probabilidad_mutacion,
        probabilidad_cruce=probabilidad_cruce,
        nombre_archivo=nombre_archivo,
        km_hr=km_hr
    )
    end_time = time.time()
    tiempo_c_normal = resultado_normal['tiempo_ejecucion']  # Asumiendo que este valor es preciso
    tiempos_ejecucion['C (Normal)'] = tiempo_c_normal
    print(f"Tiempo de ejecución: {tiempo_c_normal:.4f} segundos\n")

    # 4. Ejecutar Algoritmo Genético en C (Paralelizado)
    print("Ejecutando Algoritmo Genético en C (Paralelizado)...")
    nombre_biblioteca_paralell = "genetic_algo_vent_paralell.dll" if os.name == 'nt' else "libgenetic_algo_vent_paralell.so"
    ruta_biblioteca_paralell = os.path.join(directorio_actual, nombre_biblioteca_paralell)

    # Verificamos que el archivo de la biblioteca existe en la ruta especificada
    if not os.path.exists(ruta_biblioteca_paralell):
        raise RuntimeError(f"No se encuentra la biblioteca en {ruta_biblioteca_paralell}")

    # Crear instancia del wrapper para la biblioteca paralelizada
    ag_paralell = AlgoritmoGeneticoParalelizado(ruta_biblioteca_paralell)

    start_time = time.time()
    resultado_paralelizado = ag_paralell.ejecutar(
        tamano_poblacion=tamano_poblacion,
        longitud_genotipo=longitud_genotipo,
        num_generaciones=num_generaciones,
        num_competidores=num_competidores,
        m=m,
        probabilidad_mutacion=probabilidad_mutacion,
        probabilidad_cruce=probabilidad_cruce,
        nombre_archivo=nombre_archivo,
        km_hr=km_hr
    )
    end_time = time.time()
    tiempo_c_paralel = resultado_paralelizado['tiempo_ejecucion']  # Asumiendo que este valor es preciso
    tiempos_ejecucion['C (Paralelizado)'] = tiempo_c_paralel
    print(f"Tiempo de ejecución: {tiempo_c_paralel:.4f} segundos\n")

    # Imprimir resumen de resultados
    print("Resumen de Tiempos de Ejecución:")
    for algoritmo, tiempo in tiempos_ejecucion.items():
        print(f"{algoritmo}: {tiempo:.4f} segundos")

    # Graficar los resultados
    algoritmos = list(tiempos_ejecucion.keys())
    tiempos = list(tiempos_ejecucion.values())

    plt.figure(figsize=(10, 6))
    bars = plt.bar(algoritmos, tiempos, color=['blue', 'green', 'red', 'purple'])
    plt.xlabel('Algoritmos')
    plt.ylabel('Tiempo de Ejecución (segundos)')
    plt.title('Comparación de Tiempos de Ejecución de Algoritmos Genéticos')
    plt.ylim(0, max(tiempos)*1.1)  # Añadir un 10% de margen arriba

    # Añadir etiquetas de tiempo encima de cada barra
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + max(tiempos)*0.01, f'{yval:.2f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()