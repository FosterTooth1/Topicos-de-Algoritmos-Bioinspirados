import numpy as np
import time
import pandas as pd
from numba import njit
import time

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

def random_permutation(n, rng):
    arr = np.arange(n, dtype=np.int64)
    for i in range(n - 1, 0, -1):
        j = rng.integers(0, i + 1)
        arr[i], arr[j] = arr[j], arr[i]
    return arr

def main():

    tiempo_algoritmo = time.time()
    # Parámetros
    num_pob, num_gen, pm, pc, m, num_competidores = 100, 100, 0.15, 0.9, 3, 2
    km_hr= 80
    
    # Carga de datos
    try:
        distancias = np.loadtxt('Distancias_no_head.csv', delimiter=',')
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Convertir distancias a tiempo
    distancias = distancias / km_hr
    
    nombres_ciudades = ["Aguascalientes", "Baja California", "Baja California Sur",
        "Campeche", "Chiapas", "Chihuahua", "Coahuila", "Colima", "Durango",
        "Guanajuato", "Guerrero", "Hidalgo", "Jalisco", "Estado de México",
        "Michoacán", "Morelos", "Nayarit", "Nuevo León", "Oaxaca", "Puebla",
        "Querétaro", "Quintana Roo", "San Luis Potosí", "Sinaloa", "Sonora",
        "Tabasco", "Tamaulipas", "Tlaxcala", "Veracruz", "Yucatán",
        "Zacatecas", "CDMX"]
    
    ventanas_tiempo = np.array([
        [8, 12], [9, 13], [10, 14], [11, 15], [12, 16], [13, 17], [14, 18],
        [15, 19], [16, 20], [17, 21], [18, 22], [19, 23], [20, 0], [21, 1],
        [22, 2], [23, 3], [0, 4], [1, 5], [2, 6], [3, 7], [4, 8], [5, 9],
        [6, 10], [7, 11], [8, 12], [9, 13], [10, 14], [11, 15], [12, 16],
        [13, 17], [14, 18], [15, 19]
    ])
    
    mejor_individuo, aptitud = ejecutar_algoritmo_python_paralelizado(num_pob, num_gen, pm, pc, m, num_competidores, distancias, ventanas_tiempo)
    
    print(f"Mejor individuo: {mejor_individuo}")
    print(f"Aptitud: {aptitud}")
    print(f"Ruta: {[nombres_ciudades[i] for i in mejor_individuo]}")
    
    tiempo_algoritmo_total = time.time() - tiempo_algoritmo
    print(tiempo_algoritmo_total)

if __name__ == "__main__":
    main()