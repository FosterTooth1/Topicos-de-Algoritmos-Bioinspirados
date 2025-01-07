import random
import time

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
    poblacion = [random_permutation(num_var) for _ in range(num_pob)]
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

def random_permutation(n):
    arr = list(range(n))
    random.shuffle(arr)
    return arr

def main():
    tiempo_algoritmo = time.time()
    # Parámetros
    num_pob, num_gen, pm, pc, m, num_competidores = 100, 100, 0.15, 0.9, 3, 2
    km_hr = 80

    # Carga de datos
    try:
        with open('Distancias_no_head.csv', 'r') as f:
            distancias = [list(map(float, line.strip().split(','))) for line in f]
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Convertir distancias a tiempo
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

    ventanas_tiempo = [
        [8, 12], [9, 13], [10, 14], [11, 15], [12, 16], [13, 17], [14, 18],
        [15, 19], [16, 20], [17, 21], [18, 22], [19, 23], [20, 0], [21, 1],
        [22, 2], [23, 3], [0, 4], [1, 5], [2, 6], [3, 7], [4, 8], [5, 9],
        [6, 10], [7, 11], [8, 12], [9, 13], [10, 14], [11, 15], [12, 16],
        [13, 17], [14, 18], [15, 19]
    ]

    mejor_individuo, aptitud = ejecutar_algoritmo_python(
        num_pob, num_gen, pm, pc, m, num_competidores, distancias, ventanas_tiempo
    )

    print(f"Mejor individuo: {mejor_individuo}")
    print(f"Aptitud: {aptitud}")
    print(f"Ruta: {[nombres_ciudades[i] for i in mejor_individuo]}")

    tiempo_algoritmo_total = time.time() - tiempo_algoritmo
    print(f"Tiempo total del algoritmo: {tiempo_algoritmo_total:.2f} segundos")

if __name__ == "__main__":
    main()
