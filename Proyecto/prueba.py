import numpy as np
import time
import pandas as pd
from numba import cuda, float64, int64, boolean
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
import random
import matplotlib.pyplot as plt

# Configuración de la semilla para reproducibilidad
np.random.seed(42)

############################################################
# Algoritmo Genético con Ventanas de Tiempo en CUDA Python #
############################################################

# Kernel para calcular el costo de cada individuo en la GPU
@cuda.jit
def costo_kernel(population, distancias, ventanas_tiempo, costos, num_ciudades):
    idx = cuda.grid(1)
    if idx < population.shape[0]:
        individuo = population[idx]
        total_cost = 0.0
        tiempo_acumulado = 0.0

        for i in range(num_ciudades):
            origen = individuo[i]
            destino = individuo[(i + 1) % num_ciudades]

            tiempo_viaje = distancias[origen, destino]
            tiempo_acumulado += tiempo_viaje

            hora_llegada = tiempo_acumulado % 24
            ventana_inicio = ventanas_tiempo[destino, 0]
            ventana_fin = ventanas_tiempo[destino, 1]

            if hora_llegada < ventana_inicio:
                tiempo_acumulado += (ventana_inicio - hora_llegada)
            elif ventana_fin < ventana_inicio:
                if hora_llegada > ventana_fin and hora_llegada < ventana_inicio:
                    tiempo_acumulado += (24 - hora_llegada + ventana_inicio)
            elif hora_llegada > ventana_fin:
                tiempo_acumulado += (24 - hora_llegada + ventana_inicio)

            total_cost += tiempo_viaje

        costos[idx] = total_cost

# Kernel para realizar la selección de torneo en la GPU
@cuda.jit
def seleccion_torneo_kernel(population, aptitudes, padres, num_pob, num_competidores, rng_states):
    idx = cuda.grid(1)
    if idx < num_pob:
        state = rng_states[idx]
        ganador = -1
        mejor_aptitud = 1e20  # Un valor muy grande
        for _ in range(num_competidores):
            competidor = int(xoroshiro128p_uniform_float32(state) * num_pob)
            if competidor >= num_pob:
                competidor = num_pob - 1  # Asegurar que el índice esté dentro del rango
            if aptitudes[competidor] < mejor_aptitud:
                mejor_aptitud = aptitudes[competidor]
                ganador = competidor
        padres[idx] = population[ganador]

# Kernel para decidir si se realiza el cruzamiento basado en la probabilidad pc
@cuda.jit
def crossover_decision_kernel(padres, crossover_flags, pc, num_pob, rng_states):
    idx = cuda.grid(1)
    if idx < num_pob:
        state = rng_states[idx]
        rand_val = xoroshiro128p_uniform_float32(state)
        if rand_val < pc and (idx % 2) == 0 and (idx + 1) < num_pob:
            crossover_flags[idx] = 1  # Indica que se realizará el cruzamiento para este par
        else:
            crossover_flags[idx] = 0  # No se realiza cruzamiento

# Kernel para realizar el cruzamiento de ciclo en la GPU
@cuda.jit
def cruzamiento_ciclo_kernel(padres, hijos, crossover_flags, num_pob, num_ciudades, rng_states):
    idx = cuda.grid(1)
    pair_idx = idx // 2
    if (pair_idx * 2 + 1) < num_pob and idx < (num_pob // 2):
        if crossover_flags[pair_idx * 2] == 1:
            padre1 = padres[pair_idx * 2]
            padre2 = padres[pair_idx * 2 + 1]
            
            # Inicializar los hijos con -1
            for i in range(num_ciudades):
                hijos[pair_idx * 2, i] = -1
                hijos[pair_idx * 2 + 1, i] = -1

            # Cycle Crossover para hijo1
            start = 0
            current = padre1[start]
            while True:
                hijos[pair_idx * 2, start] = padre1[start]
                current = padre2[start]
                # Encontrar el siguiente índice donde padre1 tiene el valor current
                next_idx = -1
                for k in range(num_ciudades):
                    if padre1[k] == current:
                        next_idx = k
                        break
                if next_idx == -1 or next_idx == start:
                    break
                start = next_idx

            # Rellenar los genes restantes de hijo1 con genes de padre2
            for i in range(num_ciudades):
                if hijos[pair_idx * 2, i] == -1:
                    hijos[pair_idx * 2, i] = padre2[i]

            # Cycle Crossover para hijo2
            start = 0
            current = padre2[start]
            while True:
                hijos[pair_idx * 2 + 1, start] = padre2[start]
                current = padre1[start]
                # Encontrar el siguiente índice donde padre2 tiene el valor current
                next_idx = -1
                for k in range(num_ciudades):
                    if padre2[k] == current:
                        next_idx = k
                        break
                if next_idx == -1 or next_idx == start:
                    break
                start = next_idx

            # Rellenar los genes restantes de hijo2 con genes de padre1
            for i in range(num_ciudades):
                if hijos[pair_idx * 2 + 1, i] == -1:
                    hijos[pair_idx * 2 + 1, i] = padre1[i]
        else:
            # Si no se realiza cruzamiento, los hijos son copias de los padres
            for i in range(num_ciudades):
                hijos[pair_idx * 2, i] = padres[pair_idx * 2, i]
                hijos[pair_idx * 2 + 1, i] = padres[pair_idx * 2 + 1, i]

# Kernel para aplicar la heurística de abruptos en la GPU
@cuda.jit
def heuristica_abruptos_kernel(hijos, num_pob, num_ciudades, m, distancias, ventanas_tiempo, costos):
    idx = cuda.grid(1)
    if idx < num_pob:
        # Crear una copia local de la ruta
        ruta = cuda.local.array(50, dtype=int64)  # Ajusta el tamaño si num_ciudades > 50
        for i in range(num_ciudades):
            ruta[i] = hijos[idx, i]

        mejor_costo = costos[idx]

        for i in range(num_ciudades):
            ciudad_actual = ruta[i]

            # Ordenar las distancias a otras ciudades usando Bubble Sort
            dist_ordenadas = cuda.local.array(50, dtype=float64)
            ciudades_ordenadas = cuda.local.array(50, dtype=int64)
            for j in range(num_ciudades):
                dist_ordenadas[j] = distancias[ciudad_actual, j]
                ciudades_ordenadas[j] = j

            # Bubble Sort
            for j in range(num_ciudades - 1):
                for k in range(num_ciudades - j - 1):
                    if dist_ordenadas[k] > dist_ordenadas[k + 1]:
                        # Swap distances
                        temp_dist = dist_ordenadas[k]
                        dist_ordenadas[k] = dist_ordenadas[k + 1]
                        dist_ordenadas[k + 1] = temp_dist
                        # Swap corresponding cities
                        temp_city = ciudades_ordenadas[k]
                        ciudades_ordenadas[k] = ciudades_ordenadas[k + 1]
                        ciudades_ordenadas[k + 1] = temp_city

            # Evaluar las m ciudades más cercanas
            for j in range(1, min(m + 1, num_ciudades)):
                ciudad_cercana = ciudades_ordenadas[j]

                # Encontrar la posición de la ciudad cercana
                pos_cercana = -1
                for k in range(num_ciudades):
                    if ruta[k] == ciudad_cercana:
                        pos_cercana = k
                        break

                if pos_cercana != -1:
                    for posicion_antes_o_despues in range(2):  # Antes o después
                        ruta_temp = cuda.local.array(50, dtype=int64)
                        for l in range(num_ciudades):
                            ruta_temp[l] = ruta[l]

                        # Eliminar la ciudad de la posición actual
                        for l in range(i, num_ciudades - 1):
                            ruta_temp[l] = ruta_temp[l + 1]
                        ruta_temp[num_ciudades - 1] = -1  # Marcamos como vacío

                        # Insertar en nueva posición
                        nueva_pos = pos_cercana + posicion_antes_o_despues
                        if nueva_pos > i:
                            nueva_pos -= 1
                        if nueva_pos >= num_ciudades:
                            nueva_pos = num_ciudades - 1

                        # Desplazar hacia la derecha para insertar
                        for l in range(num_ciudades - 1, nueva_pos, -1):
                            ruta_temp[l] = ruta_temp[l - 1]
                        ruta_temp[nueva_pos] = ciudad_actual

                        # Recalcular el costo
                        total_cost = 0.0
                        tiempo_acumulado = 0.0
                        for p in range(num_ciudades):
                            origen = ruta_temp[p]
                            destino = ruta_temp[(p + 1) % num_ciudades]
                            tiempo_viaje = distancias[origen, destino]
                            tiempo_acumulado += tiempo_viaje
                            hora_llegada = tiempo_acumulado % 24
                            ventana_inicio = ventanas_tiempo[destino, 0]
                            ventana_fin = ventanas_tiempo[destino, 1]
                            if hora_llegada < ventana_inicio:
                                tiempo_acumulado += (ventana_inicio - hora_llegada)
                            elif ventana_fin < ventana_inicio:
                                if hora_llegada > ventana_fin and hora_llegada < ventana_inicio:
                                    tiempo_acumulado += (24 - hora_llegada + ventana_inicio)
                            elif hora_llegada > ventana_fin:
                                tiempo_acumulado += (24 - hora_llegada + ventana_inicio)
                            total_cost += tiempo_viaje

                        if total_cost < mejor_costo:
                            mejor_costo = total_cost
                            # Actualizar la ruta
                            for p in range(num_ciudades):
                                ruta[p] = ruta_temp[p]
                            costos[idx] = mejor_costo

        # Actualizar la ruta del hijo si se encontró una mejor
        for i in range(num_ciudades):
            hijos[idx, i] = ruta[i]

# Kernel para realizar la mutación en la GPU
@cuda.jit
def mutacion_kernel(population, aptitudes, distancias, ventanas_tiempo, pm, num_ciudades, rng_states):
    idx = cuda.grid(1)
    if idx < population.shape[0]:
        state = rng_states[idx]
        rand_val = xoroshiro128p_uniform_float32(state)
        if rand_val < pm:
            # Seleccionar dos índices aleatorios para intercambiar
            idx1 = int(xoroshiro128p_uniform_float32(state) * num_ciudades)
            idx2 = int(xoroshiro128p_uniform_float32(state) * num_ciudades)
            if idx1 >= num_ciudades:
                idx1 = num_ciudades - 1
            if idx2 >= num_ciudades:
                idx2 = num_ciudades - 1
            # Asegurarse de que idx1 y idx2 sean diferentes
            if idx1 != idx2:
                # Intercambiar las ciudades
                temp = population[idx, idx1]
                population[idx, idx1] = population[idx, idx2]
                population[idx, idx2] = temp
                # Recalcular la aptitud
                total_cost = 0.0
                tiempo_acumulado = 0.0
                for i in range(num_ciudades):
                    origen = population[idx, i]
                    destino = population[idx, (i + 1) % num_ciudades]
                    tiempo_viaje = distancias[origen, destino]
                    tiempo_acumulado += tiempo_viaje
                    hora_llegada = tiempo_acumulado % 24
                    ventana_inicio = ventanas_tiempo[destino, 0]
                    ventana_fin = ventanas_tiempo[destino, 1]
                    if hora_llegada < ventana_inicio:
                        tiempo_acumulado += (ventana_inicio - hora_llegada)
                    elif ventana_fin < ventana_inicio:
                        if hora_llegada > ventana_fin and hora_llegada < ventana_inicio:
                            tiempo_acumulado += (24 - hora_llegada + ventana_inicio)
                    elif hora_llegada > ventana_fin:
                        tiempo_acumulado += (24 - hora_llegada + ventana_inicio)
                    total_cost += tiempo_viaje
                aptitudes[idx] = total_cost

# Función principal para ejecutar el algoritmo genético en GPU
def ejecutar_algoritmo_cuda(num_pob, num_gen, pm, pc, m, num_competidores, distancias, ventanas_tiempo):
    num_ciudades = distancias.shape[0]

    # Transferir datos a la GPU
    d_distancias = cuda.to_device(distancias)
    d_ventanas_tiempo = cuda.to_device(ventanas_tiempo)

    # Inicializar población aleatoria
    poblacion = np.array([random.sample(range(num_ciudades), num_ciudades) for _ in range(num_pob)], dtype=np.int64)
    d_poblacion = cuda.to_device(poblacion)

    # Inicializar costos
    costos = np.zeros(num_pob, dtype=np.float64)
    d_costos = cuda.to_device(costos)

    # Inicializar estados del RNG
    rng_states = create_xoroshiro128p_states(num_pob, seed=42)

    # Configurar los hilos y bloques
    threads_per_block = 256
    blocks_per_grid = (num_pob + (threads_per_block - 1)) // threads_per_block
    if blocks_per_grid < 8:
        blocks_per_grid = 8  # Aumentar para mejorar la utilización

    # Lanzar el kernel de costo inicial
    costo_kernel[blocks_per_grid, threads_per_block](d_poblacion, d_distancias, d_ventanas_tiempo, d_costos, num_ciudades)
    cuda.synchronize()

    # Recuperar costos a la CPU
    costos = d_costos.copy_to_host()
    poblacion = d_poblacion.copy_to_host()
    mejor_aptitud_historico = np.min(costos)
    mejor_individuo_historico = poblacion[np.argmin(costos)].copy()

    # Preparar arrays para padres e hijos
    padres = np.empty_like(poblacion)
    d_padres = cuda.to_device(padres)

    hijos = np.empty_like(poblacion)
    d_hijos = cuda.to_device(hijos)

    # Array para flags de cruzamiento
    crossover_flags = np.zeros(num_pob, dtype=np.int32)
    d_crossover_flags = cuda.to_device(crossover_flags)

    for gen in range(num_gen):
        # Selección de torneo en GPU
        seleccion_torneo_kernel[blocks_per_grid, threads_per_block](d_poblacion, d_costos, d_padres, num_pob, num_competidores, rng_states)
        cuda.synchronize()

        # Decidir si realizar el cruzamiento basado en la probabilidad pc
        crossover_decision_kernel[blocks_per_grid, threads_per_block](d_padres, d_crossover_flags, pc, num_pob, rng_states)
        cuda.synchronize()

        # Cruzamiento de ciclo en GPU
        cruzamiento_ciclo_kernel[blocks_per_grid * 2, threads_per_block](d_padres, d_hijos, d_crossover_flags, num_pob, num_ciudades, rng_states)
        cuda.synchronize()

        # Aplicar heurística en GPU
        heuristica_abruptos_kernel[blocks_per_grid, threads_per_block](d_hijos, num_pob, num_ciudades, m, d_distancias, d_ventanas_tiempo, d_costos)
        cuda.synchronize()

        # Actualizar población con los hijos
        d_poblacion = d_hijos.copy()
        cuda.synchronize()

        # Mutación en GPU
        mutacion_kernel[blocks_per_grid, threads_per_block](d_poblacion, d_costos, d_distancias, d_ventanas_tiempo, pm, num_ciudades, rng_states)
        cuda.synchronize()

        # Recalcular costos después de la mutación
        costo_kernel[blocks_per_grid, threads_per_block](d_poblacion, d_distancias, d_ventanas_tiempo, d_costos, num_ciudades)
        cuda.synchronize()

        # Recuperar costos y población a la CPU
        costos = d_costos.copy_to_host()
        poblacion = d_poblacion.copy_to_host()

        # Actualizar mejor historial
        mejor_aptitud_generacion = np.min(costos)
        mejor_individuo_generacion = poblacion[np.argmin(costos)].copy()
        if mejor_aptitud_generacion < mejor_aptitud_historico:
            mejor_aptitud_historico = mejor_aptitud_generacion
            mejor_individuo_historico = mejor_individuo_generacion.copy()

        print(f"Generación {gen+1}: Mejor Costo = {mejor_aptitud_historico}")

    return mejor_individuo_historico, mejor_aptitud_historico

# Función auxiliar para generar una permutación aleatoria (no necesaria en GPU)
def random_permutation(n, rng):
    perm = np.arange(n)
    rng.shuffle(perm)
    return perm

# Ejemplo de uso
if __name__ == "__main__":
    # Parámetros del algoritmo
    num_pob = 512        # Tamaño de la población
    num_gen = 100        # Número de generaciones
    pm = 0.01            # Probabilidad de mutación
    pc = 0.7             # Probabilidad de cruzamiento
    m = 5                # Número de ciudades más cercanas para la heurística
    num_competidores = 3 # Número de competidores en la selección de torneo

    # Generar datos de ejemplo
    num_ciudades = 50
    distancias = np.random.uniform(1, 100, size=(num_ciudades, num_ciudades)).astype(np.float64)
    np.fill_diagonal(distancias, 0)
    ventanas_tiempo = np.random.uniform(0, 24, size=(num_ciudades, 2)).astype(np.float64)

    # Ejecutar el algoritmo genético en GPU
    mejor_individuo, mejor_costo = ejecutar_algoritmo_cuda(num_pob, num_gen, pm, pc, m, num_competidores, distancias, ventanas_tiempo)

    print("Mejor Individuo:", mejor_individuo)
    print("Mejor Costo:", mejor_costo)

