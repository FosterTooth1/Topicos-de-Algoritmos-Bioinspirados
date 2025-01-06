import ctypes
from ctypes import c_int, c_double, c_char_p, c_char, POINTER, Structure
import os

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
class AlgoritmoGenetico:
    def __init__(self, ruta_biblioteca):
        # Cargamos la biblioteca compartida desde la ruta proporcionada
        self.biblioteca = ctypes.CDLL(ruta_biblioteca)
        
        # Configuramos el tipo de retorno de la función `ejecutar_algoritmo_genetico`
        self.biblioteca.ejecutar_algoritmo_genetico.restype = POINTER(ResultadoGenetico)
        
        # Especificamos los tipos de argumentos que espera `ejecutar_algoritmo_genetico`
        self.biblioteca.ejecutar_algoritmo_genetico.argtypes = [
            c_int,      # tamano_poblacion
            c_int,      # longitud_genotipo
            c_int,      # num_generaciones
            c_int,      # num_competidores
            c_int,      # m parametro de heurística
            c_double,   # probabilidad_mutacion
            c_double,   # probabilidad_cruce
            c_char_p    # nombre_archivo (ruta al archivo con matriz de distancias)
        ]
        
        # Configuramos los argumentos de la función para liberar resultados
        self.biblioteca.liberar_resultado.argtypes = [POINTER(ResultadoGenetico)]

    def ejecutar(self, tamano_poblacion, longitud_genotipo, num_generaciones,
                 num_competidores, m, probabilidad_mutacion, 
                 probabilidad_cruce, nombre_archivo):
        try:
            # Convertimos el nombre del archivo a una cadena de bytes
            nombre_archivo_bytes = nombre_archivo.encode('utf-8')
            
            # Llamamos a la función `ejecutar_algoritmo_genetico` de la biblioteca C
            resultado = self.biblioteca.ejecutar_algoritmo_genetico(
                tamano_poblacion,
                longitud_genotipo,
                num_generaciones,
                num_competidores,
                m,
                probabilidad_mutacion,
                probabilidad_cruce,
                nombre_archivo_bytes
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
    # Obtenemos la ruta absoluta del archivo actual
    directorio_actual = os.path.dirname(os.path.abspath(__file__))
    
    # Definimos el nombre del archivo de la biblioteca compartida según el sistema operativo
    nombre_biblioteca = "genetic_algo.dll" if os.name == 'nt' else "libgenetic_algo.so"
    ruta_biblioteca = os.path.join(directorio_actual, nombre_biblioteca)
    
    # Verificamos que el archivo de la biblioteca existe en la ruta especificada
    if not os.path.exists(ruta_biblioteca):
        raise RuntimeError(f"No se encuentra la biblioteca en {ruta_biblioteca}")
    
    # Creamos una instancia del wrapper para la biblioteca
    ag = AlgoritmoGenetico(ruta_biblioteca)
    
    # Definir los parámetros para el algoritmo genético
    tamano_poblacion = 1000
    longitud_genotipo = 32
    num_generaciones = 100
    num_competidores = 2
    m = 3
    probabilidad_mutacion = 0.3
    probabilidad_cruce = 0.9
    nombre_archivo = "Distancias_no_head.csv"
    
    # Ejecutar el algoritmo genético con estos parámetros
    resultado = ag.ejecutar(
        tamano_poblacion=tamano_poblacion,
        longitud_genotipo=longitud_genotipo,
        num_generaciones=num_generaciones,
        num_competidores=num_competidores,
        m=m,
        probabilidad_mutacion=probabilidad_mutacion,
        probabilidad_cruce=probabilidad_cruce,
        nombre_archivo=nombre_archivo
    )
    
    # Mostrar los resultados
    print("\nMejor ruta encontrada:")
    for i, (indice_ciudad, nombre_ciudad) in enumerate(zip(resultado['recorrido'], resultado['nombres_ciudades'])):
        print(f"{i+1}. {nombre_ciudad} (índice: {indice_ciudad})")
    print(f"\nFitness: {resultado['fitness']}")
    print(f"Tiempo de ejecución: {resultado['tiempo_ejecucion']:.2f} segundos")

# Punto de entrada principal del programa
if __name__ == "__main__":
    main()
