    clear, clc, close all;

    % Solicitar al usuario el número de veces que se debe repetir el algoritmo
    Num_iteraciones = input('Ingrese el número de veces que se repetirá el algoritmo: ');

    % Solicitar al usuario el número de veces que se debe repetir el algoritmo
    ciudad_inicio = input('Ingrese el numero de la ciudad de inicio: ');
    
    % Creamos un arreglo donde se almacenaran los individuos con mejor aptitud (minimización) por cada iteración
    Resultados_Generales = zeros(1, Num_iteraciones);
    
    Distancias = [0, 61.82, 18.54, 37.52, 54.08,  1.88, 59.98, 32.82, 69.42, 36.76, 60.26;
    61.82, 0, 50.84, 33.62,  7.50, 59.88,  2.76, 28.84,  7.78, 28.14,  5.80;
    18.54, 50.84, 0, 26.74, 43.38, 18.60, 49.28, 22.00, 58.70, 23.36, 49.30;
    37.52, 33.62, 26.74, 0, 26.16, 35.56, 32.06,  4.80, 41.50,  3.26, 32.08;
    54.08,  7.50, 43.38, 26.16, 0, 52.06, 57.96, 21.38, 15.34, 20.68,  5.92;
     1.88, 59.88, 18.60, 35.56, 52.06, 0, 57.96, 30.86, 67.38, 34.80, 58.30;
    59.98,  2.76, 49.28, 32.06, 57.96, 57.96, 0, 27.28, 10.62, 26.58,  6.76;
    32.82, 28.84, 22.00,  4.80, 21.38, 30.86, 27.28, 0, 36.72,  4.02, 27.30;
    69.42,  7.78, 58.70, 41.50, 15.34, 67.38, 10.62, 36.72, 0, 36.02, 12.14;
    36.76, 28.14, 23.36,  3.26, 20.68, 34.80, 26.58,  4.02, 36.02, 0, 26.60;
    60.26,  5.80, 49.30, 32.08,  5.92, 58.30,  6.76, 27.30, 12.14, 26.60, 0];

    Ventana_tiempo= [ ...
        0, 10;
        50, 70;
        15, 35;
        30, 50;
        40, 60;
         5, 15;
        55, 75;
        25, 45;
        65, 85;
        20, 40;
        45, 65];
    
    
    
    Permutacion_ciudades = zeros(Num_iteraciones, size(Distancias, 1));
    
    for iteracion = 1:Num_iteraciones
        
        Num_pob = 211;
        Num_Gen = 1500;  % Aumentar el número de generaciones
        Pm = 0.15;  % Aumentar la probabilidad de mutación
        Num_var = size(Distancias, 1);  % Número de ciudades basado en la matriz de distancias
        
        % Población inicial
        Pob = zeros(Num_pob, Num_var);
        for i = 1:Num_pob
            Pob(i, :) = randperm(Num_var);
        end
        
        % Evaluación
        Aptitud_Pob = zeros(Num_pob, 1);
        for i = 1:Num_pob
            Aptitud_Pob(i) = Costo(Pob(i, :),ciudad_inicio ,Distancias, Ventana_tiempo , Num_var);
        end
        
        % Inicializar mejor histórico
        [Mejor_Aptitud_Historico, idx] = min(Aptitud_Pob);
        Mejor_Individuo_Historico = Pob(idx, :);
        
        iter = 1;
        while iter <= Num_Gen
            % Elitismo: mantener los mejores individuos
            [~, indices] = sort(Aptitud_Pob);
            elite_size = round(0.1 * Num_pob);
            Nueva_Pob = Pob(indices(1:elite_size), :);
            Nueva_Aptitud = Aptitud_Pob(indices(1:elite_size));
            
            % Selección de padres por torneo
            padres = SeleccionTorneo(Pob, Aptitud_Pob, Num_pob - elite_size);
            
            % Cruzamiento y evaluación
            Hijos = zeros(Num_pob - elite_size, Num_var);
            Aptitud_Hijos = zeros(Num_pob - elite_size, 1);
            for i = 1:2:(Num_pob - elite_size)
                % Generar hijo
                hijo = CycleCrossover(padres(i, :), padres(i+1, :), Num_var);
                Hijos(i, :) = hijo;
                Aptitud_Hijos(i) = Costo(hijo,ciudad_inicio ,Distancias, Ventana_tiempo , Num_var);
                
                if i+1 <= Num_pob - elite_size
                    hijo = CycleCrossover(padres(i+1, :), padres(i, :), Num_var);
                    Hijos(i+1, :) = hijo;
                    Aptitud_Hijos(i+1) = Costo(hijo,ciudad_inicio ,Distancias, Ventana_tiempo , Num_var);
                end
            end
        
            % Mezclar nueva población con hijos
            Pob = [Nueva_Pob; Hijos];
            Aptitud_Pob = [Nueva_Aptitud; Aptitud_Hijos];
        
            % Mutación
            for i = 1:Num_pob
                if rand <= Pm
                    Pob(i, :) = Mutacion(Pob(i, :), Num_var);
                    Aptitud_Pob(i) = Costo(Pob(i, :),ciudad_inicio ,Distancias, Ventana_tiempo , Num_var);
                end
            end
            
            % Actualizar mejor histórico si es necesario
            [Mejor_Aptitud_Generacion, idx] = min(Aptitud_Pob);
            if Mejor_Aptitud_Generacion < Mejor_Aptitud_Historico
                Mejor_Aptitud_Historico = Mejor_Aptitud_Generacion;
                Mejor_Individuo_Historico = Pob(idx, :);
            end
        
            iter = iter + 1;
        end
        
        % Mostrar resultados finales
        [Mejor_Aptitud_Final, idx] = min(Aptitud_Pob);
        fprintf('Solucion %d : %s, Costo: %d\n', iteracion, mat2str(Mejor_Individuo_Historico), Mejor_Aptitud_Historico);
        
        Resultados_Generales(iteracion) = Mejor_Aptitud_Historico;
        Permutacion_ciudades(iteracion, :) = Mejor_Individuo_Historico;
    end
    
    % Se obtienen las estadísticas de los mejores resultados de cada iteración
    mejor = min(Resultados_Generales);
    media = mean(Resultados_Generales);
    peor = max(Resultados_Generales);
    desviacion_estandar = std(Resultados_Generales);
    
    disp('Los resultados generales del algoritmo genético son: ');
    disp(['Mejor: ', num2str(mejor)]);
    disp(['Media: ', num2str(media)]);
    disp(['Peor: ', num2str(peor)]);
    disp(['Desviación estándar: ', num2str(desviacion_estandar)]);
    
    nombres_ciudades = {'New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', ...
                    'Philadelphia', 'San Diego', 'Dallas', 'San Francisco', ...
                    'Austin', 'Las Vegas'};
    
    % Imprimir el orden de ciudades recorridas por la mejor solución encontrada
    [~, idx] = min(Resultados_Generales);
    mejor_solucion = Permutacion_ciudades(idx, :);
    mejor_solucion= rearrangeFromIndex(mejor_solucion, Pos_inicio);
    
    fprintf('Orden de ciudades recorridas:\n');
    for i = 1:length(mejor_solucion)
        nombre_ciudad = nombres_ciudades{mejor_solucion(i)}; % Usar {} para acceder a un elemento de celda
        fprintf('%d. %s\n', i, nombre_ciudad); % Usar fprintf correctamente
    end
    
    function Ciudades_Vecinas = CrearLista(padre1, padre2, noCiudades)
        Ciudades_Vecinas = cell(1, noCiudades); % Arreglo de celdas
        for i = 1:noCiudades
            ciudad = padre1(i);
            indx1 = [i-1, i+1];
            indx2 = find(padre2 == ciudad);
            indx2 = [indx2-1, indx2+1];
    
            indx1(indx1 == 0) = noCiudades;
            indx1(indx1 > noCiudades) = 1;
            indx2(indx2 == 0) = noCiudades;
            indx2(indx2 > noCiudades) = 1;
    
            vecinos = unique([padre1(indx1), padre2(indx2)]);
            Ciudades_Vecinas{ciudad} = vecinos;
        end
    end
    
    
    function hijo = CycleCrossover(padre1, padre2, noCiudades)
        hijo = zeros(1, noCiudades);           % Inicializar el hijo con ceros
        visitado = false(1, noCiudades);       % Marcadores de las posiciones visitadas
        ciclo = 0;                             % Contador para el ciclo
    
        while any(~visitado) % Mientras haya alguna ciudad no visitada
            pos_inicio = find(~visitado, 1);   % Primera ciudad no visitada
            ciclo = ciclo + 1;
            pos_actual = pos_inicio;
            
            % Alternar padres según el ciclo
            if mod(ciclo, 2) == 1
                padre_actual = padre1;
                otro_padre = padre2;
            else
                padre_actual = padre2;
                otro_padre = padre1;
            end
    
            while true
                hijo(pos_actual) = padre_actual(pos_actual); % Asignar valor
                visitado(pos_actual) = true; % Marcar como visitado
    
                % Buscar siguiente posición
                valor_actual = padre_actual(pos_actual);
                pos_actual = find(otro_padre == valor_actual, 1);
    
                % Validar posición actual
                if isempty(pos_actual)
                    error('Error: posición actual no encontrada en el otro padre.');
                end
    
                % Condición de salida del ciclo
                if visitado(pos_actual) || pos_actual == pos_inicio
                    break;
                end
            end
        end
    end
    
    function costo = Costo(recorrido, ciudad_inicio, Distancias , Ventana_tiempo, noCiudades)
        costo = 0;
        Pos_inicio= find(recorrido==ciudad_inicio);
        recorrido= rearrangeFromIndex(recorrido, Pos_inicio);
        for i = 1:noCiudades-1
            if i+1 > length(recorrido) || recorrido(i+1) > size(Distancias, 1)
                error('Índice fuera de los límites al calcular costo. Verifica el recorrido.');
            end
            costo = costo + Distancias(recorrido(i), recorrido(i+1));
            vent_inferior = Ventana_tiempo(recorrido(i), 1);
            vent_superior = Ventana_tiempo(recorrido(i), 2);
            if costo < vent_inferior
                costo= vent_inferior;
            end 
            if costo > vent_superior
                penalizacion = sum((max(costo-vent_superior, 0)).^2);
                costo = costo + 1*penalizacion;
            end
        end
        costo = costo + Distancias(recorrido(noCiudades), recorrido(1));
        vent_inferior = Ventana_tiempo(recorrido(1), 1);
        vent_superior = Ventana_tiempo(recorrido(1), 2);
        if costo < vent_inferior
            costo= vent_inferior;
        end 
        if costo > vent_superior
            penalizacion = sum((max(costo-vent_superior, 0)).^2);
            costo = costo + 1*penalizacion;
        end
    end
    
    function mutado = Mutacion(individuo, noCiudades)
        idx = randperm(noCiudades, 2);
        mutado = individuo;
        mutado(idx(1)) = individuo(idx(2));
        mutado(idx(2)) = individuo(idx(1));
    end
    
    function padres = SeleccionTorneo(Pob, Aptitud_Pob, num_pares)
        num_competidores = 2;
        num_pob = size(Pob, 1);
        padres = zeros(num_pares, size(Pob, 2));
        for i = 1:num_pares
            competidores = randperm(num_pob, num_competidores);
            [~, idx] = min(Aptitud_Pob(competidores));
            padres(i, :) = Pob(competidores(idx), :);
        end
    end
    
function rearrangedArray = rearrangeFromIndex(array, startIndex)
    % Verificar que el índice sea válido
    if startIndex < 1 || startIndex > length(array)
        error('El índice debe estar entre 1 y el tamaño del arreglo.');
    end
    
    % Rearreglar el arreglo desde el índice dado
    rearrangedArray = [array(startIndex:end), array(1:startIndex-1)];
end

    