clear, clc, close all;

% Solicitar al usuario el número de veces que se debe repetir el algoritmo
Num_iteraciones = input('Ingrese el número de veces que se repetirá el algoritmo: ');

% Creamos un arreglo donde se almacenaran los individuos con mejor aptitud (minimización) por cada iteración
Resultados_Generales = zeros(1, Num_iteraciones);

% Leer el archivo CSV
data = readmatrix('distancias.csv');

% Eliminar la primera columna, que contiene los encabezados laterales
Distancias = data(:, 2:end);

Permutacion_ciudades = zeros(Num_iteraciones, size(Distancias, 1));

for iteracion = 1:Num_iteraciones
    
    Num_pob = 100;
    Num_Gen = 200;  % Aumentar el número de generaciones
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
        Aptitud_Pob(i) = Costo(Pob(i, :), Distancias, Num_var);
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
            Aptitud_Hijos(i) = Costo(hijo, Distancias, Num_var);
            
            if i+1 <= Num_pob - elite_size
                hijo = CycleCrossover(padres(i+1, :), padres(i, :), Num_var);
                Hijos(i+1, :) = hijo;
                Aptitud_Hijos(i+1) = Costo(hijo, Distancias, Num_var);
            end
        end
    
        % Mezclar nueva población con hijos
        Pob = [Nueva_Pob; Hijos];
        Aptitud_Pob = [Nueva_Aptitud; Aptitud_Hijos];
    
        % Mutación
        for i = 1:Num_pob
            if rand <= Pm
                Pob(i, :) = Mutacion(Pob(i, :), Num_var);
                Aptitud_Pob(i) = Costo(Pob(i, :), Distancias, Num_var);
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

% Leer el archivo CSV
fid = fopen('distancias.csv');
nombres_ciudades_cell = textscan(fid, '%s', 'Delimiter', ',');
fclose(fid);

% Convertir a matriz de caracteres (string)
nombres_ciudades = char(nombres_ciudades_cell{1}(2:end)); % Convertir y omitir la primera fila

% Obtener solo el número de ciudades necesarias
num_ciudades = size(Distancias, 1);  % Número de ciudades basado en la matriz de distancias
nombres_ciudades = nombres_ciudades(1:num_ciudades, :); % Desde la segunda celda hasta el número deseado

% Imprimir el orden de ciudades recorridas por la mejor solución encontrada
[~, idx] = min(Resultados_Generales);
mejor_solucion = Permutacion_ciudades(idx, :);

fprintf('Orden de ciudades recorridas:\n');
for i = 1:length(mejor_solucion)
    nombre_ciudad = nombres_ciudades(mejor_solucion(i), :);
    fprintf('%d. %s\n', i, nombre_ciudad);
end

function Ciudades_Vecinas = CrearLista(padre1, padre2, noCiudades)
    % Arreglo de celdas
    Ciudades_Vecinas = cell(1, noCiudades); 
    for i = 1:noCiudades
        ciudad = padre1(i);
        indx1 = [i-1, i+1];
        indx1(indx1 == 0) = noCiudades; % Ajustar índices fuera de rango
        indx1(indx1 > noCiudades) = 1;

        indx2 = find(padre2 == ciudad);
        if isempty(indx2)
            indx2 = []; % Si no hay coincidencia
        else
            indx2 = [indx2-1, indx2+1];
            indx2(indx2 == 0) = noCiudades; % Ajustar índices fuera de rango
            indx2(indx2 > noCiudades) = 1;
        end

        % Ciudades no repetidas
        vecinos = unique([padre1(indx1), padre2(indx2)]);
        Ciudades_Vecinas{ciudad} = vecinos;
    end
end


function hijo = CycleCrossover(padre1, padre2, noCiudades)
    hijo = zeros(1, noCiudades);           % Inicializar el hijo con ceros
    visitado = false(1, noCiudades);       % Marcadores de las posiciones visitadas
    ciclo = 0;                             % Contador para el ciclo

    while any(~visitado) % Mientras haya alguna ciudad no visitada
        if mod(ciclo, 2) == 1
            % Si el ciclo es impar, empezamos con el primer padre
            pos_inicio = find(~visitado, 1);
            pos_inicio = padre1(pos_inicio);
        else
            % Si el ciclo es par, empezamos con el segundo padre
            pos_inicio = find(~visitado, 1);
            pos_inicio = padre2(pos_inicio);
        end
        ciclo = ciclo + 1;

        % Definir los padres actuales según el ciclo
        if mod(ciclo, 2) == 1
            padre_actual = padre1;        
            otro_padre = padre2;
        else
            padre_actual = padre2; 
            otro_padre = padre1;
        end

        pos_actual = pos_inicio;
        while true
            hijo(pos_actual) = padre_actual(pos_actual); % Asignar valor del padre actual
            visitado(pos_actual) = true; % Marcar posición como visitada

            % Encontramos la siguiente posición en el otro padre
            valor_actual = padre_actual(pos_actual);
            pos_actual = find(otro_padre == valor_actual);

            % Romper el bucle si se completa el ciclo
            if pos_actual == pos_inicio
                break;
            end
        end
    end
    hijo(hijo == 0) = padre2(hijo == 0);
end




function costo = Costo(recorrido, Distancias, noCiudades)
    costo = 0;
    for i = 1:noCiudades-1
        costo = costo + Distancias(recorrido(i), recorrido(i+1));
    end
    costo = costo + Distancias(recorrido(noCiudades), recorrido(1)); % Retorno a la ciudad inicial
end

function mutado = Mutacion(individuo, noCiudades)
    idx = randperm(noCiudades, 2);
    mutado = individuo;
    mutado(idx(1)) = individuo(idx(2));
    mutado(idx(2)) = individuo(idx(1));
end

function padres = SeleccionTorneo(Pob, Aptitud_Pob, num_pares)
    num_competidores = 2;  % Número de competidores en el torneo
    num_pob = size(Pob, 1);
    padres = zeros(num_pares, size(Pob, 2));
    for i = 1:num_pares
        competidores = randperm(num_pob, num_competidores);
        [~, idx] = min(Aptitud_Pob(competidores));
        padres(i, :) = Pob(competidores(idx), :);
    end
end


