clear, clc, close all;

% Solicitar al usuario el número de veces que se debe repetir el algoritmo
Num_iteraciones = input('Ingrese el número de veces que se repetirá el algoritmo: ');

% Solicitar al usuario el valor de m para la heuristica
m = input('Ingrese el valor de m para considerar en la heuristica: ');

% Creamos un arreglo donde se almacenaran los individuos con mejor aptitud (minimización) por cada iteración
Resultados_Generales = zeros(1, Num_iteraciones);

% Leer el archivo CSV
data = readmatrix('distancias.csv');

% Eliminar la primera columna, que contiene los encabezados laterales
Distancias = data(:, 2:end);

Permutacion_ciudades = zeros(Num_iteraciones, size(Distancias, 1));

for iteracion = 1:Num_iteraciones

    Num_Pob = 50;
    Num_Gen = 100;
    Pm = 0.15;
    Num_var = size(Distancias, 1);  % Número de ciudades basado en la matriz de distancias

    % Población inicial
    Pob = zeros(Num_Pob, Num_var);
    for i = 1:Num_Pob
        Pob(i, :) = randperm(Num_var);
        %Aplicacion de heuristica de remoción de apruptos
        Pob(i, :) = heuristica_abruptos(Pob(i,:), m, Distancias, Num_var);
    end

    % Evaluación
    Aptitud_Pob = zeros(Num_Pob, 1);
    for i = 1:Num_Pob
        Aptitud_Pob(i) = Costo(Pob(i, :), Distancias, Num_var);
    end

    % Inicializar mejor histórico
    [Mejor_Aptitud_Historico, idx] = min(Aptitud_Pob);
    Mejor_Individuo_Historico = Pob(idx, :);

    iter = 1;
    while iter <= Num_Gen

        % Selección de Padres por torneo
        Padres = SeleccionTorneo(Pob, Aptitud_Pob, Num_Pob);

        % Cruzamiento y evaluación
        Hijos = zeros(Num_Pob, Num_var);
        Aptitud_Hijos = zeros(Num_Pob, 1);

        for i = 1:2:Num_Pob
            % Generar hijo 1 e hijo 2
            hijo_1 = CycleCrossover(Padres(i, :), Padres(i+1, :), Num_var);
            hijo_2 = CycleCrossover(Padres(i+1, :), Padres(i, :), Num_var);

            %Aplicacion de heuristica de remoción de abruptos
            hijo_1 = heuristica_abruptos(hijo_1, m, Distancias, Num_var);
            hijo_2 = heuristica_abruptos(hijo_2, m, Distancias, Num_var);

            % Calcular aptitudes de los hijos
            Aptitud_Hijo_1 = Costo(hijo_1, Distancias, Num_var);
            Aptitud_Hijo_2 = Costo(hijo_2, Distancias, Num_var);

            % Calcular aptitudes de los Padres
            Aptitud_Padre_1 = Costo(Padres(i, :), Distancias, Num_var);
            Aptitud_Padre_2 = Costo(Padres(i+1, :), Distancias, Num_var);

            % Crear una matriz con todos los individuos y sus aptitudes
            individuos = [Padres(i, :); Padres(i+1, :); hijo_1; hijo_2];
            aptitudes = [Aptitud_Padre_1; Aptitud_Padre_2; Aptitud_Hijo_1; Aptitud_Hijo_2];

            % Ordenar individuos por aptitud
            [aptitudes_ordenadas, indices] = sort(aptitudes);
            mejores_individuos = individuos(indices(1:2), :);
            mejores_aptitudes = aptitudes_ordenadas(1:2);

            % Guardar los mejores individuos y sus aptitudes para la
            % siguiente generación
            Hijos(i, :) = mejores_individuos(1, :);
            Hijos(i+1, :) = mejores_individuos(2, :);
            Aptitud_Hijos(i) = mejores_aptitudes(1);
            Aptitud_Hijos(i+1) = mejores_aptitudes(2);
        end

        % Reemplazar nueva población con hijos
        Pob = Hijos;
        Aptitud_Pob = Aptitud_Hijos;

        % Mutación
        for i = 1:Num_Pob
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

nombres_ciudades = {'Aguascalientes', 'Baja California', 'Baja California Sur', ...
    'Campeche', 'Chiapas', 'Chihuahua', 'Coahuila', 'Colima', 'Durango', ...
    'Guanajuato', 'Guerrero', 'Hidalgo', 'Jalisco', 'Estado de México', ...
    'Michoacán', 'Morelos', 'Nayarit', 'Nuevo León', 'Oaxaca', 'Puebla', ...
    'Querétaro', 'Quintana Roo', 'San Luis Potosí', 'Sinaloa', 'Sonora', ...
    'Tabasco', 'Tamaulipas', 'Tlaxcala', 'Veracruz', 'Yucatán', ...
    'Zacatecas', 'CDMX'};

% Imprimir el orden de ciudades recorridas por la mejor solución encontrada
[~, idx] = min(Resultados_Generales);
mejor_solucion = Permutacion_ciudades(idx, :);

fprintf('Orden de ciudades recorridas:\n');
for i = 1:length(mejor_solucion)
    nombre_ciudad = nombres_ciudades{mejor_solucion(i)}; % Usar {} para acceder a un elemento de celda
    fprintf('%d. %s\n', i, nombre_ciudad); % Usar fprintf correctamente
end

function hijo = CycleCrossover(padre1, padre2, noCiudades)
hijo = zeros(1, noCiudades);           % Inicializar el hijo con ceros
visitado = false(1, noCiudades);       % Marcadores de las posiciones visitadas
ciclo = 0;                             % Contador para el ciclo

while any(~visitado) % Mientras haya alguna ciudad no visitada
    pos_inicio = find(~visitado, 1);   % Primera ciudad no visitada
    ciclo = ciclo + 1;
    pos_actual = pos_inicio;

    % Alternar Padres según el ciclo
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

function Padres = SeleccionTorneo(Pob, Aptitud_Pob, Num_Pob)
    Num_Competidores = 2;
    Padres = zeros(Num_Pob, size(Pob, 2));
    for i = 1:Num_Pob
        Competidores = randperm(Num_Pob, Num_Competidores);
        [~, idx] = min(Aptitud_Pob(Competidores));
        Padres(i, :) = Pob(Competidores(idx), :);
    end
end

function hijo_premium = heuristica_abruptos(Hijo, m, distancias, num_ciudades)
    for i=1:num_ciudades
        %Seleccion entre ciudades cercanas
        [~,idx]= sort(distancias(i,:));
        idx=idx(2:m+1);
        idx=randsample(idx, 1);

        %Posicion de inserción
        posiciones=find(Hijo==idx);
        posiciones=[posiciones posiciones+1];
        
        %Eliminar ciudad de su posición
        Ruta=Hijo;
        Premove=find(Hijo==i);
        Ruta(Premove)=[];
        
        %Ajustar las posiciones de inserción según la eliminación
        posiciones(posiciones>Premove)=posiciones(posiciones>Premove)-1;
        
        %Insertar el elemento en las nuevas posiciones (Concatenacion)
        Ruta1 = [Ruta(1:posiciones(1)-1), i, Ruta(posiciones(1):end)];
        Ruta2 = [Ruta(1:posiciones(2)-1), i, Ruta(posiciones(2):end)];
        
        %Seleccionar la mejor ruta entre: Hijo, Ruta1 y Ruta 2
        Aptitud_Hijo= Costo(Hijo, distancias, num_ciudades);
        Ruta_1 = Costo(Ruta1, distancias, num_ciudades);
        Ruta_2 = Costo(Ruta2, distancias, num_ciudades);
        
        %Sustitucion
        individuos=[Hijo; Ruta1; Ruta2];
        aptitudes=[Aptitud_Hijo; Ruta_1; Ruta_2];
        [~,idx]=sort(aptitudes);
        Hijo=individuos(idx(1),:);
    end
    hijo_premium=Hijo;
end