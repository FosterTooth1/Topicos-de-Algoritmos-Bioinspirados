clear, clc, close all;

% Solicitar al usuario el número de veces que se debe repetir el algoritmo
Num_iteraciones = input('Ingrese el número de veces que se repetirá el algoritmo: ');

% Solicitar al usuario el número de generaciones en el algoritmo
Num_Generaciones = input('Ingrese el número de generaciones en el algoritmo: ');

% Establecemos nuestra poblacion y cantidad de variables con las que vamos a trabajar
Num_pob = input('Ingrese el numero de individuos dentro de la población: ');
Num_var = input('Ingrese la cantidad de variables que tiene cada individuo: ');

c1 = 3;
c2 = 3;
w = 0.6;

% Establecemos los limites inferiores y superiores de las variables en un ciclo for y los guardamos en un arreglo
Limite_Inferior = zeros(1, Num_var);
Limite_Superior = zeros(1, Num_var);
for i = 1:Num_var
    Limite_Inferior(i) = input(['Ingrese el limite inferior de la variable ', num2str(i), ': ']);
    Limite_Superior(i) = input(['Ingrese el limite superior de la variable ', num2str(i), ': ']);
end

Resultados_Generales = zeros(1, Num_iteraciones); % Para almacenar el mejor resultado de cada iteración

% Veces que se repite el algoritmo
for iteracion = 1:Num_iteraciones

    x = zeros(Num_pob, Num_var);
    v = zeros(Num_pob, Num_var);

    % Inicialización de velocidad y posición
    for i = 1:Num_pob
        for j = 1:Num_var
            x(i, j) = Limite_Inferior(j) + rand * (Limite_Superior(j) - Limite_Inferior(j));
            v(i, j) = -(Limite_Superior(j) - Limite_Inferior(j)) + 2 * rand * (Limite_Superior(j) - Limite_Inferior(j));
        end
    end

    x_p_best = x; % Mejores posiciones personales iniciales
    best_aptitud = inf(1, Num_pob); % Inicialización de la mejor aptitud

    % Ciclo de generaciones
    for iter = 1:Num_Generaciones
        aptitud = zeros(1, Num_pob);
        % Evaluación de la población en la función objetivo
        for i = 1:Num_pob
            aptitud(i) = - (x(i, 2) + 47) * sin(sqrt(abs(x(i, 2) + x(i, 1)/2 + 47))) - x(i, 1) * sin(sqrt(abs(x(i, 1) - (x(i, 2) + 47))));
        end

        [Mejor_Aptitud_Generacion, idx] = min(aptitud);
        x_g_best = x(idx, :); % Mejor global en esta generación

        if Mejor_Aptitud_Generacion < min(best_aptitud)
            x_p_best(idx, :) = x(idx, :);
            best_aptitud(idx) = aptitud(idx);
        end

        % Actualizar la velocidad y posición de partículas
        for i = 1:Num_pob
            for j = 1:Num_var
                r1 = rand;
                r2 = rand;
                v(i, j) = w * v(i, j) + c1 * r1 * (x_p_best(i, j) - x(i, j)) + c2 * r2 * (x_g_best(j) - x(i, j));
                x(i, j) = x(i, j) + v(i, j);

                % Aplicar restricciones de límites
                if x(i, j) < Limite_Inferior(j) || x(i, j) > Limite_Superior(j)
                    % Generar nuevos valores dentro del rango permitido
                    x(i, j) = Limite_Inferior(j) + rand * (Limite_Superior(j) - Limite_Inferior(j));
                    d = Limite_Superior(j) - Limite_Inferior(j);
                    v(i, j) = -d + 2 * rand * d;
                end
            end
        end
    end

    Resultados_Generales(iteracion) = Mejor_Aptitud_Generacion;
    fprintf('Solucion %d : %s, Costo: %d\n', iteracion, mat2str(x_g_best), Mejor_Aptitud_Generacion);
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