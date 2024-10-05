clear, clc, close all;

a=imread("kodim23.png");

b=im2double(a);

% Solicitar al usuario el número de veces que se debe repetir el algoritmo
Num_iteraciones = input('Ingrese el número de veces que se repetirá el algoritmo: ');

% Solicitar al usuario el número de generaciones en el algoritmo
Num_Generaciones = input('Ingrese el número de generaciones en el algoritmo: ');

% Establecemos nuestra poblacion y cantidad de variables con las que vamos a trabajar
Num_pob = input('Ingrese el numero de individuos dentro de la población: ');
Num_var = 2;

c1 = 3;
c2 = 3;
w = 0.6;

% Establecemos los limites inferiores y superiores de las variables en un ciclo for y los guardamos en un arreglo
Limite_Inferior = zeros(1, Num_var);
Limite_Superior = zeros(1, Num_var);
Limite_Superior(1)=10;
Limite_Superior(2)=1;

Resultados_Generales = zeros(1, Num_iteraciones); % Para almacenar el mejor resultado de cada iteración
Mejor_Individuo_Iteracion= zeros(Num_iteraciones, Num_var);

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

    % Generacion de la matriz topologica
    nbh = zeros(Num_pob, 3);
    nbh(:,1) = (1:Num_pob)';
    nbh(:,2) = [Num_pob; (1:Num_pob-1)'];
    nbh(:,3) = [(2:Num_pob)'; 1];

    x_p_best = x; % Mejores posiciones personales iniciales
    best_aptitud = inf(1, Num_pob); % Inicialización de la mejor aptitud

    % Ciclo de generaciones
    for iter = 1:Num_Generaciones
        aptitud = zeros(1, Num_pob);
        % Evaluación de la población en la función objetivo
        for i = 1:Num_pob
            sumatoria = 0;
            c = 1 ./ (1 + exp(-(x(i,1)) * (b - x(i,2))));
            c = mat2gray(c);
            for t = 1:3
                sumatoria= sumatoria + (entropy(c(:,:,t)));
            end
            aptitud(i) = -sumatoria;
            %aptitud(i)=-entropy(c);
        end

        for i = 1:Num_pob
            if aptitud(i) < best_aptitud(i)
                x_p_best(i, :) = x(i, :);
                best_aptitud(i) = aptitud(i);
            end

            P1 = x(nbh(i,1),:);
            aptitudP1 = aptitud(nbh(i,1));
            P2 = x(nbh(i,2),:);
            aptitudP2 = aptitud(nbh(i,2));
            P3 = x(nbh(i,3),:);
            aptitudP3 = aptitud(nbh(i,3));

            [min_aptitud, idx_min] = min([aptitudP1, aptitudP2, aptitudP3]);
            if idx_min == 1
                x_g_best = P1;
            elseif idx_min == 2
                x_g_best = P2;
            else
                x_g_best = P3;
            end

            % Actualizar la velocidad y posición de partículas
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
    Mejor_Individuo_Iteracion(iteracion,:)=x_g_best(:);
    [Mejor_Aptitud_Generacion, idx] = min(aptitud);
    Resultados_Generales(iteracion) = Mejor_Aptitud_Generacion;
    fprintf('Solucion %d : %s, Costo: %d\n', iteracion, mat2str(x_g_best), Mejor_Aptitud_Generacion);
end

% Se obtienen las estadísticas de los mejores resultados de cada iteración
mejor = min(Resultados_Generales);
media = mean(Resultados_Generales);
peor = max(Resultados_Generales);
desviacion_estandar = std(Resultados_Generales);

disp('Los resultados generales del algoritmo son: ');
disp(['Mejor: ', num2str(mejor)]);
disp(['Media: ', num2str(media)]);
disp(['Peor: ', num2str(peor)]);
disp(['Desviación estándar: ', num2str(desviacion_estandar)]);

% Se encuentra el individuo (valores de x) asociado al mejor resultado
mejor_iteracion = find(Resultados_Generales == mejor, 1);
valores_mejor_x = Mejor_Individuo_Iteracion(mejor_iteracion, :);

entropia_original=entropy(a);

final = 1 ./ (1 + exp(-(valores_mejor_x(1)) * (b - (valores_mejor_x(2)))));
final = mat2gray(final);

entropia_mejorada=entropy(final);
disp(['Entropia Original: ', num2str(entropia_original)]);
disp(['Entropia Mejorada: ', num2str(entropia_mejorada)]);

figure(1)
subplot(1,2,1);
imshow(a);
title('Imagen Original');

subplot(1,2,2);
imshow(final);
title('Imagen Mejorada');