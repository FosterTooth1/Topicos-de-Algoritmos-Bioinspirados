% Flores Lara Alberto 6BV1
% Practica 1

clc % Limpia la ventana de comandos

% Lectura de la imagen "Caries.jpg"
a = imread("Caries.jpg");

% Conversión de la imagen a tipo double
b = im2double(a);

% Solicitar al usuario el número de veces que se debe repetir el algoritmo
Num_iteraciones = input('Ingrese el número de veces que se repetirá el algoritmo: ');

% Solicitar al usuario el número de generaciones en el algoritmo
Num_Generaciones = input('Ingrese el número de generaciones en el algoritmo: ');

% Establecemos nuestra población y cantidad de variables con las que vamos a trabajar
Num_pob = input('Ingrese el numero de individuos dentro de la población: '); % Número de individuos en la población
Num_var = 2; % Número de variables con las que se trabajará

% Establecemos los límites inferiores y superiores de las variables
Limite_Inferior = zeros(1, Num_var); % Inicializamos límites inferiores
Limite_Superior = zeros(1, Num_var); % Inicializamos límites superiores
Limite_Superior(1) = 10; % Límite superior para la primera variable
Limite_Superior(2) = 1;  % Límite superior para la segunda variable

% Solicitar la probabilidad de cruza y mutación
Pc = input('Ingrese la probabilidad de cruza del algoritmo: ');
Pm = input('Ingrese la probabilidad de mutacion del algoritmo: ');

% Inicializamos la población con valores aleatorios dentro de los límites establecidos
Poblacion = zeros(Num_pob, Num_var);
for i = 1:Num_pob
    for j = 1:Num_var
        Poblacion(i,j) = Limite_Inferior(j) + rand * (Limite_Superior(j) - Limite_Inferior(j));
    end
end

% Inicialización de arreglos para almacenar resultados y los mejores individuos por iteración
Resultados_Generales = zeros(1, Num_iteraciones);
Mejor_Individuo_General = zeros(Num_iteraciones, Num_var);

% Ciclo principal de iteraciones
for iteracion = 1:Num_iteraciones

    % Inicialización de arreglos para almacenar resultados por generación
    Mejor_Aptitud = zeros(1, Num_Generaciones);
    Mejor_Individuo = zeros(Num_Generaciones, Num_var);

    % Ciclo para evaluar cada generación
    for generacion = 1:Num_Generaciones
        % Ajuste de parámetros de cruza y mutación dependiendo de la generación
        if generacion <= Num_Generaciones * 0.5
            Nc = 2;
            Nm = 20;
        elseif generacion <= Num_Generaciones * 0.75
            Nc = 5;
            Nm = 50;
        elseif generacion <= Num_Generaciones * 0.80
            Nc = 10;
            Nm = 75;
        elseif generacion <= Num_Generaciones * 0.95
            Nc = 15;
            Nm = 85;
        else
            Nc = 20;
            Nm = 100;
        end

        % Evaluación de la aptitud de la población actual
        aptitud = zeros(1, Num_pob);
        for i = 1:Num_pob
            c = 1 ./ (1 + exp(-(Poblacion(i,1)) * (b - Poblacion(i,2))));
            c = mat2gray(c);
            aptitud(i) = -std(c(:)); % Minimización de la desviación estándar
        end

        % Encontrar el mejor individuo de la generación
        posiciones_menor = find(aptitud == min(aptitud));
        Pos_Mejor = posiciones_menor(1);
        Mejor_Aptitud(generacion) = aptitud(Pos_Mejor);
        Mejor_Individuo(generacion, :) = Poblacion(Pos_Mejor, :);

        % Selección de padres mediante torneo
        Padres = zeros(Num_pob, Num_var);
        Torneo = [randperm(Num_pob); randperm(Num_pob)]';
        for i = 1:Num_pob
            if aptitud(Torneo(i,1)) < aptitud(Torneo(i,2))
                Padres(i,:) = Poblacion(Torneo(i,1),:);
            else
                Padres(i,:) = Poblacion(Torneo(i,2),:);
            end
        end

        % Cruza de padres para generar hijos
        Hijos = zeros(Num_pob, Num_var);
        hijo1 = zeros(1, Num_var);
        hijo2 = zeros(1, Num_var);
        for i = 1:2:Num_pob-1
            if rand <= Pc
                U = rand;
                for j = 1:Num_var
                    Padre1 = Padres(i,j);
                    Padre2 = Padres(i+1,j);
                    beta = 1 + (2/(Padre2 - Padre1)) * min([(Padre1 - Limite_Inferior(j)), (Limite_Superior(j) - Padre2)]);
                    alpha = 2 - (abs(beta)^-(Nc+1));
                    if U <= 1/alpha
                        beta_c = (U*alpha)^(1/(Nc+1));
                    else
                        beta_c = (1/(2 - U*alpha))^(1/(Nc+1));
                    end
                    hijo1(1,j) = 0.5 * ((Padre1 + Padre2) - beta_c * abs(Padre2 - Padre1));
                    hijo2(1,j) = 0.5 * ((Padre1 + Padre2) + beta_c * abs(Padre2 - Padre1));
                end
            else
                hijo1 = Padres(i,:);
                hijo2 = Padres(i+1,:);
            end
            Hijos(i,:) = hijo1;
            Hijos(i+1,:) = hijo2;
        end

        % Mutación de los hijos
        for i = 1:Num_pob
            for j = 1:Num_var
                if rand <= Pm
                    r = rand;
                    delta = min((Limite_Superior(j) - Hijos(i,j)), (Hijos(i,j) - Limite_Inferior(j))) / (Limite_Superior(j) - Limite_Inferior(j));
                    if r <= 0.5
                        deltaq = (2*r + (1 - 2*r)*(1 - delta)^(Nm + 1))^(1/(Nm + 1)) - 1;
                    else
                        deltaq = 1 - (2*(1 - r) + 2*(r - 0.5)*(1 - delta)^(Nm + 1))^(1/(Nm + 1));
                    end
                    Hijos(i,j) = Hijos(i,j) + deltaq * (Limite_Superior(j) - Limite_Inferior(j));
                end
            end
        end

        % Sustitución con elitismo extintivo
        Poblacion = Hijos; % Se transfieren los hijos a la nueva generación
        indice_remplazo = randi(Num_pob); % Selección aleatoria de un individuo para reemplazar
        Poblacion(indice_remplazo, :) = Mejor_Individuo(generacion, :); % Reemplazo con el mejor individuo de la generación

    end

    % Almacenar el mejor resultado y el mejor individuo de la iteración
    mejor = min(Mejor_Aptitud);
    Resultados_Generales(iteracion) = mejor;
    mejor_generacion = find(Mejor_Aptitud == mejor, 1);
    Mejor_Individuo_General(iteracion, :) = Mejor_Individuo(mejor_generacion, :);
end

% Se obtienen las estadísticas finales y se comparan con la imagen original
% Calcular las estadísticas de los mejores resultados obtenidos en cada iteración
mejor = min(Resultados_Generales); % Encuentra el mejor valor de aptitud global
media = mean(Resultados_Generales); % Calcula la media de los resultados generales
peor = max(Resultados_Generales); % Encuentra el peor valor de aptitud global
desviacion_estandar = std(Resultados_Generales); % Calcula la desviación estándar de los resultados

% Se encuentra el individuo (valores de x) asociado al mejor resultado global
mejor_iteracion = find(Resultados_Generales == mejor, 1); % Encuentra la iteración donde se obtuvo el mejor resultado
valores_mejor_x = Mejor_Individuo_General(mejor_iteracion, :); % Valores de las variables para la mejor aptitud

% Mostrar en pantalla los resultados generales del algoritmo genético
disp('Los resultados generales del algoritmo genético son: ');
disp(['Mejor aptitud: ', num2str(mejor)]);
disp(['Valores de las variables (x) para la mejor aptitud: ', num2str(valores_mejor_x)]);
disp(['Media: ', num2str(media)]);
disp(['Peor: ', num2str(peor)]);
disp(['Desviación estándar: ', num2str(desviacion_estandar)]);

% Comparación de la desviación estándar de la imagen original con la imagen mejorada
Desviacion_original = std(b(:)); % Cálculo de la desviación estándar de la imagen original

% Aplicar la transformación con los valores del mejor individuo
final = 1 ./ (1 + exp(-(valores_mejor_x(1)) * (b - valores_mejor_x(2))));
final = mat2gray(final); % Normalización de la imagen mejorada

final = im2double(final); % Conversión de la imagen mejorada a tipo double
Desviacion_mejorada = std(final(:)); % Cálculo de la desviación estándar de la imagen mejorada

% Mostrar en pantalla las desviaciones estándar de las imágenes
disp(['Desviacion Original: ', num2str(Desviacion_original, '%.30f')]);
disp(['Desviacion Mejorada: ', num2str(Desviacion_mejorada, '%.30f')]);

% Comparación de las desviaciones estándar entre la imagen original y la mejorada
if (Desviacion_mejorada > Desviacion_original)
    disp('La imagen mejorada tiene mayor desviacion estandar que la original');
elseif (Desviacion_original > Desviacion_mejorada)
    disp('La imagen mejorada tiene menor desviacion estandar que la original');
else
    disp('Las dos imagenes tienen la misma desviacion estandar');
end

% Comparación de la entropía de la imagen original con la imagen mejorada
final = mat2gray(final); % Asegura que la imagen mejorada también esté normalizada
entropia_original = entropy(a); % Cálculo de la entropía de la imagen original
entropia_mejorada = entropy(final); % Cálculo de la entropía de la imagen mejorada

% Mostrar en pantalla las entropías de las imágenes
disp(['Entropia Original: ', num2str(entropia_original, '%.30f')]);
disp(['Entropia Mejorada: ', num2str(entropia_mejorada, '%.30f')]);

% Comparación de las entropías entre la imagen original y la mejorada
if (entropia_mejorada > entropia_original)
    disp('La imagen mejorada tiene mayor entropia que la original');
elseif (entropia_original > entropia_mejorada)
    disp('La imagen mejorada tiene menor entropia que la original');
else
    disp('Las dos imagenes tienen la misma entropia');
end

% Visualización de la imagen original y la imagen mejorada
figure(1);
subplot(1,2,1); % Posición de la imagen original en la figura
imshow(a); % Mostrar la imagen original
title('Imagen Original'); % Título de la imagen original

subplot(1,2,2); % Posición de la imagen mejorada en la figura
imshow(final); % Mostrar la imagen mejorada
title('Imagen Mejorada'); % Título de la imagen mejorada

