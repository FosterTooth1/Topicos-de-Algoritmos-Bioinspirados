% Flores Lara Alberto 6BV1
% Practica 1

clc
clear 
close all

% Solicitar al usuario el número de veces que se debe repetir el algoritmo
Num_iteraciones = input('Ingrese el número de veces que se repetirá el algoritmo: ');

% Solicitar al usuario el número de generaciones en el algoritmo
Num_Generaciones = input('Ingrese el número de generaciones en el algoritmo: ');

% Establecemos nuestra población y cantidad de variables con las que vamos a trabajar
Num_pob = input('Ingrese el numero de individuos dentro de la población: ');
Num_var = input('Ingrese la cantidad de variables que tiene cada individuo: ');

% Establecemos los límites inferiores y superiores de las variables en un ciclo for y los guardamos en un arreglo
Limite_Inferior = zeros(1, Num_var);
for i = 1:Num_var
    valor = input(['Ingrese el limite inferior de la variable ', num2str(i), ': ']);
    Limite_Inferior(i) = valor;
end

Limite_Superior = zeros(1, Num_var);
for i = 1:Num_var
    valor = input(['Ingrese el limite superior de la variable ', num2str(i), ': ']);
    Limite_Superior(i) = valor;
end

% Solicitar al usuario el valor de alfa en el fitness sharing
alpha_sharing = input('Ingrese el valor de alpha en el fitness sharing: ');

% Solicitar al usuario la cantidad de picos que quiere considerar en la exploración
q = input('Ingrese la cantidad de picos para la exploración: ');

% Calcular el radio de nicho utilizando la fórmula proporcionada
suma_diferencias_cuadradas = 0;
for k = 1:Num_var
    suma_diferencias_cuadradas = suma_diferencias_cuadradas + (Limite_Superior(k) - Limite_Inferior(k))^2;
end
sigma_share = sqrt(suma_diferencias_cuadradas) / (2 * q^(1/Num_var));

disp(['El radio de nicho calculado es: ', num2str(sigma_share)]);

% Probabilidad de cruza
Pc = input('Ingrese la probabilidad de cruza del algoritmo: ');

% Probabilidad de mutación
Pm = input('Ingrese la probabilidad de mutacion del algoritmo: ');

% Inicialización de la población
Poblacion = zeros(Num_pob, Num_var);
for i = 1:Num_pob
    for j = 1:Num_var
        Poblacion(i,j) = (Limite_Inferior(j) + rand * (Limite_Superior(j) - Limite_Inferior(j)));
    end
end

% Creamos un arreglo donde se almacenarán los mejores resultados por iteración
Resultados_Generales = zeros(1, Num_iteraciones);

% Creamos un arreglo para almacenar los mejores individuos (sus variables) por iteración
Mejor_Individuo_General = zeros(Num_iteraciones, Num_var);

for iteracion = 1:Num_iteraciones
    % Creamos un arreglo donde se almacenarán los mejores resultados por generación
    Mejor_Aptitud = zeros(1, Num_Generaciones);
    
    % Creamos un arreglo para almacenar el mejor individuo por generación
    Mejor_Individuo = zeros(Num_Generaciones, Num_var);

    % Ciclo para las generaciones
    for generacion = 1:Num_Generaciones
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

        % Evaluación de la población en la función
        aptitud = zeros(1, Num_pob);
        
        for i = 1:Num_pob
            x = Poblacion(i, 1);
            y = Poblacion(i, 2);
            aptitud(i) = 20 + (x^2 - 10 * cos(2 * pi * x)) + (y^2 - 10 * cos(2 * pi * y));
        end
        
        % Calcular la aptitud compartida utilizando Fitness Sharing
        aptitud_compartida = zeros(1, Num_pob);
        for i = 1:Num_pob
            Compartido = 0; % Inicializar la suma de recursos compartidos
            for j = 1:Num_pob
                if i ~= j
                    % Calcular la distancia entre el individuo i y el individuo j
                    Distancia = norm(Poblacion(i, :) - Poblacion(j, :));
                    
                    % Verificar si la distancia es menor que el radio de nicho
                    if Distancia < sigma_share
                        % Calcular la porción de recurso compartido usando la función de repartición
                        sh = 1 - (Distancia / sigma_share)^(alpha_sharing); % alpha = 1
                        Compartido = Compartido + sh; % Sumar la porción de recurso compartido
                    end
                end
            end
            % Modificar la aptitud del individuo con el valor de recurso compartido
            aptitud_compartida(i) = aptitud(i) * (1 + Compartido); % Actualizar la aptitud compartida
        end

        aptitud=aptitud_compartida;

        % Gráfica de la población en el espacio 2D para la generación actual
        figure(iteracion); % Crear una nueva figura para cada iteración
        scatter(Poblacion(:, 1), Poblacion(:, 2), 50, 'filled'); % Graficar la población actual
        title(['Iteración ', num2str(iteracion), ' - Generación ', num2str(generacion)]);
        xlabel('Variable X');
        ylabel('Variable Y');
        xlim([Limite_Inferior(1), Limite_Superior(1)]);
        ylim([Limite_Inferior(2), Limite_Superior(2)]);
        grid on;
        drawnow; % Actualizar la gráfica en cada generación

        
        % Obtener la posición de la población con mejor aptitud (minimización)
        posiciones_menor = find(aptitud == min(aptitud));
        Pos_Mejor = posiciones_menor(1);

        % Almacenar la mejor aptitud y el mejor individuo por generación
        Mejor_Aptitud(generacion) = aptitud(Pos_Mejor);
        Mejor_Individuo(generacion, :) = Poblacion(Pos_Mejor, :);

        %Codigo del Torneo
        Padres = zeros(Num_pob,Num_var);
        Torneo = [randperm(Num_pob); randperm(Num_pob)]';

        for i = 1:Num_pob
            if aptitud(Torneo(i,1))<aptitud(Torneo(i,2))
                Padres(i,:) = Poblacion(Torneo(i,1),:);

            else
                Padres(i,:) = Poblacion(Torneo(i,2),:);
            end
        end

        %Cruzamiento
        Hijos=zeros(Num_pob,Num_var);
        hijo1 = zeros(1, Num_var);
        hijo2 = zeros(1, Num_var);

        for i=1:2:Num_pob-1
            if rand<=Pc
                U=rand;
                for j=1:Num_var
                    Padre1=Padres(i,j);
                    Padre2=Padres(i+1,j);
                    beta = 1 + (2/(Padre2 - Padre1))* (min([(Padre1-Limite_Inferior(j)), (Limite_Superior(j)-Padre2)]));
                    alpha=2-(abs(beta)^-(Nc+1));
                    if U<=1/alpha
                        beta_c=(U*alpha)^(1/(Nc+1));
                    else
                        beta_c=(1/(2-U*alpha))^(1/(Nc+1));
                    end
                    hijo1(1,j)=0.5*((Padre1+Padre2)-beta_c*abs(Padre2-Padre1));
                    hijo2(1,j)=0.5*((Padre1+Padre2)+beta_c*abs(Padre2-Padre1));
                end
            else
                hijo1=Padres(i,:);
                hijo2=Padres(i+1,:);
            end
            Hijos(i,:)=hijo1;
            Hijos(i+1,:)=hijo2;
        end

        %mutation: polynomial
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
                    %Mutar el individuo
                    Hijos(i,j) = Hijos(i,j) + deltaq * (Limite_Superior(j) - Limite_Inferior(j));
                end
            end
        end

        % Sustitución con elitismo extintivo
        % Seleccionamos todos los hijos para la siguiente generación excepto uno
        Poblacion = Hijos;
        
        % Seleccionamos aleatoriamente un individuo de la población de hijos
        indice_remplazo = randi(Num_pob);
        
        % Reemplazamos al individuo seleccionado con el mejor individuo de la generación anterior
        Poblacion(indice_remplazo, :) = Mejor_Individuo(generacion, :);
    end

    % Se obtienen las estadísticas de los resultados obtenidos en cada iteración
    mejor = min(Mejor_Aptitud);

    % Almacenar el mejor resultado y el mejor individuo de la iteración
    Resultados_Generales(iteracion) = mejor;
    mejor_generacion = find(Mejor_Aptitud == mejor, 1);
    Mejor_Individuo_General(iteracion, :) = Mejor_Individuo(mejor_generacion, :);
end

% Se obtienen las estadísticas de los mejores resultados de cada iteración
mejor = min(Resultados_Generales);
media = mean(Resultados_Generales);
peor = max(Resultados_Generales);
desviacion_estandar = std(Resultados_Generales);

% Se encuentra el individuo (valores de x) asociado al mejor resultado
mejor_iteracion = find(Resultados_Generales == mejor, 1);
valores_mejor_x = Mejor_Individuo_General(mejor_iteracion, :);

disp('Los resultados generales del algoritmo genético son: ');
disp(['Mejor aptitud: ', num2str(mejor)]);
disp(['Valores de las variables (x) para la mejor aptitud: ', num2str(valores_mejor_x)]);
disp(['Media: ', num2str(media)]);
disp(['Peor: ', num2str(peor)]);
disp(['Desviación estándar: ', num2str(desviacion_estandar)]);

