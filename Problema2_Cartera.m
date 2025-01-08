clc
clear
close all

% Solicitar al usuario el número de veces que se debe repetir el algoritmo
Num_iteraciones = input('Ingrese el número de veces que se repetirá el algoritmo: ');

% Solicitar al usuario el número de generaciones en el algoritmo
Num_Generaciones = input('Ingrese el número de generaciones en el algoritmo: ');

% Establecemos nuestra población y cantidad de variables con las que vamos a trabajar
Num_pob = input('Ingrese el numero de individuos dentro de la población: ');
Num_var = 6;

% Establecemos los límites inferiores y superiores de las variables en un ciclo for y los guardamos en un arreglo
Limite_Inferior = zeros(1, Num_var);

Limite_Superior = zeros(1, Num_var);
for i = 1:Num_var
    valor = 0.4;
    Limite_Superior(i) = valor;
end

% Probabilidad de cruza
Pc = input('Ingrese la probabilidad de cruza del algoritmo: ');

% Probabilidad de mutación
Pm = input('Ingrese la probabilidad de mutacion del algoritmo: ');

lambda_P = 1000;

RBB=0.2;
RLOP=0.42;
RILI=1;
RHEAL=0.5;
RQUI=0.46;
RAUA=0.3;

Pesos= [RBB, RLOP, RILI, RHEAL, RQUI, RAUA];

% 
covarianza =  [0.032,  0.005,  0.03,  -0.031, -0.027,  0.01;
                    0.005,  0.1,    0.085, -0.07,  -0.05,   0.02;
                    0.03,   0.085,  0.333, -0.11,  -0.02,   0.042;
                   -0.031, -0.07,  -0.11,  0.125,  0.05,   -0.06;
                   -0.027, -0.05,  -0.02,  0.05,   0.065,  -0.02;
                    0.01,   0.02,   0.042, -0.06, -0.02,   0.08];

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
            x1 = Poblacion(i, 1);
            x2 = Poblacion(i, 2);
            x3 = Poblacion(i, 3);
            x4 = Poblacion(i, 4);
            x5 = Poblacion(i, 5);
            x6 = Poblacion(i, 6);
            X=[x1,x2,x3,x4,x5,x6];
            
            aux=0;
            for j=1:6
                aux=0;
                for k=1:6
                    aux= aux + X(j)*X(k)*covarianza(j,k);
                end
                aptitud(i)= aptitud(i) + aux;
            end
            
            g1 = .35 - (Pesos(1)*x1+Pesos(2)*x2+Pesos(3)*x3+Pesos(4)*x4+Pesos(5)*x5+Pesos(6)*x6);  % g1 <= 0
            g2 = (sum(X)) -1;
            Rd = [g1, g2];  % Vector de restricciones g(x)
            penalizacion = sum((max(Rd, 0)).^2);  % Penalización de g(x)
            %Ri = [h1, h2, ...]; % h(x) restricciones de igualdad
            %penalizacion = penalizacion + sum(Ri.^2);
    
            % Cálculo de la función penalizada
            aptitud(i) = aptitud(i) + lambda_P * penalizacion;

        end
        
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