padre1=[5 3 9 8 2 1 7 4 6];
padre2=[3 9 5 6 4 7 1 8 2];
hijo1 = CycleCrossover(padre1, padre2, 9)
hijo2 = CycleCrossover(padre2, padre1, 9)

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