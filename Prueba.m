clc
clear
close all

a=imread("bones_hand_two.jpg");

b=im2double(a);


alpha=input('Ingrese el valor de Alpha: ');
delta=input('Ingrese el valor de Delta: ');

c = 1 ./ (1 + exp(-alpha * (b - delta)));
c = mat2gray(c);

figure(1)
subplot(1,2,1);
imshow(a);
title('Imagen Original');

subplot(1,2,2);
imshow(c);
title('Imagen Mejorada');

entropia_original=entropy(b);
entropia_mejorada=entropy(c);

d=im2double(c);

desviacion_estandar = std(b(:))
desviacion_estandara = std(d(:))

disp(['Entropia Original: ', num2str(entropia_original, '%.30f')]);
disp(['Entropia Mejorada: ', num2str(entropia_mejorada, '%.30f')]);

