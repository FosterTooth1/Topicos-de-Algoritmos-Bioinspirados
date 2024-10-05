clc
clear
close all

a=imread("kodim23.png");

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

entropia_original=entropy(a);
entropia_mejorada=entropy(c);
disp(['Entropia Original: ', num2str(entropia_original)]);
disp(['Entropia Mejorada: ', num2str(entropia_mejorada)]);