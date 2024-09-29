clc

a=imread("kodim23.png");
c=a;

b=im2double(a);

alpha=input('Ingrese el valor de Alpha: ');
delta=input('Ingrese el valor de Delta: ');

for t = 1:3
    c(:,:,t) = 255* (1 ./ (1 + exp(-alpha .* (b(:,:,t) - delta))));
end

figure(1)
subplot(1,2,1);
imshow(a);
title('Imagen Original');

subplot(1,2,2);
imshow(c);
title('Imagen Mejorada');
