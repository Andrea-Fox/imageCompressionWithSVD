clear all
image = 'immagine_2';
immagine_originale = imread([image, '/immagine_prova.jpg']);
immagine_compressa_ROI = imread([image, '/compressed_image_ROI.png']);
immagine_compressa = imread([image, '/compressed_image.png']);
immagine_compressa_jpeg = imread([image, '/compressed_image_jpeg.png']); 

[m, n, p] = size(immagine_originale);

MAX_I = 256;
for i=1:3
    MSE_ROI(i) = 1/(3*m*n) * sum(sum((immagine_originale(:, :,i)- immagine_compressa_ROI(:,:, i)).^2));
end
MSE_ROI = sum(MSE_ROI);
pnsr_ROI = 10 * log10(MAX_I^2 ./ MSE_ROI )

for i=1:3
    MSE(i) = 1/(3*m*n) * sum(sum((immagine_originale(:, :,i)- immagine_compressa(:,:, i)).^2));
end
MSE = sum(MSE);
pnsr = 10 * log10(MAX_I^2 ./ MSE )

for i=1:3
    MSE_jpeg(i) = 1/(3*m*n) * sum(sum((immagine_originale(:, :,i)- immagine_compressa_jpeg(:,:, i)).^2));
end
MSE_jpeg = sum(MSE_jpeg);
pnsr_jpeg = 10 * log10(MAX_I^2 ./ MSE_jpeg )