clc;
clear all;
folder = 'kodak_dataset/image_';
image = int2str(1);
filename = [folder, image, '/original_image.png'];
I = imread(filename);
fileinfo = dir(filename);
size_original_image = fileinfo(1).bytes;
    
[row, coln, depth]= size(I);

%% we create three different matrixes, one for each color 
red = double(I(:,:,1));
green = double(I(:,:,2));
blue = double(I(:,:,3)); 

red1 = red;
green1 = green;
blue1 = blue;

quality = 25;%input('What quality of compression you require - ');


%% compression

red = red - (128*ones(row, coln));
green = green - (128*ones(row, coln));
blue = blue - (128*ones(row, coln));

%% quality matrix formulation
Q50 = [ 16 11 10 16 24 40 51 61;
     12 12 14 19 26 58 60 55;
     14 13 16 24 40 57 69 56;
     14 17 22 29 51 87 80 62; 
     18 22 37 56 68 109 103 77;
     24 35 55 64 81 104 113 92;
     49 64 78 87 103 121 120 101;
     72 92 95 98 112 100 103 99];
 
 if quality > 50
     QX = round(Q50.*(ones(8)*((100-quality)/50)));
     QX = uint8(QX);
 elseif quality < 50
     QX = round(Q50.*(ones(8)*(50/quality)));
     QX = uint8(QX);
 elseif quality == 50
     QX = Q50;
 end

%% Formulation of forward DCT Matrix and inverse DCT matrix

DCT_matrix8 = dct(eye(8));
iDCT_matrix8 = DCT_matrix8';   %inv(DCT_matrix8);


%% Jpeg Compression

dct_restored = zeros(row,coln);
QX = double(QX);

%% red matrix compression

%Forward Discret Cosine Transform

for i1=[1:8:row]
    for i2=[1:8:coln]
        zBLOCK=red(i1:i1+7,i2:i2+7);
        win1=DCT_matrix8*zBLOCK*iDCT_matrix8;
        dct_domain(i1:i1+7,i2:i2+7)=win1;
    end
end

%Quantization of the DCT coefficients

for i1=[1:8:row]
    for i2=[1:8:coln]
        win1 = dct_domain(i1:i1+7,i2:i2+7);
        win2=round(win1./QX);
        dct_quantized(i1:i1+7,i2:i2+7)=win2;
    end
end

% Dequantization of DCT Coefficients

for i1=[1:8:row]
    for i2=[1:8:coln]
        win2 = dct_quantized(i1:i1+7,i2:i2+7);
        win3 = win2.*QX;
        dct_dequantized(i1:i1+7,i2:i2+7) = win3;
    end
end

% Inverse DISCRETE COSINE TRANSFORM
for i1=[1:8:row]
    for i2=[1:8:coln]
        win3 = dct_dequantized(i1:i1+7,i2:i2+7);
        win4=iDCT_matrix8*win3*DCT_matrix8;
        dct_restored(i1:i1+7,i2:i2+7)=win4;
    end
end
red2 = dct_restored;
red2 = red2 + 128*ones(row, coln);

%% green matrix compression
%Forward Discret Cosine Transform

for i1=[1:8:row]
    for i2=[1:8:coln]
        zBLOCK=green(i1:i1+7,i2:i2+7);
        win1=DCT_matrix8*zBLOCK*iDCT_matrix8;
        dct_domain(i1:i1+7,i2:i2+7)=win1;
    end
end

%Quantization of the DCT coefficients

for i1=[1:8:row]
    for i2=[1:8:coln]
        win1 = dct_domain(i1:i1+7,i2:i2+7);
        win2=round(win1./QX);
        dct_quantized(i1:i1+7,i2:i2+7)=win2;
    end
end

% Dequantization of DCT Coefficients

for i1=[1:8:row]
    for i2=[1:8:coln]
        win2 = dct_quantized(i1:i1+7,i2:i2+7);
        win3 = win2.*QX;
        dct_dequantized(i1:i1+7,i2:i2+7) = win3;
    end
end

% Inverse DISCRETE COSINE TRANSFORM
for i1=[1:8:row]
    for i2=[1:8:coln]
        win3 = dct_dequantized(i1:i1+7,i2:i2+7);
        win4=iDCT_matrix8*win3*DCT_matrix8;
        dct_restored(i1:i1+7,i2:i2+7)=win4;
    end
end
green2=dct_restored;
green2 = green2 + 128*ones(row, coln);

%% blue matrix compression
%Forward Discret Cosine Transform

for i1=[1:8:row]
    for i2=[1:8:coln]
        zBLOCK=blue(i1:i1+7,i2:i2+7);
        win1=DCT_matrix8*zBLOCK*iDCT_matrix8;
        dct_domain(i1:i1+7,i2:i2+7)=win1;
    end
end

%Quantization of the DCT coefficients

for i1=[1:8:row]
    for i2=[1:8:coln]
        win1 = dct_domain(i1:i1+7,i2:i2+7);
        win2=round(win1./QX);
        dct_quantized(i1:i1+7,i2:i2+7)=win2;
    end
end

% Dequantization of DCT Coefficients

for i1=[1:8:row]
    for i2=[1:8:coln]
        win2 = dct_quantized(i1:i1+7,i2:i2+7);
        win3 = win2.*QX;
        dct_dequantized(i1:i1+7,i2:i2+7) = win3;
    end
end

% Inverse DISCRETE COSINE TRANSFORM
for i1=[1:8:row]
    for i2=[1:8:coln]
        win3 = dct_dequantized(i1:i1+7,i2:i2+7);
        win4=iDCT_matrix8*win3*DCT_matrix8;
        dct_restored(i1:i1+7,i2:i2+7)=win4;
    end
end
blue2=dct_restored;
blue2 = blue2 + 128*ones(row, coln);


%% recomposition of the original image
compressed_image = cat(3, red2, green2, blue2);
compressed_image = uint8(compressed_image);

imshow(compressed_image);
compressed_file_name = [folder, image, '/compressed_image_jpeg.png'];
imwrite(compressed_image, compressed_file_name);
compressed_fileinfo = dir(compressed_file_name);
size_compressed_image = compressed_fileinfo(1).bytes;

size_compressed_image/size_original_image


