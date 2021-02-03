clear all

%% upload of the original image and its heatmap
filename = '/immagine_prova.png';
original_image = imread(filename);
original_image = double(original_image);
importance_matrix = readtable('/roi_map.csv', 'ReadVariableNames',0 );

max(max(original_image))
min(min(original_image))


quality = 0;
while(quality<1 || quality > 10)
    quality = input('What quality of compression you require? \n(1= high compression rate, low quality, 10 = high quality, low compression rate):  ');
end
tic;
% make the original image squared
original_size = size(original_image);
max_size = max(original_size(1), original_size(2));
if (original_size(1) < original_size(2) )
    missing_rows = original_size(2) - original_size(1);
    new_original_image = zeros(original_size(2), original_size(2), 3);
    new_original_image((missing_rows+1): original_size(2), :, :) = original_image;
    original_image = new_original_image;
elseif (original_size(1) > original_size(2) )
    missing_columns = original_size(1) - original_size(2);
    new_original_image = zeros(original_size(1), original_size(1), 3);
    new_original_image(:, (missing_columns+1): original_size(1), :) = original_image;
    original_image = new_original_image;
end

importance_matrix = importance_matrix{:, :};


size_importance_map = length(importance_matrix(1, :));

%% computation of the importance of each zone of the image
areas_per_side = floor(46/9 * quality + 26/9);        % min value = 10, max value = 56
zone_importance = zeros(1, areas_per_side^2);


%dimension_zones = size_importance_map/areas_per_side;


k = mod(size_importance_map, areas_per_side);
area_size = floor(size_importance_map/areas_per_side);
vettore_segmenti_importance_map = zeros(2, areas_per_side);
ampiezza_importance_map = zeros(1, areas_per_side);
inf_lim = 0;
right_lim = 0;
if(mod(k, 2)==0)
    for i=1:(areas_per_side)
        if (i<=(k/2) || i> (areas_per_side - k/2))
            left_lim = right_lim+1;
            right_lim = right_lim + (area_size +1);
            vettore_segmenti_importance_map(:, i) =[left_lim; right_lim];
        else
            left_lim = right_lim+1;
            right_lim = right_lim + (area_size);
            vettore_segmenti_importance_map(:, i) =[left_lim; right_lim];
        end
        ampiezza_importance_map(1, i) = right_lim - left_lim +1;
    end
else
    % k dispari
     for i=1:areas_per_side
        if ( i < (k+1)/2 || i > areas_per_side - (k+1)/2 )
            left_lim = right_lim+1;
            right_lim = right_lim + (area_size +1);
            vettore_segmenti_importance_map(:, i) =[left_lim; right_lim];
        else
            left_lim = right_lim+1;
            right_lim = right_lim + (area_size);
            vettore_segmenti_importance_map(:, i) =[left_lim; right_lim];
        end
        ampiezza_importance_map(1, i) = right_lim - left_lim +1;
    end
end
vector_zones = zeros(area_size+1, area_size+1, 3, areas_per_side^2);
vettore_segmenti_importance_map;
ampiezza_importance_map;
count = 0;
for i = 1:areas_per_side
    for j= 1:areas_per_side
        count = count+1;
        zone = importance_matrix(vettore_segmenti_importance_map(1, i):vettore_segmenti_importance_map(2, i), vettore_segmenti_importance_map(1, j):vettore_segmenti_importance_map(2, j) );
        zone_importance (count) = mean(zone, 'all');
        % vector_zones(1:ampiezza_importance_map(i), 1:ampiezza_importance_map(j), :, count) = original_image( vettore_segmenti(1, i):vettore_segmenti(2, i), vettore_segmenti(1, j):vettore_segmenti(2, j), :);
    end
end
zone_importance;

%% division of the original image into zones
dimension = size(original_image);
dimension = dimension(1);

k = mod(dimension, areas_per_side);
area_size = floor(dimension/areas_per_side);
vettore_segmenti = zeros(2, areas_per_side);
ampiezza = zeros(1, areas_per_side);
inf_lim = 0;
right_lim = 0;
if(mod(k, 2)==0)
    for i=1:(areas_per_side)
        if (i<=(k/2) || i> (areas_per_side - k/2))
            left_lim = right_lim+1;
            right_lim = right_lim + (area_size +1);
            vettore_segmenti(:, i) =[left_lim; right_lim];
        else
            left_lim = right_lim+1;
            right_lim = right_lim + (area_size);
            vettore_segmenti(:, i) =[left_lim; right_lim];
        end
        ampiezza(1, i) = right_lim - left_lim +1;
    end
else
    % k dispari
     for i=1:areas_per_side
        if ( i < (k+1)/2 || i > areas_per_side - (k+1)/2 )
            left_lim = right_lim+1;
            right_lim = right_lim + (area_size +1);
            vettore_segmenti(:, i) =[left_lim; right_lim];
        else
            left_lim = right_lim+1;
            right_lim = right_lim + (area_size);
            vettore_segmenti(:, i) =[left_lim; right_lim];
        end
        ampiezza(1, i) = right_lim - left_lim +1;
    end
end
vector_zones = zeros(area_size+1, area_size+1, 3, areas_per_side^2);
vettore_segmenti;
ampiezza;
count = 0;
for i = 1:areas_per_side
    for j= 1:areas_per_side
        count = count+1;
        vector_zones(1:ampiezza(i), 1:ampiezza(j), :, count) = original_image( vettore_segmenti(1, i):vettore_segmenti(2, i), vettore_segmenti(1, j):vettore_segmenti(2, j), :);
    end
end
%end

%% vanno trovati i ranghi adatti a ciascuna zona, mettendo in relazione la dimensione (area_size) e l'importanza di ciascuna zona

% area_size = (dimension image)/areas_per_side = (dimension_image)/n
final_rank = (- quality^2 * 3/40 + quality * 33/40 - 1/4) * ( area_size/8 * tanh(2*(zone_importance-0.5)) + area_size/8 ) ;
if (quality > 5)
    final_rank = final_rank + area_size/4 * ((quality-5)^2) /10 ;
end
final_rank = uint8(final_rank);

%% si applica la hosvd a ciascuna zona, usando i ranghi precedentemente stabiliti
% hosvd usando [final_rank(i), final_rank(i), 3] come rango finale di
% ciascuna sezione
count = 0;
compressed_image = zeros(size(original_image));
for i = 1:areas_per_side
    for j=1:areas_per_side
        count = count+1;
        section_to_compress = vector_zones(1:ampiezza(i), 1:ampiezza(j), :, count); 
        % se non funziona mettere double
        [U, S] = mlsvd(section_to_compress, [final_rank(count), final_rank(count), 3]);
        [final_rank(count), final_rank(count), 3]
        fprintf("dimensione di S")
        size(S)
        fprintf("dimensione immagine da comprimere")
        size(section_to_compress)
        section_compressed = tmprod(S, {U{1}, U{2}, U{3}}, 1:3);
        compressed_image( vettore_segmenti(1, i):vettore_segmenti(2, i), vettore_segmenti(1, j):vettore_segmenti(2, j), :) = section_compressed;
        %compressed_image( ((i-1)*area_size + 1):(i*area_size), ((j-1)*area_size + 1): (j* area_size), :) = section_compressed;
    end
end
time = toc;
time
%% stampa dell'immagine compressa
if (abs(original_size(1) - original_size(2)) > 0)
    compressed_image = compressed_image((max_size - original_size(1) + 1):max_size, (max_size - original_size(2) + 1):max_size, :);
end
imshow(uint8(compressed_image));
imwrite(uint8(compressed_image), '/compressed_image_ROI_first_method.png');

