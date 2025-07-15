% Author:   Cody Costa
% Date:     7/15/2025


%% Images to use for assignment

LENA = 'lena.jpg';
BONES = 'Fig3-46a.tif';
MRI = 'Image9.jpg';


%% PROBLEM 1: Load and display GS and Color img
img = imread(LENA);
figure;
imshow(img)

gray = rgb2gray(img);
figure;
imshow(gray)


%% PROBLEM 2: Sobel and Prewitt edge detection
gray = rgb2gray(imread(LENA));

% Sobel
sx = [-1 0 1; 
      -2 0 2; 
      -1 0 1];
sy = sx';

Gx = conv2(double(gray), sx, 'same');
Gy = conv2(double(gray), sy, 'same');

G_mag = sqrt(Gx.^2 + Gy.^2);
edge_sobel = uint8(255 * mat2gray(G_mag));

figure;
imshow(edge_sobel)
title('Sobel Edge');

% Prewitt
px = [-1 0 1;
      -1 0 1;
      -1 0 1];
py = px';

Px = conv2(double(gray), px, 'same');
Py = conv2(double(gray), py, 'same');

P_mag = sqrt(Px.^2 + Py.^2);
edge_prewitt = uint8(255 * mat2gray(P_mag));

figure;
imshow(edge_prewitt)
title('Prewitt Edge');


%% PROBLEM 3: Image Process Pipeline
gray = imread(BONES);
G = im2double(gray);
figure;
imshow(G)
title('Base Gray Img')

% take laplacian of image
laplacianKernel = [0 1 0; 
                   1 -4 1; 
                   0 1 0];
Laplacian = imfilter(G, laplacianKernel, "replicate");
figure;
imshow(Laplacian, [])
title('Laplacian')

% sharpen image by subtracting laplacian from original
sharpened = G - Laplacian;
figure;
imshow(sharpened, [])
title('Sharpened')

% apply Sobel edge to original image
sobelBinary = edge(G, "sobel");
figure;
imshow(sobelBinary, [])
title('Sobel Edge')

% apply 5x5 blur to sobel edge image
avgKernel = fspecial('average', [5 5]);
smoothedBinary = imfilter(double(sobelBinary), avgKernel, 'replicate');
figure;
imshow(smoothedBinary, [])
title('Smoothed')

% multiply sharpened with blurred
Mask = sharpened .* smoothedBinary;
figure;
imshow(Mask, [])
title('Product')

% sum of original img with Mask
SharpMask = G + Mask;
figure;
imshow(SharpMask, [])
title('Sharp Mask')

% power transform to Sharpened Mask photo
gamma = 0.4;
c = 1.0;

IMG_GAMMA = c * (SharpMask .^ gamma);
figure;
imshow(IMG_GAMMA, [])
title('Gamma Transform')


%% PROBLEM 4: Image Process Pipeline
gray = imread(MRI);
G = im2double(gray);
figure;
imshow(G)
title('Base Gray Img')

% take laplacian of image
laplacianKernel = [0 1 0; 
                   1 -4 1; 
                   0 1 0];
Laplacian = imfilter(G, laplacianKernel, "replicate");
figure;
imshow(Laplacian, [])
title('Laplacian')

% sharpen image by subtracting laplacian from original
sharpened = G - Laplacian;
figure;
imshow(sharpened, [])
title('Sharpened')

% apply Sobel edge to original image
sobelBinary = edge(G, "sobel");
figure;
imshow(sobelBinary, [])
title('Sobel Edge')

% apply 5x5 blur to sobel edge image
avgKernel = fspecial('average', [5 5]);
smoothedBinary = imfilter(double(sobelBinary), avgKernel, 'replicate');
figure;
imshow(smoothedBinary, [])
title('Smoothed')

% multiply sharpened with blurred
Mask = sharpened .* smoothedBinary;
figure;
imshow(Mask, [])
title('Product')

% sum of original img with Mask
SharpMask = G + Mask;
figure;
imshow(SharpMask, [])
title('Sharp Mask')

% power transform to Sharpened Mask photo
gamma = 0.4;
c = 1.0;

IMG_GAMMA = c * (SharpMask .^ gamma);
figure;
imshow(IMG_GAMMA, [])
title('Gamma Transform')


%% close all figures
close all