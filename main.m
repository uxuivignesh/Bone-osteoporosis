clc;    % Clear the command window.
 warning off;
workspace;  % Make sure the workspace panel is showing.
format long g;
format compact;
fontSize = 10;



%i/p image get in section

% Select an image from the 'Disease Dataset' folder by opening the folder
[filename,pathname] = uigetfile({'*.*';'*.bmp';'*.tif';'*.jpg';'*.png'},'Pick a Disease Affected image');
rgbImage = imread([pathname,filename]);

disp('IMAGE LOADED');
pause(1);
%preprocessing
i=rgbImage;
pause(2);
gray=i;
% gray = rgb2gray(rgbImage);
J = histeq(gray);

disp('PRE PROCESSING STARTS');
pause(1);
figure;
imshow(rgbImage);
title('Original Image');
figure;
imshow(J);
title('Enhanced Image');
figure;
imshowpair(gray,J,'montage');
title('Enhanced Pair Image');
axis on;
figure;
subplot(1,2,1);
imhist(gray,64);
title('Original Image Histogram');
axis on;

disp('STRUCTURAL DATA ANALYSIS');
pause(1);
thresh = multithresh(gray,2);
seg_I = imquantize(gray,thresh);
RGB = label2rgb(seg_I);
grey1 = rgb2gray(RGB);
subplot(1,2,2);
imhist(grey1,64);
title('Enhanced Image Histogram');
figure;
imshow(RGB);
title('RGB Segmented Image');
axis off

figure
subplot(331)
imshow(i)
title('Original Image')
I1=i;% I1=rgb2gray(i);
subplot(332)
imshow(I1)
title('Grayscale image')

A = adapthisteq(I1,'clipLimit',0.02,'Distribution','rayleigh');

subplot(333)
imshow(A)
title('Contrast Enhanced image')

subplot(334)
I3 = threshold(A);
imshow(I3);
title('Thresholded image')


subplot(335)
I4=threshold(I1);
imshow(I4)
title('contrast segment image')

subplot(336)
imshow(RGB);
title('RGB SEGMENTED IMAGE');

%segmentation

% Get the dimensions of the image.  numberOfColorBands should be = 3.

I = imresize(rgbImage,[500 700]);
greenc = I1;                          % Extract Green Channel
ginv = imcomplement (greenc);               % Complement the Green Channel
adahist = adapthisteq(ginv);                % Adaptive Histogram Equalization
se = strel('ball',8,8);                     % Structuring Element
gopen = imopen(adahist,se);                 % Morphological Open
godisk = adahist - gopen;                   % Remove Optic Disk

medfilt = medfilt2(godisk);                 %2D Median Filter
background = imopen(medfilt,strel('disk',15));% imopen function
I2 = medfilt - background;                  % Remove Background
I3 = imadjust(I2);                          % Image Adjustment

level = graythresh(I3);                     % Gray Threshold
bw = im2bw(I3,level);                       % Binarization
bw = bwareaopen(bw, 30);                    % Morphological Open
% figure,imshow(bw);

wname = 'sym4';
[CA,CH,CV,CD] = dwt2(bw,wname,'mode','per');
figure,imshow(CA),title('LINE B/W');

b = bwboundaries(bw);
% axes(handles.axes5);
% I = imresize(I,[500 752]);
figure,imshow(I);
title('LINE TRACE');

hold on
for k = 1:numel(b)
    plot(b{k}(:,2), b{k}(:,1), 'b', 'Linewidth', 1)
end

g=I1;
rsz = imresize(g,[500 700]);
a=adapthisteq(rsz);
o=strel('ball',8,8);
s=imopen(a,o);
se=a-s;
ad=adapthisteq(se);
im=imadjust(ad,[],[],6);
ad1=adapthisteq(im);
bw=im2bw(ad1,0.1);
ar = bwarea(bw);
ar = round(ar);
wname = 'sym4';
[CA,CH,CV,CD] = dwt2(bw,wname,'mode','per');
figure, imshow(CA);
title('AFFECT PARTS');

hold on;
%data extracting

disp('DATA EXTRACTING');
pause(1);

gray=rgb2gray(RGB);

m = mean(gray,2); % Computing the average face image m = (1/P)*sum(Tj's)    (j = 1 : P)

AA=min(max(m));

D=mean(mean(abs(gray)));
E = entropy(gray);
Std = std2(gray);
figure;
plot(m);
xlabel('NUMBER OF INSTANCE');
ylabel('VALUE OF INSTANCE');
title('INPUT IMAGE MEAN VALUE');
disp('DEVIATION');
disp(D);
disp('MAXIMUM PIXEL IMAGE DATA  ');
disp(AA);
disp('ENTROPY');
disp(E);
disp('STANDARD DEVIATION');
disp(Std);

figure;

bar(1,D,'r');
hold on
bar(2,AA,'g');
hold on
bar(3,E,'y');
hold on
bar(4,Std,'m');
hold off
legend ('DEVIATION','MAXIMUM PIXEL','ENTROPY','STANDARD DEVIATION');

title('IMAGE ANALYSIS DATA');


odprz=imresize(rgbImage,[300 300]);
imwrite(odprz,'oq1.png');
pic1 = imread('oq1.png');  

er=0;
[x,y,z] = size(pic1);
if(z==1)
     ;
else
    pic1 = rgb2gray(pic1);
    imwrite(pic1,'gray1.png')
    end

bw1=im2bw(pic1,.6);

    
normal = 0;
affected = 0;
x=0;
y=0;
l=0;
m=0;

data1=xlsread('features');
distr='kernel';
X = double(data1(:,1:4));
Y = double(data1(:,5));
X1=X;
X2=Y;

P = X1';
T = X2';
net =neurofuzzy(P,T);
Y = sim(net,P);


for a = 1:1:256
    for b = 1:1:256
        if(bw1(a,b)==1)
            affected =(affected+1);
        else
            normal = (normal+1);
        end
    end
end

disp('AFFECTED PIXEL POINTS:');
    pause(1);
disp(affected);
    

disp('NORMAL PIXEL POINTS:');
   
pause(1);
disp(normal);
   

%decision tree unit
if(affected <= 290)
    disp('INPUT IMAGE NORMAL');
    
pause(1);
msgbox('INPUT IMAGE NORMAL')
else
    disp('Osteo Problem Occured'); 
   
pause(1);
msgbox('Osteo Problem Occured')

 
end


disp('PROCESS COMPLETED')
pause(1);
