originalImage = imread('Fig1051(a)(defective_weld).tiff');

%spec_origin = fft2(double(originalImage));
%spec_img = fftshift(spec_origin);

figure;
%spec_img = log(1+abs(spec_img));
imshow(originalImage,[]);