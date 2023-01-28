%% We're going to generate obliquely illuminated images of an object and sample them at 20 times loss of resolution
%% Images chosen are cameraman.tif for intensity and westconcordorthophoto.png from imdata

% First we load constants
clearvars;
clc;

system_constants;

f1 = dir('/home/legion/nownow/fourier_ptychography/data/jpg1/');
f1 = f1(3:end);
f2 = dir('/home/legion/nownow/fourier_ptychography/data/jpg2/');
f2 = f2(3:end);

files = [];
for i = 1:length(f1)
files = [files; [f1(i).folder '/' f1(i).name]];
end
for i = 1:length(f2)
files = [files; [f2(i).folder '/' f2(i).name]];
end
files = files(randperm(length(files)),:);

n = length(files) - 1;
n_train = floor(n*0.8);
n_test = n - n_train;

stacks_train = zeros(3000,64,64,25);
targets_train = zeros(3000,256,256,2);

disp("Creating Train Set");

lpf_mask = maskk(0, 0, 2*pupil_radius, 256*upsample, 256*upsample);
n_ints = randi(n_train,3000);
n_phases = randi(n_train,3000);

N = 2*illumination_layers - 1;   % side of square of illumination matrix

% imaged_images = zeros(64,64,N^2);

for i = 1:1
    tic;
    disp(i);
    
%     n_int = randi(n_train);
%     n_phase = randi(n_train);
    
    intensity_image = double(imresize(rgb2gray(imread(files(n_ints(i),:))),[256,256]));
%     intensity_image = 255.0*ones(256,256);
    phase_image = double(imresize(rgb2gray(imread(files(n_phases(i),:))),[256,256]));

    targets_train(i,:,:,1) = intensity_image/255.0;
    targets_train(i,:,:,2) = phase_image/255.0;
    
    intensity_image = imresize(intensity_image, upsample);
    phase_image = imresize(phase_image, upsample);
    
    % We generate the simulated object
    phase_image = 2*(pi/256.0).*phase_image;   % normalize phase image to [0, pi] to prevent loss of information in euler's decomposition of exponent
%     phase_image = zeros(size(phase_image));    % for ease of working, have a constant phase to begin with
    object = intensity_image.*exp(1j*phase_image);

%     object = 255.0*ifft2(ifftshift(ones(1280,1280)));
    %% Illuminate the object at different angles, save images formed
    %% we choose the origin to be the center of the image, for ease of calculation of the illumination wavevector
    %% The light source is assumed to be an LED matrix centered at origin - the origin being at the center of an LED. We will always illuminate with an odd-numbered square of LEDs, with one always at the center

    imaged_images = [];
    illuminated_objects = [];
    for a = 1:N
        for b = 1:N
            % illumination is done in layers
            % we will calculate the x,y coordinates of the current LED
            x = (a - illumination_layers)*LED_spacing;
            y = (b - illumination_layers)*LED_spacing;

            % illuminate it
            illuminated_object = illuminate(object, x, y, object_x, object_y, illumination_distance, wave_number);
%             fff = abs(fftshift(fft2(illuminated_object)));
            % imshow(fff,[]);
%             dif = abs(illuminated_object) - abs(10*object);
            illuminated_objects = cat(3,illuminated_objects,illuminated_object);
            %% Next, the object thus illuminated is IMAGED by an imaging sysem with given NA, which will behave like an LPF for spatial frequencies.
            imaged_image = imageit(illuminated_object, initial_px, sampled_px, lpf_mask);
            % figure; imshow(abs(imaged_image), []);

            % scale the images to [0 255] and save them either as images or append them to an array of images
%             imaged_image = 255*(imaged_image - min(imaged_image(:)))./(max(max(abs(imaged_image))));
            imaged_image = imaged_image - min(imaged_image(:));
            imaged_image = abs(imaged_image)./(max(abs(imaged_image(:))));
%             imaged_image = imaged_image.*conj(imaged_image);
%             imaged_image = imaged_image./(max(imaged_image(:)));
            imaged_images = cat(3, imaged_images, imaged_image);
%             imaged_images(:,:,N*(a-1) + b) = imaged_image;

            % fileName = sprintf('%s%i%i%s',folder,a,b,'.tif');
            % imwrite(uint8(imaged_image), fileName);
        end
    end
%     max(max(max(imaged_images)))
    stacks_train(i,:,:,:) = imaged_images;
    toc;
end

% stacks_test = zeros(600,64,64,25);
% targets_test = zeros(600,256,256,2);
% 
% n_ints = n + 1 - randi(n_test,600);
% n_phases = n + 1 - randi(n_test,600);
% 
% disp("Creating Test Set");
% for i = 1:1
%     tic;
%     disp(i);
%     
% %     n_int = randi(n_test);
% %     n_phase = randi(n_test);
%     
%     intensity_image = double(imresize(rgb2gray(imread(files(n_ints(i),:))),[256,256]));
%     phase_image = double(imresize(rgb2gray(imread(files(n_phases(i),:))),[256,256]));
% 
%     targets_test(i,:,:,1) = intensity_image/255.0;
%     targets_test(i,:,:,2) = phase_image/255.0;
%     
%     intensity_image = imresize(intensity_image, upsample);
%     phase_image = imresize(phase_image, upsample);
%     
%     % We generate the simulated object
%     phase_image = 2*(pi/256.0).*phase_image;   % normalize phase image to [0, pi] to prevent loss of information in euler's decomposition of exponent
% %     phase_image = zeros(size(phase_image));    % for ease of working, have a constant phase to begin with
%     object = intensity_image.*exp(1j*phase_image);
% 
%     %% Illuminate the object at different angles, save images formed
%     %% we choose the origin to be the center of the image, for ease of calculation of the illumination wavevector
%     %% The light source is assumed to be an LED matrix centered at origin - the origin being at the center of an LED. We will always illuminate with an odd-numbered square of LEDs, with one always at the center
%     N = 2*illumination_layers - 1;   % side of square of illumination matrix
% 
%     imaged_images = [];
%     for a = 1:N
%         for b = 1:N
%             % illumination is done in layers
%             % we will calculate the x,y coordinates of the current LED
%             x = (a - illumination_layers)*LED_spacing;
%             y = (b - illumination_layers)*LED_spacing;
% 
%             % illuminate it
%             illuminated_object = illuminate(object, x, y, object_x, object_y, illumination_distance, wave_number);
% %             fff = abs(fftshift(fft2(illuminated_object)));
%             % imshow(fff,[]);
% 
%             %% Next, the object thus illuminated is IMAGED by an imaging sysem with given NA, which will behave like an LPF for spatial frequencies.
%             imaged_image = imageit(illuminated_object, initial_px, sampled_px, lpf_mask);
%             % figure; imshow(abs(imaged_image), []);
% 
%             % scale the images to [0 255] and save them either as images or append them to an array of images
% %             imaged_image = 255*(imaged_image - min(imaged_image(:)))./(max(max(abs(imaged_image))));
%             imaged_image = imaged_image - min(imaged_image(:));
%             imaged_image = abs(imaged_image)./(max(abs(imaged_image(:))));
% %             imaged_image = imaged_image.*conj(imaged_image);
% %             imaged_image = imaged_image./(max(imaged_image(:)));
%             imaged_images = cat(3, imaged_images, imaged_image);
% %             imaged_images(:,:,N*(a-1) + b) = imaged_image;
% 
%             % fileName = sprintf('%s%i%i%s',folder,a,b,'.tif');
%             % imwrite(uint8(imaged_image), fileName);
%         end
%     end
%     stacks_test(i,:,:,:) = imaged_images;
%     toc;
% end
% 
% 
% % targets_train = zeros(length(stacks_train),256,256,2);
% % for i = 1:2:length(files)-1
% %     tic;
% %     disp((i+1)/2);
% %     
% %     intensity_image = double(imresize(rgb2gray(imread(files(i,:))),[256,256]));
% %     phase_image = double(imresize(rgb2gray(imread(files(i+1,:))),[256,256]));
% %     
% %     targets_train((i+1)/2,:,:,1) = intensity_image/255.0;
% %     targets((i+1)/2,:,:,2) = phase_image/255.0;
% %     toc;
% % end
% 
% initialSpect = fftshift(fft2(ifftshift(stacks_train(1,:,:,:)))); % broadSpectrum, can begin with any image.
% initialNew = initialSpect;
% c=1;
% % FPM Algorith            m
% % while c<2  % Repeating the whole process 2 times
% %     for i = 1:numberOfMasks
% %         
% %         masked = initialNew.*mask(:,:,i);
% %         inverse = (fftshift(ifft2(ifftshift(masked))));
% %         inverseNew = abs(lowRes(:,:,i)) .* exp(1i*angle(inverse));
% %         newSpectrum = fftshift(fft2(ifftshift(inverseNew)));
% %         % stacks_test = zeros(600,64,64,25);
% targets_test = zeros(600,256,256,2);
% 
% n_ints = n + 1 - randi(n_test,600);
% n_phases = n + 1 - randi(n_test,600);
% 
% disp("Creating Test Set");
% for i = 1:1
%     tic;
%     disp(i);
%     
% %     n_int = randi(n_test);
% %     n_phase = randi(n_test);
%     
%     intensity_image = double(imresize(rgb2gray(imread(files(n_ints(i),:))),[256,256]));
%     phase_image = double(imresize(rgb2gray(imread(files(n_phases(i),:))),[256,256]));
% 
%     targets_test(i,:,:,1) = intensity_image/255.0;
%     targets_test(i,:,:,2) = phase_image/255.0;
%     
%     intensity_image = imresize(intensity_image, upsample);
%     phase_image = imresize(phase_image, upsample);
%     
%     % We generate the simulated object
%     phase_image = 2*(pi/256.0).*phase_image;   % normalize phase image to [0, pi] to prevent loss of information in euler's decomposition of exponent
% %     phase_image = zeros(size(phase_image));    % for ease of working, have a constant phase to begin with
%     object = intensity_image.*exp(1j*phase_image);
% 
%     %% Illuminate the object at different angles, save images formed
%     %% we choose the origin to be the center of the image, for ease of calculation of the illumination wavevector
%     %% The light source is assumed to be an LED matrix centered at origin - the origin being at the center of an LED. We will always illuminate with an odd-numbered square of LEDs, with one always at the center
%     N = 2*illumination_layers - 1;   % side of square of illumination matrix
% 
%     imaged_images = [];
%     for a = 1:N
%         for b = 1:N
%             % illumination is done in layers
%             % we will calculate the x,y coordinates of the current LED
%             x = (a - illumination_layers)*LED_spacing;
%             y = (b - illumination_layers)*LED_spacing;
% 
%             % illuminate it
%             illuminated_object = illuminate(object, x, y, object_x, object_y, illumination_distance, wave_number);
% %             fff = abs(fftshift(fft2(illuminated_object)));
%             % imshow(fff,[]);
% 
%             %% Next, the object thus illuminated is IMAGED by an imaging sysem with given NA, which will behave like an LPF for spatial frequencies.
%             imaged_image = imageit(illuminated_object, initial_px, sampled_px, lpf_mask);
%             % figure; imshow(abs(imaged_image), []);
% 
%             % scale the images to [0 255] and save them either as images or append them to an array of images
% %             imaged_image = 255*(imaged_image - min(imaged_image(:)))./(max(max(abs(imaged_image))));
%             imaged_image = imaged_image - min(imaged_image(:));
%             imaged_image = abs(imaged_image)./(max(abs(imaged_image(:))));
% %             imaged_image = imaged_image.*conj(imaged_image);
% %             imaged_image = imaged_image./(max(imaged_image(:)));
%             imaged_images = cat(3, imaged_images, imaged_image);
% %             imaged_images(:,:,N*(a-1) + b) = imaged_image;
% 
%             % fileName = sprintf('%s%i%i%s',folder,a,b,'.tif');
%             % imwrite(uint8(imaged_image), fileName);
%         end
%     end
%     stacks_test(i,:,:,:) = imaged_images;
%     toc;
% end
% 
% 
% % targets_train = zeros(length(stacks_train),256,256,2);
% % for i = 1:2:length(files)-1
% %     tic;
% %     disp((i+1)/2);
% %     
% %     intensity_image = double(imresize(rgb2gray(imread(files(i,:))),[256,256]));
% %     phase_image = double(imresize(rgb2gray(imread(files(i+1,:))),[256,256]));
% %     
% %     targets_train((i+1)/2,:,:,1) = intensity_image/255.0;
% %     targets((i+1)/2,:,:,2) = phase_image/255.0;
% %     toc;
% % end
% 
% initialSpect = fftshift(fft2(ifftshift(stacks_train(1,:,:,:)))); % broadSpectrum, can begin with any image.
% initialNew = initialSpect;
% c=1;
% % FPM Algorith            m
% % while c<2  % Repeating the whole process 2 times
% %     for i = 1:numberOfMasks
% %         
% %         masked = initialNew.*mask(:,:,i);
% %         inverse = (fftshift(ifft2(ifftshift(masked))));
% %         inverseNew = abs(lowRes(:,:,i)) .* exp(1i*angle(inverse));
% %         newSpectrum = fftshift(fft2(ifftshift(inverseNew)));
% %         
% %         for a=1:x
% %             for b=1:y
% %                 if mask(a,b,i)~=0
% %                     initialNew(a,b) = newSpectrum(a,b); % Updating 
% %                 end
% %             end
% %         endstack
% %     end
% %     c=c+1;
% %     final = fftshift(ifft2(ifftshift(initialNew)));   

% %         for a=1:x
% %             for b=1:y
% %                 if mask(a,b,i)~=0
% %                     initialNew(a,b) = newSpectrum(a,b); % Updating 
% %                 end
% %             end
% %         endstack
% %     end
% %     c=c+1;
% %     final = fftshift(ifft2(ifftshift(initialNew)));   
