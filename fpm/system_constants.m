%% ALL DIMENSIONS ARE IN MICRONS
clear all;

% sample object constants
initial_px = 0.275;     % initial pixel size in microns
upsample = 5;
% intensity_image = double(imresize(imresize(rgb2gray(imread('/home/nownow/Documents/projects/fourier_ptychography/data/jpg1/100000.jpg')),[256,256]), upsample));
% phase_image = double(imresize(imresize(rgb2gray(imread('/home/nownow/Documents/projects/fourier_ptychography/data/jpg1/104100.jpg')), [256 256]), upsample));
object_length = 0.275*256*upsample;     % length (and width) in microns
[object_x, object_y] = meshgrid(-object_length/2:0.275:object_length/2 - 0.275, -object_length/2:0.275:object_length/2 - 0.275);

% Illumination system constants
wavelength = 0.632;     % wavelength of light used for simulated illumination, in microns
wave_number = 2*pi/wavelength;
LED_spacing = 4000;    % vertical and horizontal distance between center of each LED in the array, in microns
illumination_distance = 80000;  % distance from the LED matrix to the object, in microns
illumination_layers = 3; % no of layers of a spiral square matrix of LEDs used to illuminate the setup

% Imaging system constants
sampled_px = 5.5;
NA = 0.08;              % Numerical aperture of simulated imaging system
deltaF = 1/(2*256*0.275);  % the value (in spatial frequency units) of each pixel in the fourier domain
pupil_radius = round((2*pi*NA/wavelength)/deltaF);     % in pixels
