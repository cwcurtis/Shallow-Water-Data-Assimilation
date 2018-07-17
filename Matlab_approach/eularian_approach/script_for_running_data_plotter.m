clear all
%close all 
clc
figure

K = 64;
L1x = 10;
tf = 15; %
dt = .05;  % Time Step
dts = .2; % Sampling rate
Nens = 100;
sig = 1e-2;
Nplates = 32;

data_plotter(K,L1x,tf,dt,dts,Nens, sig, Nplates)


%% DNO expansion terms = 1
% If the number of terms in the DNO is small (say, 1), then sparser the
% measurement, the more nonlinearity plays a role over time (check this
% more robustly

% There also seemed to be something strange when we tried using 2 floats in
% that using 1 seemed to be more accurate vs. using 2 floats.  