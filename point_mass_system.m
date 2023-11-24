% MATLAB Simulation of a damped harmonic oscillator, this file 
% is supposed to be the matlab version version of the analogue julia file
% inside of ventura modelling repository

clc;
clear;
close all;

% Parameters
m = 5;
k = 1;
c = 2;
c1 = 0.5;

% State space
A = [0 1; -k/m -c1/m];
A_damped = [0 1; -k/m -c/m];
b = [0;-1/m];
c = [1 0];

sys = ss(A,b,c,0);
sys_damped = ss(A_damped,b,c,0);

% Frequency analysis
G = tf(sys); G_damped = tf(sys_damped);
bodemag(G); hold on; grid on; bodemag(G_damped);
legend("Not Damped","Damped")

% Simulation
t = 0:0.1:100;
omega = sqrt(k/m);
input = 10*cos(omega*t);
output = lsim(sys,input,t);
output_damped = lsim(sys_damped,input,t);

figure;
plot(t,output);
hold on; grid on;
plot(t,output_damped);
legend("Not Damped","Damped");

figure;
impulse(sys);
hold on; grid on;
impulse(sys_damped);
legend("Not Damped","Damped")
