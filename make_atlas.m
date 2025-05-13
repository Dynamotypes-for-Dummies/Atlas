clear all; clc;
addpath('Atlas Helper files');
% Create seizure type dictionary
Seizures = cell((22*3*5), 1);
counter = 1;

% % 
% % % % SN/SNIC
bifurcation = 1;
onset_indices = [1,1,55,55,28]; % orange #FFA500
offset_indices = [1,35,35,1,18]; % blue #1E90FF
[Seizures, counter] = make_plots_slow_wave(Seizures, counter, bifurcation, onset_indices, offset_indices);

% % % SN/SH
bifurcation = 1;
onset_indices = [1,1,12,12,7]; % orange #FFA500
offset_indices = [25,108,108,25,64]; % blue #1E90FF
[Seizures, counter] = make_plots_hysteresis(Seizures, counter, bifurcation, onset_indices, offset_indices);

% % SN/SH variant 1
bifurcation = 5;
onset_indices = [11,11,5,5,1]; % orange #FFA500
offset_indices = [75,60,60,75,94]; % blue #1E90FF
[Seizures, counter] = make_plots_hysteresis(Seizures, counter, bifurcation, onset_indices, offset_indices);

% %class SN/SH no DC
bifurcation = 2;
onset_indices = [1,1,37,37,71]; % orange #FFA500
offset_indices = [1,31,31,62,62]; % blue #1E90FF
[Seizures, counter] = make_plots_hysteresis(Seizures, counter, bifurcation, onset_indices, offset_indices);

% % SN/Sup
bifurcation = 3;
onset_indices = [5,5,20,20,15]; 
offset_indices = [40, 150,150, 40,60];
[Seizures, counter] = make_plots_piecewise(Seizures, counter, bifurcation, onset_indices, offset_indices);

% %SN/Sup variant 1
bifurcation = 3;
onset_indices = [5,5,8,8,14]; 
offset_indices = [40, 53,53, 40,1];
[Seizures, counter] = make_plots_hysteresis(Seizures, counter, bifurcation, onset_indices, offset_indices);

% %
% SN/Sup variant 2
bifurcation = 6;
onset_indices = [6,6,5,5,7]; % orange #FFA500
offset_indices = [3, 2, 2,3,1]; % blue #1E90FF
[Seizures, counter] = make_plots_hysteresis(Seizures, counter, bifurcation, onset_indices, offset_indices);

% % SN/FLC
bifurcation = 4;
onset_indices = [1,72,60,60,72]; % orange #FFA500
offset_indices = [21,94,94,125,135]; % blue #1E90FF
[Seizures, counter] = make_plots_hysteresis(Seizures, counter, bifurcation, onset_indices, offset_indices);

% % SNIC/SNIC
bifurcation = 5;
onset_indices = [1,1,22,22,18]; % orange #FFA500
offset_indices = [25,44,44,25,28]; % blue #1E90FF
[Seizures, counter] = make_plots_slow_wave(Seizures, counter, bifurcation, onset_indices, offset_indices);

% % SNIC/SH
bifurcation = 6;
onset_indices = [1,1,35,35,18]; % orange #FFA500
offset_indices = [1,55,55,1,28]; % blue #1E90FF
[Seizures, counter] = make_plots_slow_wave(Seizures, counter, bifurcation, onset_indices, offset_indices);
% % 

% % SNIC/Sup
bifurcation = 7;
onset_indices = [1,1,20,20,34]; 
offset_indices = [55,25,25,55, 10];
[Seizures, counter] = make_plots_slow_wave(Seizures, counter, bifurcation, onset_indices, offset_indices);

% % SNIC/Sup variant one
bifurcation = 7;
onset_indices = [1,1,20,20,44]; 
offset_indices = [40,150,150,40,60];
[Seizures, counter] = make_plots_piecewise(Seizures, counter, bifurcation, onset_indices, offset_indices);


% % % 
% % SNIC/FLC
bifurcation = 8;
onset_indices = [1,1,44,44,22]; % orange #FFA500
offset_indices = [1,201,201,1,101]; % blue #1E90FF
[Seizures, counter] = make_plots_slow_wave(Seizures, counter, bifurcation, onset_indices, offset_indices);
% % 
% % Sup/SNIC
bifurcation = 9;
onset_indices = [1,1,20,20,80]; 
offset_indices = [1,20,20,1,44];
[Seizures, counter] = make_plots_piecewise(Seizures, counter, bifurcation, onset_indices, offset_indices);

% % Sup/SNIC variant one
bifurcation = 9;
onset_indices = [15,15,20,20, 44]; 
offset_indices = [1,10,10,1,34];
[Seizures, counter] = make_plots_slow_wave(Seizures, counter, bifurcation, onset_indices, offset_indices);

% % Sup/SH
bifurcation = 10;
onset_indices = [40,40, 150,150,60]; 
offset_indices = [40,124,124,40,1];
[Seizures, counter] = make_plots_piecewise(Seizures, counter, bifurcation, onset_indices, offset_indices);
% 
% % Sup/Sup
bifurcation = 11;
onset_indices = [1,1,20,20,80]; 
offset_indices = [40,100,100,40,60];
[Seizures, counter] = make_plots_piecewise(Seizures, counter, bifurcation, onset_indices, offset_indices);
% 
%Sup/FLC
bifurcation = 12;
onset_indices = [1,1,46,46,23]; % orange #FFA500
offset_indices = [1,41,41,1,21]; % blue #1E90FF
[Seizures, counter] = make_plots_slow_wave(Seizures, counter, bifurcation, onset_indices, offset_indices);

% Sub/SNIC
bifurcation = 13;
onset_indices = [1,1,201,201,101]; % orange #FFA500
offset_indices = [1,44,44,1,22]; % blue #1E90FF
[Seizures, counter] = make_plots_slow_wave(Seizures, counter, bifurcation, onset_indices, offset_indices);

% Sub/SH
bifurcation = 7;
onset_indices = [1,1,7,7,12]; % orange #FFA500
offset_indices = [1,31,31,62,62]; % blue #1E90FF
[Seizures, counter] = make_plots_hysteresis(Seizures, counter, bifurcation, onset_indices, offset_indices);

% Sub/Sup
bifurcation = 15;
onset_indices = [1,1,41,41,21]; % orange #FFA500
offset_indices = [1,46,46,1,23]; % blue #1E90FF
[Seizures, counter] = make_plots_slow_wave(Seizures, counter, bifurcation, onset_indices, offset_indices);

% % Sub/FLC
bifurcation = 8;
onset_indices = [1,1,7,7,12]; % orange #FFA500
offset_indices = [100,160,160,232,232]; % blue #1E90FF
[Seizures, counter] = make_plots_hysteresis(Seizures, counter, bifurcation, onset_indices, offset_indices);

save('seizure_atlas.mat', 'Seizures');


function noisy_signal = add_pink_noise(signal, rms_signal, noise_amplitude_ratio)
addpath('Atlas Helper files');
    % Inputs:
    % signal - input signal (1D array)
    % noise_amplitude_ratio - fraction of signal amplitude for noise (e.g., 0.4 for 40%)
    
    % Compute the RMS amplitude of the signal
    
    
    % Generate pink noise of the same length as the signal
    % Pink noise can be generated using dsp.ColoredNoise in MATLAB
    L = length(signal);
    pink_noise = pinknoise([1,L],-1,10000)';
    
    % % Compute the RMS amplitude of the pink noise
     rms_noise = get_amp(pink_noise);
    % 
    % % % Scale the noise so its amplitude is noise_amplitude_ratio of the signal's amplitude
    % scaling_factor = noise_amplitude_ratio * (1 / rms_noise);
    % scaled_noise = pink_noise * scaling_factor;
    min_val = min(pink_noise(:));
max_val = max(pink_noise(:));
scaled_noise = 2*noise_amplitude_ratio*(pink_noise - min_val) / (max_val - min_val);
    % Add the scaled noise to the original signal
    noisy_signal = signal + scaled_noise;
end

function amp = get_amp(signal)

[peaks,locs] = findpeaks(signal ,'MinPeakProminence', 0.55);
[troughs_neg,locs_troughs] = findpeaks(signal, 'MinPeakProminence', 0.55);
troughs = -1*troughs_neg;

newnew = [];
len = 0;
if length(troughs) > length(peaks)
    len = length(peaks);
else
    len = length(troughs);
end
for i = 1:len
    newnew = [newnew; -1*troughs(i) + peaks(i)];
end
amp = mean(newnew);
end

function [Seizures, counter] = make_plots_piecewise(Seizures, counter, bifurcation, onset_indices, offset_indices)
addpath('Atlas Helper files');
% SETTINGS - INTEGRATION
x0=[0;0;0]; % initial conditions (must be a column)

% Settings - Model
% focus
b = 1.0; 

% radius of the sphere, do not change
R = 0.4; 

%%'The parameter k determines how many oscillations in the burst by setting the speed.The faster the movement the the less the number of oscillations per burst, and vise versa.
k = 0.001;

%Integration step/Sampling rate of the simulation
tstep = 0.01;

%%class wanted to run, input 3,7,9,10,11
Title(3) = "SN/Sup"; Stall_val(3) = 60000;
Title(7) = "SNIC/Sup"; Stall_val(7) = 60000;
Title(9) = "Sup/SNIC"; Stall_val(9) = 60000;
Title(10) = "Sup/SH"; Stall_val(10) = 60000;
Title(11) = "Sup/Sup"; Stall_val(11) = 60000;

%% function takes in class 3,7,9,10,11
[p0,onset_curve,p1_5,offset_curve,p3]=piecewise_random_path(bifurcation);
onset_curve = onset_curve';
offset_curve = offset_curve';
figure;
title(Title(bifurcation));
get_plot();
axis off;
hold on
A1 = offset_curve(:,offset_indices(1));
B1 = onset_curve(:,onset_indices(1));
plot3(A1(1),A1(2),A1(3),'Marker','o','MarkerSize',20,'MarkerFaceColor','#1E90FF','MarkerEdgeColor','#1E90FF', 'HandleVisibility', 'off')
plot3(B1(1),B1(2),B1(3),'Marker','^','MarkerSize',20,'MarkerFaceColor','#FFA500','MarkerEdgeColor','#FFA500', 'HandleVisibility', 'off')
A2 = offset_curve(:,offset_indices(3));
B2 = onset_curve(:,onset_indices(3));
plot3(A2(1),A2(2),A2(3),'Marker','^','MarkerSize',20,'MarkerFaceColor','#1E90FF','MarkerEdgeColor','#1E90FF', 'HandleVisibility', 'off')
plot3(B2(1),B2(2),B2(3),'Marker','o','MarkerSize',20,'MarkerFaceColor','#FFA500','MarkerEdgeColor','#FFA500', 'HandleVisibility', 'off')
A3 = offset_curve(:,offset_indices(5));
B3 = onset_curve(:,onset_indices(5));
plot3(A3(1),A3(2),A3(3),'Marker','square','MarkerSize',20,'MarkerFaceColor','#1E90FF','MarkerEdgeColor','#1E90FF', 'HandleVisibility', 'off')
plot3(B3(1),B3(2),B3(3),'Marker','square','MarkerSize',20,'MarkerFaceColor','#FFA500','MarkerEdgeColor','#FFA500', 'HandleVisibility', 'off')
plot3(p1_5(:,1),p1_5(:,2), p1_5(:,3),'Marker','diamond','MarkerSize',20,'MarkerFaceColor','#696969','MarkerEdgeColor','#696969', 'HandleVisibility', 'off')
onsets = [];
offsets = [];
for noise_level = 1:3
for path = 1:5

 onset_index=onset_indices(path);
    offset_index=offset_indices(path);
    
    p1 = onset_curve(:,onset_index)';
    p2 = offset_curve(:,offset_index)';



% uncomment this code to do random path
% % One random path - select random point on onset curve and offset curve
% random_onset_index=randsample(onset_curve_length,1);
% random_offset_index=randsample(offset_curve_length,1);
% p1 = offset_curve(:,random_offset_index);
% p2 = onset_curve(:,random_onset_index);


stall_val = Stall_val(bifurcation);
[mu2_straight_path0,mu1_straight_path0,nu_straight_path0,rad1] = sphereArcPath(k,tstep,p0,p1);
[mu2_straight_path0_5,mu1_straight_path0_5,nu_straight_path0_5,rad2] = sphereArcPath(k,tstep,p1,p1_5);
points = repmat(p1_5, stall_val, 1)';

sigma = 20;
Rn = [pinknoise([1,length(points)],-1, sigma);pinknoise([1,length(points)],-1, sigma);pinknoise([1,length(points)],-1, sigma)];
points = points + Rn;
[mu2_straight_path,mu1_straight_path,nu_straight_path,rad3] = sphereArcPath(k,tstep,p1_5,p2);
[mu2_straight_path1,mu1_straight_path1,nu_straight_path1,rad4] = sphereArcPath(k,tstep,p2,p3);
mu2_all = [mu2_straight_path0, mu2_straight_path0_5, points(1, :), mu2_straight_path, mu2_straight_path1];
mu1_all = [mu1_straight_path0, mu1_straight_path0_5, points(2, :), mu1_straight_path, mu1_straight_path1];
mu1_all = -mu1_all;
nu_all = [nu_straight_path0, nu_straight_path0_5, points(3,:), nu_straight_path, nu_straight_path1];

N_t = length(mu2_all);
X = zeros(3,N_t);
xx = x0;
sigma = ((N_t)/100000)*80;
noise_arr = [0, sigma, sigma*5];
sigma = noise_arr(noise_level);
Rn = [pinknoise([1,N_t],-1, sigma);pinknoise([1,N_t],-1, 00);pinknoise([1,N_t],-1, 00)];
mu2_big = zeros(1, length(N_t));
mu1_big = zeros(1, length(N_t));
nu_big = zeros(1, length(N_t));

%%get onset index by finding Radians to bifurcation, and getting index
%%through k and tstep parameters
 onset_index = floor((rad1/k)/tstep) ;
 offset_index = floor(((rad1+rad2+rad3)/k)/tstep) + stall_val;
 onsets = [onsets, onset_index];
 offsets = [offsets, offset_index];
for n = 1:N_t
    %%Euler-Meruyama method
    [Fxx,mu2,mu1,nu] = SlowWave_Model_piecewise(0,xx,b,k,mu2_all(n), mu1_all(n),nu_all(n));
    xx = xx + tstep*Fxx + sqrt(tstep)*Rn(:,n);
    X(:,n) = xx;
    mu2_big(n) = mu2;
    mu1_big(n) = mu1;
    nu_big(n) = nu;
end
x = X';
Seizures{counter} = x(:,1);
    counter = counter + 1;
    plot3(mu2_all,-mu1_all,nu_all,'color','#696969','LineWidth',2, 'HandleVisibility', 'off') % bursting path
    seizure = x(:,1);
end
end

% ── layout parameters ─────────────────────────────────────────────────
nPerBlock = 5;
nBlocks   = 3;

padBlock  = 0.09;   % margin on left/right and top/bottom
padSmallX = 0.02;   % gap between the two columns INSIDE each block
padSmallY = 0.02;   % gap between the three rows INSIDE each block

% compute widths for two‐column layout across all blocks:
halfW  = (1 - (nBlocks+1)*padBlock - nBlocks*padSmallX) / (2*nBlocks);
blockW = 2*halfW + padSmallX;

% compute heights so that top margin, bottom margin, and 2 inter-row gaps
% are taken into account, filling full [0,1]:
smallH = (1 - 2*padBlock - 2*padSmallY)/3;

% y‐positions of the three rows (top of axes at yTop+smallH):
yTop = 1 - padBlock - smallH;
yMid = yTop - smallH - padSmallY;
yBot = yMid - smallH - padSmallY;  % this bottom now sits at padBlock

% ── draw ──────────────────────────────────────────────────────────────
figure('Color','w');
set(gcf,'Units','normalized','Position',[.1 .1 .8 .7]);
title(Title(bifurcation));
axis off;
count_seiz = counter - 15;
count_onset_offset = 1;
for b = 1:nBlocks
    x0 = padBlock + (b-1)*(blockW + padBlock);
    
    pos = {
      [x0                , yTop,    halfW, smallH];  % top-left
      [x0+halfW+padSmallX, yTop,    halfW, smallH];  % top-right
      [x0                , yMid,    halfW, smallH];  % mid-left
      [x0+halfW+padSmallX, yMid,    halfW, smallH];  % mid-right
      [x0 + (blockW-halfW)/2, yBot, halfW, smallH];  % bottom-centered
    };
    
    for i = 1:5
        idx = (b-1)*5 + i;
        ax  = axes('Position',pos{i});
        seizure = Seizures{count_seiz};
        count_seiz = count_seiz + 1;
    marker_array_1 = ['o', 'o', '^', '^', 'square'];
marker_array_2 = [ '^', 'o', 'o','^' 'square'];
plot(seizure,'color','#696969','LineWidth',1)
hold on
plot(onsets(count_onset_offset),seizure(onsets(count_onset_offset)),'Marker',marker_array_1(i),'MarkerFaceColor','#FFA500','MarkerEdgeColor','#FFA500','MarkerSize',10)
plot(offsets(count_onset_offset),seizure(offsets(count_onset_offset)),'Marker',marker_array_2(i),'MarkerFaceColor','#1E90FF','MarkerEdgeColor','#1E90FF','MarkerSize',10)
        axis off
        count_onset_offset = count_onset_offset + 1;
    end
end








end

function [Xdot, mu2,mu1,nu] = SlowWave_Model_piecewise(~,x,~,k,mu2,mu1,nu)
% Parametrization of the path in the spherical parameter space in terms
% of a circle defined by 3 points
% System
xdot = - x(2);
ydot = x(1)^3 - mu2*x(1) - mu1 - x(2)*( nu + x(1) + x(1)^2);
zdot = k;
Xdot = [xdot;ydot;zdot];
end
function   [p0,p1,p1_5,p2,p3]=piecewise_random_path(bifurcation)
addpath(fullfile('Atlas Helper files'));
load('curves.mat');
load('bifurcation_crossing.mat')
load("curves2.mat")

if bifurcation==3
    %fixed rest point
    p0 = Hopf(:,930)';
    %bifurcation curve
    randomNumber = randi([145,170]);
    p1 = Fold(:,145:170)';
    randomNumber2 = randi([600,750]);
   p1_5 = [0.3196, 0.2389, -0.0279];
    %bifurcation curve
    p2 = Hopf(:,600:750)' ;
    %fixed rest
    p3 = [ 0.1944 , 0.0893 , 0.3380];

end

if bifurcation==7
    randomNumber = randi([600,750]);
 randomNumber2 = randi([1,44]);
%fixed rest point
p0 = Fold(:,400)';
%bifurcation curve
p1 =  SNIC' ;
%random point in limit cycle
p1_5 = [0.1314, 0.3298, -0.1843];
%bifurcation curve
p2 = Hopf(:,600:750)';
%fixed rest
p3 = [ 0.1944 , 0.0893 , 0.3380];
end

if bifurcation==9
    randomNumber = randi([600,750]);
    %fixed rest point
    p0 = [ 0.1944 , 0.0893 , 0.3380];
    %bifurcation curve
    p1 = Hopf(:,600:750)';
    %change here
    randomNumber2 = randi([1,44]);
    %random point in limit cycle
    p1_5 = [-0.0441, 0.2591, -0.3015];
    %bifurcation curve
    p2 = SNIC' ;
    %fixed rest
    p3 = Fold(:,450)';
end

if bifurcation==10
    randomNumber = randi([600,750]);
    %fixed rest point
    p0 = [ 0.1944 , 0.0893 , 0.3380];
    %bifurcation curve
    p1 = Hopf(:,600:750)';%get_nearest_hopf(p0(1),p0(2),p0(3))';
    %change here
    randomNumber2 = randi([1,124]);
    %random point in limit cycle
    p1_5 = [-0.123686721647726	0.338825918756816	-0.172912092308889];
    %bifurcation curve
    p2 = Homoclinic_to_saddle' ;
    %fixed rest
    p3 = Fold(:,400)';
end

if bifurcation==11
     randomNumber = randi([600,750]);
%fixed rest point
p0 = [ 0.1944 , 0.0893 , 0.3380];
%bifurcation curve
p1 = Hopf(:,600:750)';
%change here
randomNumber2 = randi([600,750]);
%random point in limit cycle
p1_5 = [-0.2104, 0.3180, -0.1209];
%bifurcation curve
p2 = Hopf(:,600:750)' ;
%fixed rest
p3 = [ 0.1944 , 0.0893 , 0.3380];
end


end



% save('sample_Simulated_Seizures.mat', 'Seizures');
function [Seizures, counter] = make_plots_slow_wave(Seizures, counter, bifurcation, onset_indices, offset_indices)
% Settings - Integration
x0=[0;0;0]; % initial conditions (must be a column)

addpath('Atlas Helper files');
% Settings - Model Sphere
b = 1.0; % focus
R = 0.4; % radius


% Class Information (paths)
% c1
P1(1,:)=[0.3479,0.07897,0.181]; P2(1,:)=[0.3697,0.08652,0.1258]; P3(1,:)=[0.3541,0.06534,0.1741]; kk(1)=0.005; Tmax(1)=12000; Tstep(1)=0.01; PeakP(1)=1; Title(1) = "SN/SNIC";
% c5 
P1(5,:)=[0.3821,0.09092,0.07562]; P2(5,:)=[0.3757,0.1221,0.062]; P3(5,:)=[0.3647,0.08476,0.1409]; kk(5)=0.005; Tmax(5)=12000; Tstep(5)=0.01; PeakP(5)=0.75; Title(5) = "SNIC/SNIC";
% c6 
P1(6,:)=[0.3479,0.07897,0.181]; P2(6,:)=[0.3541,0.06534,0.1741]; P3(6,:)=[0.3697,0.08652,0.1258]; kk(6)=0.005; Tmax(6)=12000; Tstep(6)=0.01; PeakP(6)=1; Title(6) = "SNIC/SH";
% c7 
P1(7,:)=[0.3489,-0.07931,0.189]; P2(7,:)=[0.3697,0.08652,0.1258]; P3(7,:)=[0.2038,0.2939,0.179]; kk(7)=0.005; Tmax(7)=12000; Tstep(7)=0.01; PeakP(7)=1.25; Title(7) = "SNIC/Sup";
% c8 
P1(8,:)=[0.3883,0.09312,-0.02389]; P2(8,:)=[0.1363,0.01502,-0.3758]; P3(8,:)=[0.1912,-0.02344,-0.3506];  kk(8)=0.008; Tmax(8)=12000; Tstep(8)=0.01; PeakP(8)=1; Title(8) = "SNIC/FLC";
% c9
P1(9,:)=[0.2038,0.2939,0.179]; P2(9,:)=[0.3697,0.08652,0.1258]; P3(9,:)=[0.3489,-0.07931,0.189]; kk(9)=0.005; Tmax(9)=12000; Tstep(9)=0.01; PeakP(9)=1.25; Title(9) = "Sup/SNIC";
% c12 
P1(12,:)=[-0.274,-0.07482,-0.2817]; P2(12,:)=[-0.2786,-0.08796,-0.2732]; P3(12,:)=[-0.3831,-0.03874,-0.1083]; kk(12)=0.006; Tmax(12)=12000; Tstep(12)=0.01; PeakP(12)=1; Title(12) = "Sup/FLC";
% c13
P1(13,:)=[0.1912,-0.02344,-0.3506]; P2(13,:)=[0.1363,0.01502,-0.3758]; P3(13,:)=[0.3883,0.09312,-0.02389];  kk(13)=0.008; Tmax(13)=12000; Tstep(13)=0.01; PeakP(13)=1; Title(13) = "Sub/SNIC";
% c15
P1(15,:)=[-0.3831,-0.03874,-0.1083]; P2(15,:)=[-0.2786,-0.08796,-0.2732]; P3(15,:)=[-0.274,-0.07482,-0.2817]; kk(15)=0.005; Tmax(15)=12000; Tstep(15)=0.01; PeakP(15)=1; Title(15) = "Sub/Sup";

load("curves2.mat")
load('curves2.mat');


if bifurcation == 1
    addpath('Atlas Helper files');
    onset_curve = SHl(:,50:104); %55
    offset_curve = [0.33, 0.11, 0.18]'; 
    offset_curve2 = SNIC(:,1:35); %35
    load('class1_fit.mat');
    flag = 1;
end


if bifurcation == 5
    onset_curve = SNIC; %44
    offset_curve = SNIC; %44
    offset_curve2 = [0.34,0.14,0.06]'; 
    load('class5_fit.mat');
    flag = 2;
end


if bifurcation == 6
    onset_curve = SNIC(:,1:35); %35
    offset_curve = [0.33, 0.11, 0.18]'; 
    offset_curve2 = SHl(:,50:104); %55
    load('class6_fit.mat');
    flag = 1;
end


if bifurcation == 7
    onset_curve=SNIC; %44
    offset_curve = Hopf(:,800:855); %56
    offset_curve2 = [0.36,-0.12,0.12]'; 
    load('class7_fit.mat');
    flag = 3;
end


if bifurcation == 8
    onset_curve=SNIC; %44
    offset_curve = [0.34,0.2,-0.06]';
    offset_curve2 = FLC(:,100:300); %201
    load('class8_fit.mat');
    flag = 1;
end


if bifurcation == 9
    onset_curve= Hopf(:,800:855); %56
    offset_curve = SNIC; %44
    offset_curve2 = [0.36,-0.12,0.12]';
    load('class7_fit.mat');
    flag = 3;
end


if bifurcation == 12
    onset_curve= Hopf(:,450:495);%46  
    offset_curve = FLC(:,60:100);%41
    offset_curve2 = [-0.3, -0.2, -0.2]';
    load('class12_fit.mat');
    flag = 3;
end


if bifurcation == 13
    onset_curve = FLC(:,100:300); %201
    offset_curve = [0.34,0.2,-0.06]';
    offset_curve2 = SNIC; %44 
    load('class13_fit.mat');
    flag = 1;
end


if bifurcation == 15
    onset_curve= FLC(:,60:100);%41
    offset_curve = Hopf(:,450:495);%46  
    offset_curve2 = [-0.3, -0.2, -0.2]';
    load('class15_fit.mat');
    flag = 3;
end







% Class specific timespan
tstep = Tstep(bifurcation);
tmax = Tmax(bifurcation);
tspan = 0:tstep:tmax;


% Class specific timescale separation
k = kk(bifurcation);

addpath('Atlas Helper files');

% Plot endpoints
if( flag == 1)
figure;
title(Title(bifurcation));
get_plot();
hold on
A1 = offset_curve2(:,offset_indices(1));
B1 = onset_curve(:,onset_indices(1));
plot3(A1(1)*1,A1(2)*1,A1(3)*1,'Marker','o','MarkerSize',20,'MarkerFaceColor','#1E90FF','MarkerEdgeColor','#1E90FF', HandleVisibility='off')
plot3(B1(1)*1,B1(2)*1,B1(3)*1,'Marker','^','MarkerSize',20,'MarkerFaceColor','#FFA500','MarkerEdgeColor','#FFA500', HandleVisibility='off')
A2 = offset_curve2(:,offset_indices(3))*1;
B2 = onset_curve(:,onset_indices(3))*1;
plot3(A2(1)*1,A2(2)*1,A2(3)*1,'Marker','^','MarkerSize',20,'MarkerFaceColor','#1E90FF','MarkerEdgeColor','#1E90FF', HandleVisibility='off')
plot3(B2(1)*1,B2(2)*1,B2(3)*1,'Marker','o','MarkerSize',20,'MarkerFaceColor','#FFA500','MarkerEdgeColor','#FFA500', HandleVisibility='off')
A3 = offset_curve2(:,offset_indices(5));
B3 = onset_curve(:,onset_indices(5));
plot3(A3(1)*1,A3(2)*1,A3(3)*1,'Marker','square','MarkerSize',20,'MarkerFaceColor','#1E90FF','MarkerEdgeColor','#1E90FF', HandleVisibility='off')
plot3(B3(1)*1,B3(2)*1,B3(3)*1,'Marker','square','MarkerSize',20,'MarkerFaceColor','#FFA500','MarkerEdgeColor','#FFA500', HandleVisibility='off')
plot3(offset_curve(1),offset_curve(2),offset_curve(3),'Marker','diamond','MarkerSize',20,'MarkerFaceColor','#696969','MarkerEdgeColor','#696969', HandleVisibility='off')
else
figure;
title(Title(bifurcation));
get_plot();
hold on
A1 = offset_curve(:,offset_indices(1));
B1 = onset_curve(:,onset_indices(1));
plot3(A1(1)*1,A1(2)*1,A1(3)*1,'Marker','o','MarkerSize',20,'MarkerFaceColor','#1E90FF','MarkerEdgeColor','#1E90FF', HandleVisibility='off')
plot3(B1(1)*1,B1(2)*1,B1(3)*1,'Marker','^','MarkerSize',20,'MarkerFaceColor','#FFA500','MarkerEdgeColor','#FFA500', HandleVisibility='off')
A2 = offset_curve(:,offset_indices(3));
B2 = onset_curve(:,onset_indices(3));
plot3(A2(1)*1,A2(2)*1,A2(3)*1,'Marker','^','MarkerSize',20,'MarkerFaceColor','#1E90FF','MarkerEdgeColor','#1E90FF', HandleVisibility='off')
plot3(B2(1)*1,B2(2)*1,B2(3)*1,'Marker','o','MarkerSize',20,'MarkerFaceColor','#FFA500','MarkerEdgeColor','#FFA500', HandleVisibility='off')
A3 = offset_curve(:,offset_indices(5));
B3 = onset_curve(:,onset_indices(5));
plot3(A3(1)*1,A3(2)*1,A3(3)*1,'Marker','square','MarkerSize',20,'MarkerFaceColor','#1E90FF','MarkerEdgeColor','#1E90FF', HandleVisibility='off')
plot3(B3(1)*1,B3(2)*1,B3(3)*1,'Marker','square','MarkerSize',20,'MarkerFaceColor','#FFA500','MarkerEdgeColor','#FFA500', HandleVisibility='off')
plot3(offset_curve2(1),offset_curve2(2),offset_curve2(3),'Marker','diamond','MarkerSize',20,'MarkerFaceColor','#696969','MarkerEdgeColor','#696969', HandleVisibility='off')

end

for noise_level = 1:3
for path = 1:5

if flag == 2    % c5 (same onset/offset curve)

    A = offset_curve(:,offset_indices(path));
    B = onset_curve(:,onset_indices(path));
    C = offset_curve2;
    sigma = f(onset_indices(path),offset_indices(path));
end


if flag == 1     % c1, c6, c8, c13 (fixed point is in the middle)
    A = onset_curve(:,onset_indices(path));
    B = offset_curve;
    C = offset_curve2(:,offset_indices(path));
    sigma = f(onset_indices(path),offset_indices(path));
end


if flag == 3     % c7, c9, c12, c15 (fixed point is at the end)
    A = onset_curve(:,onset_indices(path));
    B = offset_curve(:,offset_indices(path));
    C = offset_curve2;
    sigma = f(onset_indices(path),offset_indices(path));
end
    if ismember(bifurcation, [6,15])
    noise_level_arr = [0, sigma, sigma*3];
    elseif ismember(bifurcation, [8,13])
    noise_level_arr = [0, sigma, sigma*7];
    else
    noise_level_arr = [0, sigma, sigma*5];
    end
    sigma = noise_level_arr(noise_level);
    p1 = A';
    p2 = B';
    p3 = C';
    % Create circular path based 3 defining points
    [E, F, C, r] = Parametrization_3PointsCircle(p1,p2,p3);


    N_t = length(tspan);
    X = zeros(3,N_t);
    xx = x0;
    Rn = [pinknoise([1,N_t],-1, sigma);pinknoise([1,N_t],-1, 0);pinknoise([1,N_t],-1, 0)];


    for n = 1:N_t


        % Euler-Meruyama method
        [Fxx, mu2, mu1, nu] = SlowWave_Model(tspan(n),xx,b,k,E,F,C,r);
        xx = xx + tstep*Fxx + sqrt(tstep)*Rn(:,n);
        X(:,n) = xx;


    end
    

    x = X';
    t = tspan;

    % Calculate Bursting Path
    z=0:0.01:2*pi;
    mu2=C(1)+r*(E(1)*cos(z)+F(1)*sin(z));
    mu1=-(C(2)+r*(E(2)*cos(z)+F(2)*sin(z)));
    nu=C(3)+r*(E(3)*cos(z)+F(3)*sin(z));


    % %%% Plot bursting path
    plot3(mu2*1,-mu1*1,nu*1,'color','#696969','LineWidth',2, HandleVisibility='off') % bursting path





%Make a copy of x that will not be smoothed
xtemp = x;
%smooth x for better resolution of seizure envelope
x = smoothdata(x, 'gaussian', 3000);
 %highpass filter
x_HP = highpass(xtemp(:,1), 0.1, 100);
%Create envelope, normalize, and smooth
Amp2 = movmean(abs(x_HP),5000);
Amp2=normalize(Amp2);
Amp2 = smoothdata(Amp2, 'gaussian', sigma*8+3000);
min_val = min(Amp2);
max_val = max(Amp2);
Amp2 = 2 * ((Amp2 - min_val) / (max_val - min_val)) - 1;
%Threshold for Amp2 crossing to determine offset/offset, adjust manually
Amp2_base = -1;
%Normalize threshold to be zero point in Amp2
Add_term = 0.9;
Amp2 = Amp2 + Add_term;
%Find onset/offset using zero crossing
zeroCrossings = find(Amp2(1:end-1) .* Amp2(2:end) < 0);
%Case where simulation starts from rest
if(Amp2(1) < 0)
onset_time = zeroCrossings(1:2:end);%*tstep; % Even indices
offset_time = zeroCrossings(2:2:end);%*tstep; % Odd indices
end
%Case where simulation starts from seizure
if(Amp2(1) > 0)
onset_time = zeroCrossings(2:2:end);%*tstep; % Even indices
offset_time = zeroCrossings(3:2:end);%*tstep; % Odd indices
end
onset_time = max(onset_time(1)-7500, 1);
seizure = xtemp(onset_time:offset_time(1)+7500, 1);

    Seizures{counter} = seizure;
    counter = counter + 1;

end
end

% idx = counter - 5;
% count_itr = 1;
% for i = idx:(counter-1)
%     seizure = Seizures{i};
%     marker_array_1 = ['o', 'o', '^', '^', 'square'];
% marker_array_2 = [ '^', 'o', 'o','^' 'square'];
% figure()
% plot(seizure,'color','#696969','LineWidth',1)
% hold on
% plot(7500,seizure(7500),'Marker',marker_array_1(count_itr),'MarkerFaceColor','#FFA500','MarkerEdgeColor','#FFA500','MarkerSize',10)
% plot(length(seizure)-7500,seizure(length(seizure)-7500),'Marker',marker_array_2(count_itr),'MarkerFaceColor','#1E90FF','MarkerEdgeColor','#1E90FF','MarkerSize',10)
% set(gca,'XTick',[], 'YTick', [])
% count_itr = count_itr + 1;
%title(Title(bifurcation));






% ── layout parameters ─────────────────────────────────────────────────
nPerBlock = 5;
nBlocks   = 3;

padBlock  = 0.09;   % margin on left/right and top/bottom
padSmallX = 0.02;   % gap between the two columns INSIDE each block
padSmallY = 0.02;   % gap between the three rows INSIDE each block

% compute widths for two‐column layout across all blocks:
halfW  = (1 - (nBlocks+1)*padBlock - nBlocks*padSmallX) / (2*nBlocks);
blockW = 2*halfW + padSmallX;

% compute heights so that top margin, bottom margin, and 2 inter-row gaps
% are taken into account, filling full [0,1]:
smallH = (1 - 2*padBlock - 2*padSmallY)/3;

% y‐positions of the three rows (top of axes at yTop+smallH):
yTop = 1 - padBlock - smallH;
yMid = yTop - smallH - padSmallY;
yBot = yMid - smallH - padSmallY;  % this bottom now sits at padBlock

% ── draw ──────────────────────────────────────────────────────────────
figure('Color','w');
title(Title(bifurcation));
axis off ;
set(gcf,'Units','normalized','Position',[.1 .1 .8 .7]);

count_seiz = counter - 15;
for b = 1:nBlocks
    x0 = padBlock + (b-1)*(blockW + padBlock);
    
    pos = {
      [x0                , yTop,    halfW, smallH];  % top-left
      [x0+halfW+padSmallX, yTop,    halfW, smallH];  % top-right
      [x0                , yMid,    halfW, smallH];  % mid-left
      [x0+halfW+padSmallX, yMid,    halfW, smallH];  % mid-right
      [x0 + (blockW-halfW)/2, yBot, halfW, smallH];  % bottom-centered
    };
    
    for i = 1:5
        idx = (b-1)*5 + i;
        ax  = axes('Position',pos{i});
        seizure = Seizures{count_seiz};
        count_seiz = count_seiz + 1;
    marker_array_1 = ['o', 'o', '^', '^', 'square'];
marker_array_2 = [ '^', 'o', 'o','^' 'square'];
plot(seizure,'color','#696969','LineWidth',1)
hold on
plot(7500,seizure(7500),'Marker',marker_array_1(i),'MarkerFaceColor','#FFA500','MarkerEdgeColor','#FFA500','MarkerSize',10)
plot(length(seizure)-7500,seizure(length(seizure)-7500),'Marker',marker_array_2(i),'MarkerFaceColor','#1E90FF','MarkerEdgeColor','#1E90FF','MarkerSize',10)
        axis off
    end
end


end












function [Seizures, counter] = make_plots_hysteresis(Seizures, counter, bifurcation, onset_indices, offset_indices)
x0=[0;0;0]; % initial conditions (must be a column)
load('curves.mat');
load('curves2.mat');
load('bifurcation_crossing.mat')
% Settings - Model Sphere
b = 1.0; % focus
R = 0.4; % radius


%%%%
% Class Information (paths)
% % c2s 
% AA(1,:)=[0.3448,0.02285,0.2014]; BB(1,:)=[0.3351,0.07465,0.2053]; kk(1)=0.001; DSTAR(1)=0.3; NN(1)=1; Tmax(1)=15000; Tstep(1)=0.01;
% % c2b 
% AA(2,:)=[0.3216,0.0454,-0.2335]; BB(2,:)=[0.2850,0.0585,-0.2745]; kk(2)=0.002; DSTAR(2)=0.3; NN(2)=1; Tmax(2)=50000; Tstep(2)=0.01;
% % c3s 
% AA(3,:)=[0.2552,-0.0637,0.3014]; BB(3,:)=[0.3496,0.0795,0.1774]; kk(3)=0.003; DSTAR(3)=0.3; NN(3)=1; Tmax(3)=30000; Tstep(3)=0.01;
% % c4b 
% AA(4,:)=[0.1871,-0.02512,-0.3526]; BB(4,:)=[0.2081,-0.01412,-0.3413]; kk(4)=0.004; DSTAR(4)=0.3; NN(4)=1; Tmax(4)=30000; Tstep(4)=0.01;
% % c10s 
% AA(5,:)=[0.3448,0.0228,0.2014]; BB(5,:)=[0.3118,0.0670,0.2415]; kk(5)=0.0001; DSTAR(5)=0.3; NN(5)=1; Tmax(5)=160000; Tstep(5)=0.01;
% % c11s - supH onset
% AA(6,:)=[0.3131,-0.06743,0.2396]; BB(6,:)=[0.3163,0.06846,0.2351]; kk(6)=0.0001; DSTAR(6)=0.3; NN(6)=1; Tmax(6)=250000; Tstep(6)=0.01;
% % c14b 
% AA(7,:)=[0.3883,0.03687,-0.2521]; BB(7,:)=[0.184,0.02903,-0.354]; kk(7)=0.008; DSTAR(7)=0.3; NN(7)=1; Tmax(7)=20000; Tstep(7)=0.01;
% % c16b - subH onset
% AA(8,:)=[-0.0652,-0.0947,-0.3831]; BB(8,:)=[0.0282,-0.0198,-0.3985]; kk(8)=0.004; DSTAR(8)=0.3; NN(8)=1; Tmax(8)=20000; Tstep(8)=0.01;



%%%%
% Class Information (paths)
% c2s 
AA(1,:)=[0.3448,0.02285,0.2014]; BB(1,:)=[0.3351,0.07465,0.2053]; kk(1)=0.001; DSTAR(1)=0.3; NN(1)=1; Tmax(1)=15000; Tstep(1)=0.01; Title(1)="SN/SH";
% c2b 
AA(2,:)=[0.3216,0.0454,-0.2335]; BB(2,:)=[0.2850,0.0585,-0.2745]; kk(2)=0.002; DSTAR(2)=0.3; NN(2)=1; Tmax(2)=50000; Tstep(2)=0.01; Title(2)="SN/SH";
%class 3s
AA(3,:)=[0.2552,-0.0637,0.3014]; BB(3,:)=[0.3496,0.0795,0.1774]; kk(3)=0.003; DSTAR(3)=0.3; NN(3)=1; Tmax(3)=30000; Tstep(3)=0.01;Title(3)="SN/Sup";
% c4b 
AA(4,:)=[0.1871,-0.02512,-0.3526]; BB(4,:)=[0.2081,-0.01412,-0.3413]; kk(4)=0.007; DSTAR(4)=0.1; NN(4)=1; Tmax(4)=30000; Tstep(4)=0.01; Title(4)="SN/FLC";
%10s
AA(5,:)=[0.3448,0.0228,0.2014]; BB(5,:)=[0.3118,0.0670,0.2415]; kk(5)=0.001; DSTAR(5)=0.3; NN(5)=1; Tmax(5)=160000; Tstep(5)=0.01; Title(5)="SN/SH";
% % c11s - supH onset
AA(6,:)=[0.3131,-0.06743,0.2396]; BB(6,:)=[0.3163,0.06846,0.2351]; kk(6)=0.000025; DSTAR(6)=0.15; NN(6)=1; Tmax(6)=200000; Tstep(6)=0.01; Title(6)="SN/Sup";
% c14b 
AA(7,:)=[0.3883,0.03687,-0.2521]; BB(7,:)=[0.184,0.02903,-0.354]; kk(7)=0.004; DSTAR(7)=0.3; NN(7)=1; Tmax(7)=15000; Tstep(7)=0.01; Title(7)="Sub/SH";
% c16b - subH onset
AA(8,:)=[-0.0652,-0.0947,-0.3831]; BB(8,:)=[0.0282,-0.0198,-0.3985]; kk(8)=0.004; DSTAR(8)=0.3; NN(8)=1; Tmax(8)=20000; Tstep(8)=0.01; Title(8)="Sub/FLC";

% Class specific timespan
tstep = Tstep(bifurcation);
tmax = Tmax(bifurcation);
tspan = 0:tstep:tmax;


% Class specific timescale separation
k = kk(bifurcation);


% Class specific threshold
dstar = DSTAR(bifurcation);


% Equilibrium Branch for Resting State
N = NN(bifurcation);



if bifurcation==1
    onset_curve=SNr_LCs;
    offset_curve=SHl; 
    load('class2s_fit.mat')
end


if bifurcation==2
    onset_curve=SNr_LCb;
    offset_curve=SHb;
    load('class2b_fit.mat')
end


if bifurcation==3
    onset_curve=SNr_LCs;
    offset_curve=SNl_ActiveRest;
    load('class3s_fit.mat')
end


if bifurcation==4
    onset_curve=SNr_LCb;
    offset_curve=FLC_top;
    load('class4b_fit.mat')
end


if bifurcation==5
    onset_curve=Fold(:,200:210);
    offset_curve=SHl;
    load('class10s_fit.mat')
end


if bifurcation==6
    onset_curve= SNr_ActiveRest;
    offset_curve=[[0.3171; -0.066; 0.2347], [0.3115; -0.0546; 0.2450],[0.3166; -0.0654; 0.2356]];%SNl_ActiveRest;
    load('class11s_fit.mat')
   f = @(x,y) f(x,y) /5;
end


if bifurcation==7
    onset_curve=subH;
    offset_curve=SHb;
    load('class14b_fit.mat')
end


if bifurcation==8
    onset_curve=subH;
    offset_curve=FLC;
    load('class16b_fit.mat')
end



% Plot endpoints
figure;
title(Title(bifurcation));
get_plot();
hold on
A1 = offset_curve(:,offset_indices(1))*1;
B1 = onset_curve(:,onset_indices(1))*1;
plot3(A1(1),A1(2),A1(3),'Marker','o','MarkerSize',20,'MarkerFaceColor','#1E90FF','MarkerEdgeColor','#1E90FF', HandleVisibility='off')
plot3(B1(1),B1(2),B1(3),'Marker','^','MarkerSize',20,'MarkerFaceColor','#FFA500','MarkerEdgeColor','#FFA500', HandleVisibility='off')
A2 = offset_curve(:,offset_indices(3))*1;
B2 = onset_curve(:,onset_indices(3))*1;
plot3(A2(1),A2(2),A2(3),'Marker','^','MarkerSize',20,'MarkerFaceColor','#1E90FF','MarkerEdgeColor','#1E90FF', HandleVisibility='off')
plot3(B2(1),B2(2),B2(3),'Marker','o','MarkerSize',20,'MarkerFaceColor','#FFA500','MarkerEdgeColor','#FFA500', HandleVisibility='off')
A3 = offset_curve(:,offset_indices(5))*1;
B3 = onset_curve(:,onset_indices(5))*1;
plot3(A3(1),A3(2),A3(3),'Marker','square','MarkerSize',20,'MarkerFaceColor','#1E90FF','MarkerEdgeColor','#1E90FF', HandleVisibility='off')
plot3(B3(1),B3(2),B3(3),'Marker','square','MarkerSize',20,'MarkerFaceColor','#FFA500','MarkerEdgeColor','#FFA500', HandleVisibility='off')



for noise_level = 1:3
for path = 1:5


    % Onset and Offset Points
    onset_index=onset_indices(path);
    offset_index=offset_indices(path);
    A = offset_curve(:,offset_index);
    B = onset_curve(:,onset_index);
    
    sigma=f(onset_index,offset_index);
    if bifurcation == 1
        noise_arr = [0, sigma, sigma*7];
    elseif ismember(bifurcation , [4 7 8])
        noise_arr = [0, sigma, sigma*3];

    else
        noise_arr = [0, sigma, sigma*5];
    end
    sigma = noise_arr(noise_level);
    
    % Create circular path based 3 defining points
    [E, F] = Parametrization_2PointsArc(A,B,R);


    N_t = length(tspan);
    X = zeros(3,N_t);
    xx = x0;
    Rn = [pinknoise([1,N_t],-1, sigma);pinknoise([1,N_t],-1, 0);pinknoise([1,N_t],-1, 0)];
   
    
    for n = 1:N_t


        % Euler-Meruyama method
        Fxx = HysteresisLoop_Model(tspan(n),xx,b,k,R,dstar,E,F,N);
        xx = xx + tstep*Fxx + sqrt(tstep)*Rn(:,n);
        X(:,n) = xx;


    end


    x = X';

    t = tspan;


    % Calculate Bursting Path
    z = x(:,3);
    mu2 = R.*(E(1).*cos(z)+F(1).*sin(z));
    mu1 = -R.*(E(2).*cos(z)+F(2).*sin(z));
    nu = R.*(E(3).*cos(z)+F(3).*sin(z));


    % Plot Bursting path
    plot3(mu2*1,-mu1*1,nu*1,'color','#696969','LineWidth',2, HandleVisibility='off') % bursting path
% Calculate Onset Times
[pks,times]=findpeaks(x(:,3),'MinPeakProminence',0.03);
onset_time = times*tstep;


% Calculate Offset Times
[pks2,times2]=findpeaks(-x(:,3),'MinPeakProminence',0.03);
offset_time = times2*tstep;


% Seizure Lengths
seizure_length = [];
for i=1:(length(onset_time)-1)
    if offset_time(1)>onset_time(1)
        seizure_length = [seizure_length, offset_time(i)-onset_time(i)];
    else
        seizure_length = [seizure_length, offset_time(i+1)-onset_time(i)];
    end
end


SD1 = std(seizure_length);
R1 = SD1/mean(seizure_length);


% Single seizure
if offset_time(1)>onset_time(1) % if system starts at rest
    start_index = times(1)-10000;
    stop_index = times2(1)+10000;
    start_index = max(start_index, 1);
    seizure = x(start_index:stop_index,1);
    onset = 10000;
    offset = stop_index-start_index-10000;
else % if system starts in a seizure
    start_index = times(1)-10000;
    stop_index = times2(2)+10000;
    start_index = max(start_index, 1);
    seizure = x(start_index:stop_index,1);
    onset = 10000;
    offset = stop_index-start_index-10000;
end


    Seizures{counter} = seizure;
    counter = counter + 1;
end
end

% ── layout parameters ─────────────────────────────────────────────────
nPerBlock = 5;
nBlocks   = 3;

padBlock  = 0.09;   % margin on left/right and top/bottom
padSmallX = 0.02;   % gap between the two columns INSIDE each block
padSmallY = 0.02;   % gap between the three rows INSIDE each block

% compute widths for two‐column layout across all blocks:
halfW  = (1 - (nBlocks+1)*padBlock - nBlocks*padSmallX) / (2*nBlocks);
blockW = 2*halfW + padSmallX;

% compute heights so that top margin, bottom margin, and 2 inter-row gaps
% are taken into account, filling full [0,1]:
smallH = (1 - 2*padBlock - 2*padSmallY)/3;

% y‐positions of the three rows (top of axes at yTop+smallH):
yTop = 1 - padBlock - smallH;
yMid = yTop - smallH - padSmallY;
yBot = yMid - smallH - padSmallY;  % this bottom now sits at padBlock

% ── draw ──────────────────────────────────────────────────────────────
figure('Color','w');
set(gcf,'Units','normalized','Position',[.1 .1 .8 .7]);
title(Title(bifurcation));
axis off;
count_seiz = counter - 15;
for b = 1:nBlocks
    x0 = padBlock + (b-1)*(blockW + padBlock);
    
    pos = {
      [x0                , yTop,    halfW, smallH];  % top-left
      [x0+halfW+padSmallX, yTop,    halfW, smallH];  % top-right
      [x0                , yMid,    halfW, smallH];  % mid-left
      [x0+halfW+padSmallX, yMid,    halfW, smallH];  % mid-right
      [x0 + (blockW-halfW)/2, yBot, halfW, smallH];  % bottom-centered
    };
    
    for i = 1:5
        idx = (b-1)*5 + i;
        ax  = axes('Position',pos{i});
        seizure = Seizures{count_seiz};
        count_seiz = count_seiz + 1;
    marker_array_1 = ['o', 'o', '^', '^', 'square'];
marker_array_2 = [ '^', 'o', 'o','^' 'square'];
plot(seizure,'color','#696969','LineWidth',1)
hold on
plot(7500,seizure(7500),'Marker',marker_array_1(i),'MarkerFaceColor','#FFA500','MarkerEdgeColor','#FFA500','MarkerSize',10)
plot(length(seizure)-7500,seizure(length(seizure)-7500),'Marker',marker_array_2(i),'MarkerFaceColor','#1E90FF','MarkerEdgeColor','#1E90FF','MarkerSize',10)
        axis off
    end
end

end


function get_plot()
addpath('Atlas Helper files');
    marker_size = 10;
    load('curves.mat')
    load('curves2.mat')
    load('bifurcation_crossing.mat')
    load('sphere_mesh.mat')
    load('testmesh.mat');

    hold on;
    linewidth = 2;

    % Plot different meshes with assigned DisplayName for the legend
    vertices = BCSmesh.vertices;
    faces = BCSmesh.faces;
    h1 = patch('Vertices', vertices, 'Faces', faces, ...
          'FaceColor', [0.973, 0.965, 0.722], 'EdgeColor', 'none', 'FaceAlpha', 0.3, 'DisplayName', 'BCS Mesh');

    vertices = Active_restmesh.vertices;
    faces = Active_restmesh.faces;
    h2 = patch('Vertices', vertices, 'Faces', faces, ...
          'FaceColor', [0.9216, 0.9216, 0.9216], 'EdgeColor', 'none', 'FaceAlpha', 0.3, 'DisplayName', 'Active Rest Mesh');

    vertices = Seizure_mesh.vertices;
    faces = Seizure_mesh.faces;
    h3 = trisurf(faces, vertices(:,1), vertices(:,2), vertices(:,3), ...
            'FaceColor', [0.894, 0.706, 0.831], 'EdgeColor', 'none', 'FaceAlpha', 0.3, 'DisplayName', 'Seizure Mesh');

    vertices = Bistable_Lcb_mesh.vertices;
    faces = Bistable_Lcb_mesh.faces;
    h4 = patch('Vertices', vertices, 'Faces', faces, ...
          'FaceColor', [0.973, 0.965, 0.722], 'EdgeColor', 'none', 'FaceAlpha', 0.3, 'DisplayName', 'Bistable Lcb Mesh');

    scale_array = [0.4];
    scale_array = scale_array / 0.4;

    % Arrays to store plot handles for the legend
    legend_handles = [h1, h2, h3];
    legend_names = {'Bistable region', 'Active Rest region', 'Seizure region'};

    % Scale factors to adjust radius from 0.4
    for i = 1:length(scale_array)
        % Scale the coordinates of the points for radius 0.39
        Fold_of_cycles_scaled = scale_array(i) * Fold_of_cycles;
        Homoclinic_to_saddle3_scaled = scale_array(i) * Homoclinic_to_saddle3;
        Homoclinic_to_saddle2_scaled = scale_array(i) * Homoclinic_to_saddle2;
        Homoclinic_to_saddle1_scaled = scale_array(i) * Homoclinic_to_saddle1;
        Homoclinic_to_saddle_scaled = scale_array(i) * Homoclinic_to_saddle;
        Fold_scaled = scale_array(i) * Fold;
        Hopf_scaled = scale_array(i) * Hopf;
        SNIC_scaled = scale_array(i) * SNIC;

        % Plot all scaled lines with DisplayName for the legend
        h5 = plot3(Fold_of_cycles_scaled(1, :), Fold_of_cycles_scaled(2, :), Fold_of_cycles_scaled(3, :), 'Color', [0.9725,0.2667,0.5843], 'LineWidth', linewidth, 'DisplayName', 'Fold of Cycles');
        h6 = plot3(Homoclinic_to_saddle3_scaled(1, :), Homoclinic_to_saddle3_scaled(2, :), Homoclinic_to_saddle3_scaled(3, :), 'Color', [0.404, 0.702, 0.851], 'LineWidth', linewidth, 'LineStyle', '--', 'DisplayName', 'Homoclinic to Saddle 3');
        h7 = plot3(Homoclinic_to_saddle2_scaled(1, :), Homoclinic_to_saddle2_scaled(2, :), Homoclinic_to_saddle2_scaled(3, :), 'Color', [0.404, 0.702, 0.851],  'LineWidth', linewidth, 'DisplayName', 'Homoclinic to Saddle 2');
        h8 = plot3(Homoclinic_to_saddle1_scaled(1, :), Homoclinic_to_saddle1_scaled(2, :), Homoclinic_to_saddle1_scaled(3, :), 'Color', [0.404, 0.702, 0.851],  'LineWidth', linewidth, 'LineStyle', '--', 'DisplayName', 'Homoclinic to Saddle 1');
        h9 = plot3(Homoclinic_to_saddle_scaled(1, :), Homoclinic_to_saddle_scaled(2, :), Homoclinic_to_saddle_scaled(3, :), 'Color', [0.404, 0.702, 0.851],  'LineWidth', linewidth, 'DisplayName', 'Homoclinic to Saddle');
        h10 = plot3(Fold_scaled(1, 140:564), Fold_scaled(2, 140:564), Fold_scaled(3, 140:564), 'Color', [0.957, 0.612, 0.204], 'LineWidth', linewidth, 'DisplayName', 'Fold Part 1');
        h11 = plot3(Fold_scaled(1, 575:end), Fold_scaled(2, 575:end), Fold_scaled(3, 575:end), 'Color', [0.957, 0.612, 0.204], 'LineWidth', linewidth, 'DisplayName', 'Fold Part 2');
        h12 = plot3(Fold_scaled(1, 1:80), Fold_scaled(2, 1:80), Fold_scaled(3, 1:80), 'Color', [0.957, 0.612, 0.204], 'LineWidth', linewidth, 'DisplayName', 'Fold Part 3');
        h13 = plot3(Hopf_scaled(1, 1:400), Hopf_scaled(2, 1:400), Hopf_scaled(3, 1:400), 'Color',[0.4549 ,0.7490 ,0.2706], 'LineWidth', linewidth, 'LineStyle', '--', 'DisplayName', 'Hopf Part 1');
        h14 = plot3(Hopf_scaled(1, 400:973), Hopf_scaled(2, 400:973), Hopf_scaled(3, 400:973), 'Color',[0.4549 ,0.7490 ,0.2706], 'LineWidth', linewidth, 'DisplayName', 'Hopf Part 2');
        h15 = plot3(SNIC_scaled(1, :), SNIC_scaled(2, :), SNIC_scaled(3, :), 'Color',[0.957, 0.612, 0.204], 'LineWidth', linewidth, 'LineStyle', '--', 'DisplayName', 'SNIC');

        % Add line handles to the legend array
        legend_handles = [legend_handles, h5, h6,  h9,  h12, h13, h14, h15];
        legend_names = [legend_names, 'Fold of Cycles', 'subSH', 'SH', 'Fold', 'SubH', 'SupH', 'SNIC'];
    end

    % Add the sphere mesh with transparency
    surf(X_sphere, Y_sphere, Z_sphere, 'FaceColor', [0.96, 0.96, 0.86], 'FaceAlpha', 0.2, 'EdgeColor', 'none', 'HandleVisibility', 'off');

    % Add labels and view adjustment
    xlabel('\mu_2');
    ylabel('-\mu_1');
    zlabel('\nu');
    
    lineVector = [-0.19, 0.2, 0.07];
    az = atan2d(lineVector(2), lineVector(1)); % Azimuth angle
    el = atan2d(lineVector(3), norm(lineVector(1:2))); % Elevation angle
    view(az, el);

    % Display the legend with all handles and names
    legend(legend_handles, legend_names);
end



function [Xdot, mu2, mu1,nu] = HysteresisLoop_Model(~,x,~,k,R,dstar,E,F,N)
    
    % Parametrization of the path in the spherical parameter space in terms of great
    % circles      
    mu2=R*(E(1)*cos(x(3))+F(1)*sin(x(3)));
    mu1=-R*(E(2)*cos(x(3))+F(2)*sin(x(3)));
    nu=R*(E(3)*cos(x(3))+F(3)*sin(x(3)));

    % x coordinate of resting state (i.e. upper branch of eq)     
    x_rs=real(Resting_State(mu2,mu1,nu,N));
    
    % equations    
    xdot = - x(2);
    ydot = x(1)^3 - mu2*x(1) - mu1 - x(2)*( nu + x(1) + x(1)^2);
    
    zdot =  -k*(sqrt((x(1)-x_rs)^2+x(2)^2)-dstar);
    
    Xdot = [xdot;ydot;zdot];

end  

function [E,F] = Parametrization_2PointsArc(A,B,R)

    E = A./R;

    F=cross(cross(A,B),A);
    F=F./norm(F);

end

function x = pinknoise(DIM,BETA, MAG),
% This function generates 1/f spatial noise, with a normal error
% distribution 
%
% DIM is a two component vector that sets the size of the spatial pattern
%       (DIM=[10,5] is a 10x5 spatial grid)
%
% BETA defines the spectral distribution.
%      Spectral density S(f) = N f^BETA
%      (f is the frequency, N is normalisation coeff).
%           BETA = 0 is random white noise.  
%           BETA  -1 is pink noise
%           BETA = -2 is Brownian noise
%      The fractal dimension is related to BETA by, D = (6+BETA)/2  
%
% MAG is the scaling variable for the noise amplitude
%
% The method is briefly descirbed in Lennon, J.L. "Red-shifts and red
% herrings in geographical ecology", Ecography, Vol. 23, p101-113 (2000)
rng(720);
u = [(0:floor(DIM(1)/2)) -(ceil(DIM(1)/2)-1:-1:1)]'/DIM(1);
u = repmat(u,1,DIM(2));
v = [(0:floor(DIM(2)/2)) -(ceil(DIM(2)/2)-1:-1:1)]/DIM(2);
v = repmat(v,DIM(1),1);
S_f = (u.^2 + v.^2).^(BETA/2);
S_f(S_f==inf) = 0;
phi = rand(DIM);
y= S_f.^0.5 .* (cos(2*pi*phi)+i*sin(2*pi*phi));
y=y.*MAG/max(abs(y));  %set the mag to the level you want
x= ifft2(y);
x = real(x);

end

function x_rs = Resting_State(mu2,mu1,nu,N)

    switch N
        case 1 % resting state on upper branch
            x_rs=mu2/(3*(mu1/2 + (mu1^2/4 - mu2^3/27)^(1/2))^(1/3)) + (mu1/2 + (mu1^2/4 - mu2^3/27)^(1/2))^(1/3);
    
        case 2 % resting state on lower branch
            x_rs=- mu2/(6*(mu1/2 + (mu1^2/4 - mu2^3/27)^(1/2))^(1/3)) - (mu1/2 + (mu1^2/4 - mu2^3/27)^(1/2))^(1/3)/2 - (3^(1/2)*i*(mu2/(3*(mu1/2 + (mu1^2/4 - mu2^3/27)^(1/2))^(1/3)) - (mu1/2 + (mu1^2/4 - mu2^3/27)^(1/2))^(1/3)))/2;
     
        case 3
            x_rs= (3^(1/2)*i*(mu2/(3*(mu1/2 + (mu1^2/4 - mu2^3/27)^(1/2))^(1/3)) - (mu1/2 + (mu1^2/4 - mu2^3/27)^(1/2))^(1/3)))/2 - (mu1/2 + (mu1^2/4 - mu2^3/27)^(1/2))^(1/3)/2 - mu2/(6*(mu1/2 + (mu1^2/4 - mu2^3/27)^(1/2))^(1/3));
    end

end
function [mu2,mu1,nu] = sphere(point1, point2, numPoints)
    % sphereArcPath - Generates an arc path between two points on a sphere
    %
    % Syntax: arcPath = sphereArcPath(point1, point2, numPoints)
    %
    % Inputs:
    %    point1 - [x1, y1, z1] Coordinates of the first point on the sphere
    %    point2 - [x2, y2, z2] Coordinates of the second point on the sphere
    %    numPoints - Number of points along the arc
    %
    % Outputs:
    %    arcPath - An Nx3 matrix containing the coordinates of points along the arc
    
    % Check the input points
    radius = 0.4;
    % if norm(point1) ~= radius || norm(point2) ~= radius
    %     error('The points must lie on the sphere of radius 0.4.');
    % end
    
    % Normalize the input points to make sure they are on the sphere
    point1 = point1 / norm(point1) * radius;
    point2 = point2 / norm(point2) * radius;
    
    % Compute the quaternion for rotation
    theta = acos(dot(point1, point2) / (radius^2));
    axis = cross(point1, point2);
    if norm(axis) == 0
        error('The points are the same or antipodal.');
    end
    axis = axis / norm(axis);

    % Compute points along the arc
    arcPath = zeros(numPoints, 3);
    for i = 0:numPoints-1
        t = i / (numPoints - 1);
        angle = t * theta;
        R = rotationMatrix(axis, angle);
        arcPath(i+1, :) = (R * point1')';
    end
    mu2 = arcPath(:,1)';
    mu1 = arcPath(:,2)';
    nu = arcPath(:,3)';
end


function movedPoints = movePointsFromCenter(points, distance)
    % Function to move each point in Cartesian coordinates away from the center
    % by a specified distance
    %
    % Parameters:
    % points - A 3xN matrix where each column is a point in 3D Cartesian space
    % distance - The distance to move each point away from the center

    % Calculate the current distances from the origin
    currentDistances = sqrt(sum(points.^2, 1));
    
    % Calculate the scale factors
    scaleFactors = (currentDistances + distance) ./ currentDistances;
    
    % Scale the points
    movedPoints = points .* scaleFactors;
end

function [Xdot, mu2,mu1,nu] = SlowWave_Model(~,x,~,k,E,F,C,r)
    
    % Parametrization of the path in the spherical parameter space in terms
    % of a circle defined by 3 points
    mu2=C(1)+r*(E(1)*cos(x(3))+F(1)*sin(x(3)));
    mu1=-(C(2)+r*(E(2)*cos(x(3))+F(2)*sin(x(3))));
    nu=C(3)+r*(E(3)*cos(x(3))+F(3)*sin(x(3)));
 
    % System
    xdot = - x(2);
    ydot = x(1)^3 - mu2*x(1) - mu1 - x(2)*( nu + x(1) + x(1)^2);
    zdot = k;
   
    Xdot = [xdot;ydot;zdot];
end

function point= get_random_point_c3
data = [
0.3629, 0.1620,-0.0456
0.3974,0.0406,-0.0215
-0.2810, 0.0823,-0.2726
-0.2482,0.3136,0.0059
-0.1686,0.2890,-0.2192
0.2715,0.02934,0.0128
0.3671,0.1517,-0.0472
-0.2868,0.0004,0.2788
0.2613,0.2769,-0.1227
0.1705,0.3615,-0.0173
-0.2139,0.1699,-0.2922
-0.1891,0.3299,-0.1242
0.2426,0.2843,0.1425
  -0.0640  ,  0.3929  , -0.0389
    0.2279  ,  0.0155 ,  -0.3284
    0.1286 ,   0.3034   , 0.2267
    0.0187  ,  0.3075  ,  0.2552
   -0.0640 ,   0.3929 ,  -0.0389

];
idx = randi([1,length(data)]);
point = data(idx,:);
end
function point= get_random_point_c7_c9
data = [
%0.157119269030135	0.161091875334199	0.330700684911216
-0.159101878627765	0.366197019600565	0.0242143563362679
0.240325089686399	0.304071689348835	-0.0989154133781482
0.380078172355370	0.116531602029670	-0.0442828254225105
0.376085815202690	0.0636356567306722	-0.120457306942271
0.166174992977878	0.327294142268742	0.158947840958507
0.172012914508465	0.357581320788856	-0.0504693596670386
-0.0758915517458272	0.306208966174370	0.245919786532182
-0.0486739184007403	0.108823597501720	-0.381822307224580
0.0796165762275659	0.325569951814250	-0.218324087689533
0.251981235011323	0.310614829711489	0.00488720425919465
-0.232117006686916	0.282224648390582	-0.162698933762698
%0.220275294465348	0.201237672469794	0.266424837106348
-0.332207536237643	0.183689008934820	0.126081326386816
%0.147579315557777	0.0266736449114354	0.370821874067101
%-0.243972195896027	0.0461366053360331	0.313606411410446
0.312005760944042	0.232343423323037	-0.0931071360114856
%-0.183233489251201	0.115164818276131	0.336396422466494
%0.161849841530673	0.129909400887954	0.341947622242689
-0.285057819447844	0.276172527226907	-0.0497068886246541
0.293516972528577	0.262050399776243	0.0719539770601309
0.100452183568069	0.133430682841316	-0.363463356740273
%0.0587657128391141	0.273279906407037	0.286120051287343
0.135209543713507	0.118213069098463	-0.357412995822910
%-0.399343120628290	0.0156914048904897	-0.0166988568297806
-0.208541275002113	0.323171146651893	0.109867859686961
-0.0449185896441651	0.258992664847436	-0.301504759265261
0.0449357963445848	0.160975079585810	-0.363411334384634
0.0462987815061658	0.0359526389944827	-0.395681476165333
-0.287432027169657	0.213474519904710	-0.178357671852362
%-0.0334309018751504	0.251955318843665	0.308870348376469
0.211875145592833	0.322495970331763	-0.105381553413254
0.350411490578410	0.149947999830475	0.121356436242404
-0.117148812770085	0.265217051374884	-0.275565003813205
%-0.190205660482966	0.250539203555510	0.247086855583233
0.232946913731378	0.315457300993538	0.0788823594531643
0.0427771439339962	0.351925488543204	-0.185252709752050
0.296164473879145	0.244426175265017	0.111993076827116
0.142393319279842	0.371620981937393	0.0402739172214049
0.111421877013698	0.0969703711017663	-0.371728277712538
%0.384674464390556	0.0201207759882954	0.107799400830745
0.0946871119121513	0.381474943125708	0.0742375821600734
%-0.359866162303914	0.0769286748857566	0.156774756287403
0.0536941087296937	0.387454928395370	0.0836398299248805
%0.0544849486836204	0.0436414639382243	0.393861413434055
%-0.128741399122706	0.0344709001636801	0.377143751365224
0.288413853864687	0.240285781103050	0.138131069273411
%-0.0175431145198549	0.0209970586931555	0.399063106111276
%-0.0133870427997347	0.262216240418017	0.301767179040596
-0.0940903089878144	0.357288847540074	0.153270000907104
];
idx = randi([1,length(data)]);
point = data(idx,:);
end


function [E,F,C,r]=Parametrization_3PointsCircle(p1,p2,p3)
    
    syms x y z
    
    V12=(p1-p2)/norm(p1-p2);
    V13=(p1-p3)/norm(p1-p3);
    
    n=cross(V12,V13);
    n=n/norm(n);

    dalpha=sum(p1.*n);
    alpha=n(1)*x+n(2)*y+n(3)*z-dalpha;

    hp1p2=p1+(p2-p1)/2; % punto medio
    dbeta=sum(V12.*hp1p2);
    beta=V12(1)*x+V12(2)*y+V12(3)*z-dbeta;

    hp1p3=p1+(p3-p1)/2; % punto medio
    dgamma=sum(V13.*hp1p3);
    gamma=V13(1)*x+V13(2)*y+V13(3)*z-dgamma;
    
    sx=solve(alpha==0,x);
    s1=subs(beta,x,sx);
    sy=solve(s1==0,y);
    s2=subs(gamma,x,sx);
    s2=subs(s2,y,sy);
    Cz=solve(s2==0,z);
    Cy=subs(sy,z,Cz);
    Cx=subs(sx,y,Cy);
    Cx=subs(Cx,z,Cz);
    C=[Cx, Cy, Cz];
    C=eval(C);     
    
    E=(p1-C)/norm(p1-C);
    F=-cross(E,n);
    r=norm(p1-C);

end

function [Xdot, mu2,mu1,nu] = SlowWave_Model2(~,x,~,k,mu2,mu1,nu)
% Parametrization of the path in the spherical parameter space in terms
% of a circle defined by 3 points
% System
xdot = - x(2);
ydot = x(1)^3 - mu2*x(1) - mu1 - x(2)*( nu + x(1) + x(1)^2);
zdot = k;
Xdot = [xdot;ydot;zdot];
end
function [mu2,mu1,nu, theta] = sphereArcPath(k, tstep,point1, point2)
% sphereArcPath - Generates an arc path between two points on a sphere
%
% Syntax: arcPath = sphereArcPath(point1, point2, numPoints)
%
% Inputs:
% point1 - [x1, y1, z1] Coordinates of the first point on the sphere
% point2 - [x2, y2, z2] Coordinates of the second point on the sphere
% numPoints - Number of points along the arc
%
% Outputs:
% arcPath - An Nx3 matrix containing the coordinates of points along the arc
% Check the input points
radius = 0.4;
% if norm(point1) ~= radius || norm(point2) ~= radius
% error('The points must lie on the sphere of radius 0.4.');
% end
% Normalize the input points to make sure they are on the sphere
point1 = point1 / norm(point1) * radius;
point2 = point2 / norm(point2) * radius;
% Compute the quaternion for rotation
theta = acos(dot(point1, point2) / (radius^2));
axis = cross(point1, point2);
if norm(axis) == 0
error('The points are the same or antipodal.');
end
axis = axis / norm(axis);
% Compute points along the arc
numPoints = floor((theta/k)/tstep);
arcPath = zeros(numPoints, 3);
for i = 0:numPoints-1
t = i / (numPoints - 1);
angle = t * theta;
R = rotationMatrix(axis, angle);
arcPath(i+1, :) = (R * point1')';
end
mu2 = arcPath(:,1)';
mu1 = arcPath(:,2)';
nu = arcPath(:,3)';
end
function R = rotationMatrix(axis, angle)
    % rotationMatrix - Generates a rotation matrix given an axis and an angle
    %
    % Syntax: R = rotationMatrix(axis, angle)
    %
    % Inputs:
    %    axis - A 3-element vector representing the axis of rotation
    %    angle - The angle of rotation in radians
    %
    % Outputs:
    %    R - A 3x3 rotation matrix

    ux = axis(1);
    uy = axis(2);
    uz = axis(3);

    c = cos(angle);
    s = sin(angle);
    t = 1 - c;

    R = [t*ux*ux + c,    t*ux*uy - s*uz, t*ux*uz + s*uy;
         t*ux*uy + s*uz, t*uy*uy + c,    t*uy*uz - s*ux;
         t*ux*uz - s*uy, t*uy*uz + s*ux, t*uz*uz + c];
end
function point= get_random_point
radius = 0.4;
% Generate two random numbers
theta = 2 * pi * rand(); % Random angle between 0 and 2*pi
phi = acos(2 * rand() - 1); % Random angle between 0 and pi
% Convert spherical coordinates to Cartesian coordinates
x = radius * sin(phi) * cos(theta);
y = radius * sin(phi) * sin(theta);
z = radius * cos(phi);
point = [x,y,z];
% Display the point
end
function point= get_random_point_hopf
%load("map_regions.mat")
radius = 0.4;
% Loop until a valid point with y > 0 is found
while true
% Generate two random numbers
theta = 2 * pi * rand(); % Random angle between 0 and 2*pi
phi = acos(2 * rand() - 1); % Random angle between 0 and pi
% Convert spherical coordinates to Cartesian coordinates
x = radius * sin(phi) * cos(theta);
y = radius * sin(phi) * sin(theta);
z = radius * cos(phi);
% Check if y is positive
if y > 0
point = [x, y, z];
break;
end
end
% Display the point
end
% 
function point= get_random_point_c9
data = [
%0.157119269030135	0.161091875334199	0.330700684911216
-0.159101878627765	0.366197019600565	0.0242143563362679
0.240325089686399	0.304071689348835	-0.0989154133781482
0.380078172355370	0.116531602029670	-0.0442828254225105
0.376085815202690	0.0636356567306722	-0.120457306942271
0.166174992977878	0.327294142268742	0.158947840958507
0.172012914508465	0.357581320788856	-0.0504693596670386
-0.0758915517458272	0.306208966174370	0.245919786532182
-0.0486739184007403	0.108823597501720	-0.381822307224580
0.0796165762275659	0.325569951814250	-0.218324087689533
0.251981235011323	0.310614829711489	0.00488720425919465
-0.232117006686916	0.282224648390582	-0.162698933762698
%0.220275294465348	0.201237672469794	0.266424837106348
-0.332207536237643	0.183689008934820	0.126081326386816
%0.147579315557777	0.0266736449114354	0.370821874067101
%-0.243972195896027	0.0461366053360331	0.313606411410446
0.312005760944042	0.232343423323037	-0.0931071360114856
%-0.183233489251201	0.115164818276131	0.336396422466494
%0.161849841530673	0.129909400887954	0.341947622242689
-0.285057819447844	0.276172527226907	-0.0497068886246541
0.293516972528577	0.262050399776243	0.0719539770601309
0.100452183568069	0.133430682841316	-0.363463356740273
%0.0587657128391141	0.273279906407037	0.286120051287343
0.135209543713507	0.118213069098463	-0.357412995822910
%-0.399343120628290	0.0156914048904897	-0.0166988568297806
-0.208541275002113	0.323171146651893	0.109867859686961
-0.0449185896441651	0.258992664847436	-0.301504759265261
0.0449357963445848	0.160975079585810	-0.363411334384634
0.0462987815061658	0.0359526389944827	-0.395681476165333
-0.287432027169657	0.213474519904710	-0.178357671852362
%-0.0334309018751504	0.251955318843665	0.308870348376469
0.211875145592833	0.322495970331763	-0.105381553413254
0.350411490578410	0.149947999830475	0.121356436242404
-0.117148812770085	0.265217051374884	-0.275565003813205
%-0.190205660482966	0.250539203555510	0.247086855583233
0.232946913731378	0.315457300993538	0.0788823594531643
0.0427771439339962	0.351925488543204	-0.185252709752050
0.296164473879145	0.244426175265017	0.111993076827116
0.142393319279842	0.371620981937393	0.0402739172214049
0.111421877013698	0.0969703711017663	-0.371728277712538
%0.384674464390556	0.0201207759882954	0.107799400830745
0.0946871119121513	0.381474943125708	0.0742375821600734
%-0.359866162303914	0.0769286748857566	0.156774756287403
0.0536941087296937	0.387454928395370	0.0836398299248805
%0.0544849486836204	0.0436414639382243	0.393861413434055
%-0.128741399122706	0.0344709001636801	0.377143751365224
0.288413853864687	0.240285781103050	0.138131069273411
%-0.0175431145198549	0.0209970586931555	0.399063106111276
%-0.0133870427997347	0.262216240418017	0.301767179040596
-0.0940903089878144	0.357288847540074	0.153270000907104
];
idx = randi([1,length(data)]);
point = data(idx,:);
end
function point= get_random_point_c10
data = [
    0.139443244717430	0.180180368497788	-0.328771373922177;
-0.123686721647726	0.338825918756816	-0.172912092308889;
-0.100168545573643	0.0377628990918715	-0.385409166899074;
0.377957700206015	0.0522413891185729	-0.120078366569263;
0.0670217468691031	0.393741391170461	-0.0218128935669752;
0.165551316967523	0.0423021044581705	0.361667379519813;
-0.190629713140052	0.100956538347682	-0.336850248379001;
0.237742614103209	0.317715346392282	-0.0503528361287097;
0.133352560213168	0.205062052013036	0.316491152338887;
0.116930014179559	0.380594524578872	0.0384080674409766;
    0.247294091034076	0.313367115323397	0.0254299739190403;
0.319668729544415	0.149389595714773	0.188400244277008;
%-0.265630076158132	0.280893029748762	-0.102663374573342;
0.206794351734318	0.328104632230012	0.0978950785279422;
    0.0778563146462470	0.373124628962106	-0.121311192936165;
0.0369732031123261	0.391492997572173	0.0732544544964284;
       -0.1050    0.3835   -0.0436;
        0.2092    0.3125    0.1365;
        0.3456    0.1805    0.0896;
    0.3221    0.2370    0.0071;
%0.0963164157568826	0.388100733721629	0.0100483103296496;
%-0.0137890033618096	0.234736675106870	-0.323587015725384;
-0.198974692220617	0.0411752373414647	-0.344548504111663;
];
idx = randi([1,length(data)]);
point = data(idx,:);
end
function point= get_random_point_c11
data = [
0.0373, 0.2497, -0.3102;
-0.0441, 0.2591, -0.3015;
-0.2104, 0.3180, -0.1209;
% 0.3475, 0.1347, 0.1452;
0.0806, 0.1560, -0.3594;
% 0.1743, 0.2562, 0.2529;
-0.0564, 0.3820, 0.1042;
0.3196, 0.2389, -0.0279;
-0.0564, 0.3820, 0.1042;
0.3196, 0.2389, -0.0279;
0.2394, 0.3202, -0.0114;
0.0612, 0.3715, 0.1350;
0.3186, 0.1795, -0.1620;
-0.1300, 0.2586, -0.2761;
0.1833, 0.2218, -0.2779;
-0.3059, 0.1997, -0.1629;
0.2071, 0.2684, -0.2124;
-0.0467, 0.3294, -0.2220;
-0.2104, 0.3180, -0.1209;
-0.2316, 0.3055, 0.1142;
0.1314, 0.3298, -0.1843
];
idx = randi([1,length(data)]);
point = data(idx,:);
end
% Display the point
function point= get_random_point_fixed
radius = 0.4;
% Loop until a valid point with y > 0 is found
while true
% Generate two random numbers
theta = 2 * pi * rand(); % Random angle between 0 and 2*pi
phi = acos(2 * rand() - 1); % Random angle between 0 and pi
% Convert spherical coordinates to Cartesian coordinates
x = radius * sin(phi) * cos(theta);
y = radius * sin(phi) * sin(theta);
z = radius * cos(phi);
% Check if y is positive
if y < 0 
point = [x, y, z];
break;
end
end
% Display the point
end
function bool= Bif_bool(mu2,mu1,nu)
addpath('Atlas Helper files');
load('curves2.mat');
load('bifurcation_crossing.mat');
% plot3(FLC(1, :), FLC(2, :), FLC(3, :), 'm-', 'LineWidth', 2);
% %
% plot3(Fold(1, 1:95), Fold(2, 1:95), Fold(3, 1:95), 'm-', 'LineWidth', 2);
% plot3(Fold(1, 145:end), Fold(2, 145:end), Fold(3, 145:end), 'm-', 'LineWidth', 2);
% plot3(SNIC(1, :), SNIC(2, :), SNIC(3, :), 'g-', 'LineWidth', 2);
% plot3(Hopf(1, :), Hopf(2, :), Hopf(3, :), 'm-', 'LineWidth', 2);
% %
%
% plot3(Homoclinic_to_saddle(1, :), Homoclinic_to_saddle(2, :), Homoclinic_to_saddle(3, :), 'm-', 'LineWidth', 2);
% plot3(Homoclinic_to_saddle1(1, :), Homoclinic_to_saddle1(2, :), Homoclinic_to_saddle1(3, :), 'm-', 'LineWidth', 2);
% plot3(Homoclinic_to_saddle2(1, :), Homoclinic_to_saddle2(2, :), Homoclinic_to_saddle2(3, :), 'm-', 'LineWidth', 2);
% plot3(Homoclinic_to_saddle3(1, 1:90), Homoclinic_to_saddle3(2, 1:90), Homoclinic_to_saddle3(3, 1:90), 'm-', 'LineWidth', 2);
bool = false;
%no_hopf
matrix_cell = [FLC, Fold(:,1:95), Fold(:,145:682), SNIC, Homoclinic_to_saddle, Homoclinic_to_saddle1, Homoclinic_to_saddle2, Homoclinic_to_saddle3(:,1:90) ]';
points = [mu2;-mu1;nu]';
%Homoclinic_to_saddle3, Homoclinic_to_saddle2, Homoclinic_to_saddle1, Homoclinic_to_saddle
for i = 1:size(points, 1)
point_to_check = points(i, :);
% Initialize a flag for match
match_found = false;
% Loop through matrix_cell to check for a match
for j = 1:size(matrix_cell, 1)
if all(round(matrix_cell(j, :), 2) == round(point_to_check, 2))
match_found = true;
break; % Exit the loop if a match is found
end
end
if match_found
bool = true;
break; % Exit the outer loop if a match is found
end
end
end
function point= get_nearest_hopf(mu2,mu1,nu)
load('curves2.mat');
load('bifurcation_crossing.mat');
% plot3(FLC(1, :), FLC(2, :), FLC(3, :), 'm-', 'LineWidth', 2);
% %
% plot3(Fold(1, 1:95), Fold(2, 1:95), Fold(3, 1:95), 'm-', 'LineWidth', 2);
% plot3(Fold(1, 145:end), Fold(2, 145:end), Fold(3, 145:end), 'm-', 'LineWidth', 2);
% plot3(SNIC(1, :), SNIC(2, :), SNIC(3, :), 'g-', 'LineWidth', 2);
% plot3(Hopf(1, :), Hopf(2, :), Hopf(3, :), 'm-', 'LineWidth', 2);
% %
%
% plot3(Homoclinic_to_saddle(1, :), Homoclinic_to_saddle(2, :), Homoclinic_to_saddle(3, :), 'm-', 'LineWidth', 2);
% plot3(Homoclinic_to_saddle1(1, :), Homoclinic_to_saddle1(2, :), Homoclinic_to_saddle1(3, :), 'm-', 'LineWidth', 2);
% plot3(Homoclinic_to_saddle2(1, :), Homoclinic_to_saddle2(2, :), Homoclinic_to_saddle2(3, :), 'm-', 'LineWidth', 2);
% plot3(Homoclinic_to_saddle3(1, 1:90), Homoclinic_to_saddle3(2, 1:90), Homoclinic_to_saddle3(3, 1:90), 'm-', 'LineWidth', 2);
Heatmap = 0;
matrix_cell = {Hopf(:,450:650)};
%Homoclinic_to_saddle3, Homoclinic_to_saddle2, Homoclinic_to_saddle1, Homoclinic_to_saddle
Proximity_vals = zeros(1, length(matrix_cell));
point = [mu2,-mu1,nu];
for i = 1:length(matrix_cell)
matrix = matrix_cell{i};
% Calculate the Euclidean distance from the point to each point in the matrix
%distances = sqrt(sum((matrix - point').^2, 1));
distances = sqrt((matrix(1,:)- point(1)).^2 + (matrix(2,:)- point(2)).^2 + (matrix(3,:)- point(3)).^2);
% Find the minimum distance
Proximity_vals(i) =min(distances);
% Output the minimum distance and the corresponding closest point
%closest_point = matrix(:, idx);
end
[min_dist, idx] = min(Proximity_vals);
point = Hopf(:,idx+450);
end
function [arcLength, theta] = calculateArcLength(P1, P2, radius)
 % calculateArcLength computes the arc length and central angle between two points on a sphere.
 % 
 % Input:
 % P1 - First point [x1, y1, z1]
 % P2 - Second point [x2, y2, z2]
 % radius - Radius of the sphere (default: 0.4 if not provided)
 %
 % Output:
 % arcLength - Arc length between the two points
 % theta - Central angle between the two points in radians
 
 if nargin < 3
 radius = 0.4;
 end
 
 % Compute the dot product of P1 and P2
 dotProduct = dot(P1, P2);
 
 % Compute the magnitudes of P1 and P2
 magnitudeP1 = norm(P1);
 magnitudeP2 = norm(P2);
 
 % Compute the cosine of the central angle
 cosTheta = dotProduct / (magnitudeP1 * magnitudeP2);
 
 % Compute the central angle in radians
 theta = acos(cosTheta);
 
 % Compute the arc length
 arcLength = radius * theta;
end

function [mu2,mu1,nu] = get_seizure_points(P1,P2,P3,k,tstep)
[E,F,C,r]=Parametrization_3PointsCircle(P1,P2,P3);
point1 = P1 - C;
point2 = P2 - C;
point3 = P3-C;
point1 = point1 / norm(point1) * r;
point2 = point2 / norm(point2) * r;

point3 = point3 / norm(point3) * r;
% Compute the quaternion for rotation
theta = acos(dot(point1, point3) / (r^2));
theta = 2*pi - theta;
numPoints = floor((theta/k)/tstep);
% theta = acos(dot(point2, point3) / (r^2));
% numPoints2 = floor((theta/k)/tstep);
N_t = floor(numPoints);
xx = 0;


X= zeros(1, N_t);
mu2_big = zeros(1, N_t);
mu1_big = zeros(1, N_t);
nu_big = zeros(1, N_t);
%[E,F] = Parametrization_2PointsArc(p2,p3,0.4);
for n = 1:N_t
[Fxx, mu2,mu1,nu] = SlowWave_Model(0,xx,0,k,E,F,r);
xx = xx + tstep*Fxx ;
X(:,n) = xx;

mu2_big(n) = mu2;
mu1_big(n) = mu1;
nu_big(n) = nu;
end
mu2_big = mu2_big';
mu1_big = mu1_big';
nu_big = nu_big';
z =X';
mu2=C(1)+r*(E(1)*cos(z)+F(1)*sin(z));
mu1=-(C(2)+r*(E(2)*cos(z)+F(2)*sin(z)));
nu=C(3)+r*(E(3)*cos(z)+F(3)*sin(z));
% mu1 = -mu1;
end

function   [p0,p1,p1_5,p2,p3]=Random_bifurcation_path(bifurcation)

load('curves.mat');
load('bifurcation_crossing.mat')
load("curves2.mat")

if bifurcation==3
    %fixed rest point
    p0 = Hopf(:,930)';
    %bifurcation curve
    randomNumber = randi([145,170]);
    p1 = Fold(:,randomNumber)';
    randomNumber2 = randi([600,750]);
    p1_5 = get_random_point_hopf();
    %bifurcation curve
    p2 = Hopf(:,randomNumber2)' ;
    %fixed rest
    p3 = Hopf(:,930)';

end

if bifurcation==7
    randomNumber = randi([600,750]);
 randomNumber2 = randi([1,44]);
%fixed rest point
p0 = Homoclinic_to_saddle2(:,30)';
%bifurcation curve
p1 =  SNIC(:,randomNumber2)' ;
%random point in limit cycle
p1_5 = get_random_point_c9();
%bifurcation curve
p2 = Hopf(:,randomNumber)';
%fixed rest
p3 = [ 0.1944 , 0.0893 , 0.3380];
end

if bifurcation==9
    randomNumber = randi([600,750]);
    %fixed rest point
    p0 = [ 0.1944 , 0.0893 , 0.3380];
    %bifurcation curve
    p1 = Hopf(:,randomNumber)';
    %change here
    randomNumber2 = randi([1,44]);
    %random point in limit cycle
    p1_5 = get_random_point_c9();
    %bifurcation curve
    p2 = SNIC(:,randomNumber2)' ;
    %fixed rest
    p3 = Fold(:,450)';
end

if bifurcation==10
    randomNumber = randi([600,750]);
    %fixed rest point
    p0 = [ 0.1944 , 0.0893 , 0.3380];
    %bifurcation curve
    p1 = Hopf(:,randomNumber)';%get_nearest_hopf(p0(1),p0(2),p0(3))';
    %change here
    randomNumber2 = randi([1,124]);
    %random point in limit cycle
    p1_5 = get_random_point_c10();
    %bifurcation curve
    p2 = Homoclinic_to_saddle(:,randomNumber2)' ;
    %fixed rest
    p3 = SNIC(:,30)';
end

if bifurcation==11
     randomNumber = randi([600,750]);
%fixed rest point
p0 = [ 0.1944 , 0.0893 , 0.3380];
%bifurcation curve
p1 = Hopf(:,randomNumber)';
%change here
randomNumber2 = randi([600,750]);
%random point in limit cycle
p1_5 = get_random_point_c11();
%bifurcation curve
p2 = Hopf(:,randomNumber2)' ;
%fixed rest
p3 = [ 0.1944 , 0.0893 , 0.3380];
end



end
