% Copyright 2023 The CARFAC Authors. All Rights Reserved.
% Author: Matt R. Flax, setup taken from Richard F. Lyon's CARFAC_Design.m
%
% This file is part of an implementation of Lyon's cochlear model:
% "Cascade of Asymmetric Resonators with Fast-Acting Compression"
%
% Licensed under the Apache License, Version 2.0 (the "License");
% you may not use this file except in compliance with the License.
% You may obtain a copy of the License at
%
%     http://www.apache.org/licenses/LICENSE-2.0
%
% Unless required by applicable law or agreed to in writing, software
% distributed under the License is distributed on an "AS IS" BASIS,
% WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
% See the License for the specific language governing permissions and
% limitations under the License.

function [CF, b, a] = CAR_Design(n_ears, fs, CF_CAR_params)
% function CF = CAR_Design(n_ears, fs, CF_CAR_params)
%
% This function designs the linear open loop CAR (Cascade of Asymmetric 
% Resonators) as biquadratic filters.
% That is, it take bundles of parameters and
% computes all the filter coefficients needed to run it.
%
% fs is sample rate (per second)
% CF_CAR_params bundles all the pole-zero filter cascade parameters

if nargin == 0
	CAR_Design_Test;
	return
end

if nargin < 1
	n_ears = 1;
end

if nargin < 2
	fs = 22050;
end

if nargin < 3
	CF_CAR_params = struct( ...
		'velocity_scale', 0.1, ...  % for the velocity nonlinearity
		'v_offset', 0.04, ...  % offset gives a quadratic part
		'min_zeta', 0.10, ... % minimum damping factor in mid-freq channels
		'max_zeta', 0.35, ... % maximum damping factor in mid-freq channels
		'first_pole_theta', 0.85*pi, ...
		'zero_ratio', sqrt(2), ... % how far zero is above pole
		'high_f_damping_compression', 0.5, ... % 0 to 1 to compress zeta
		'ERB_per_step', 0.5, ... % assume G&M's ERB formula
		'min_pole_Hz', 30, ...
		'ERB_break_freq', 165.3, ...  % 165.3 is Greenwood map's break freq.
		'ERB_Q', 1000/(24.7*4.37));  % Glasberg and Moore's high-cf ratio
end

CF = CARFAC_Design(n_ears, fs, CF_CAR_params);
CF = CARFAC_Init(CF);

for ear=1:CF.n_ears
	% This method 
	r1=CF.ears(ear).CAR_coeffs.r1_coeffs
	a0=CF.ears(ear).CAR_coeffs.a0_coeffs;
	h=CF.ears(ear).CAR_coeffs.h_coeffs;
	g=CF.ears(ear).CAR_coeffs.g0_coeffs;
	c=CF.ears(ear).CAR_coeffs.c0_coeffs;
	zB = CF.ears(ear).CAR_state.zB_memory; % current delta-r from undamping
	r = r1 + zB;
	for m=1:CF.n_ch
		b(m,:, ear)=g(m)*[1 (h(m)*r(m)*c(m)-2*r(m)*a0(m)) (r(m).^2*(c(m).^2+a0(m).^2))];
		a(m,:, ear)=[1 -2*r(m)*a0(m) r(m).^2*(c(m).^2+a0(m).^2)];
	end
	
	% This method uses CARFAC_Rational_Functions to generate the IIR filter coefficients
 	% [b(:,:,ear), a(:,:,ear), g(:,ear)] = CARFAC_Rational_Functions(CF, ear);
end
end

function CAR_Design_Test
n_ears=1;
fs=22050;
[~, b, a]=CAR_Design(n_ears, fs);
b=b(:,:,1); % only look at the first ear for now
a=a(:,:,1);
M=size(b,1);
N=fs; % take a one second response
% find the impulse response
x=zeros(N,1);
x(1)=1;
y(:,1)=filter(b(1,:),a(1,:),x);
for m=2:M % ripple through the cascade
	y(:,m)=filter(b(m,:),a(m,:),y(:,m-1));
end
Y=fft(y);

f=linspace(0,fs,N+1); f(end)=[];

figure(1); clf
semilogx(f,20*log10(abs(Y))); grid on;
xlabel('f (Hz)'); ylabel('dB')
title('CAR filters')
% print -depsc /tmp/CAR.DFT.eps

CF=CARFAC_Design(n_ears, fs);
CF.open_loop = 1;  % For measuring impulse response.
CF.linear_car = 1;  % For measuring impulse response.
CF = CARFAC_Init(CF);
[~, CF, bm_initial] = CARFAC_Run_Segment(CF, x);

Yref=fft(bm_initial);

figure(2); clf
semilogx(f,20*log10(abs(Yref))); grid on;
xlabel('f (Hz)'); ylabel('dB')
title('CARFAC reference')
% print -depsc /tmp/CARFAC.DFT.eps

figure(3); clf
semilogx(y); grid on;
xlabel('sample (n)'); ylabel('amp.')
title('CAR filters')
% print -depsc /tmp/CAR.t.eps

figure(4); clf
semilogx(bm_initial); grid on;
xlabel('sample (n)'); ylabel('amp.')
title('CARFAC filters')
% print -depsc /tmp/CARFAC.t.eps

figure(5); clf
semilogx(bm_initial-y); grid on;
xlabel('sample (n)'); ylabel('error')
title('Error between the CARFAC and biquad filters')
% print -depsc /tmp/CARFAC.CAR.error.t.eps

end
