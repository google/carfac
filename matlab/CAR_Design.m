% Copyright 2012 The CARFAC Authors. All Rights Reserved.
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
% This function designs the CAR (Cascade of Asymmetric Resonators);
% that is, it take bundles of parameters and
% computes all the filter coefficients needed to run it.
%
% fs is sample rate (per second)
% CF_CAR_params bundles all the pole-zero filter cascade parameters

if nargin == 0
	CAR_Design_Test;
	return
end

if nargin < 1
	n_ears = 1;  % if more than 1, make them identical channels;
	% then modify the design if necessary for different reasons
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

for ear=1:CF.n_ears
	if 1 % use CARFAC_Rational_Functions to generate the IIR filter coefficients
		[b(:,:,ear), a(:,:,ear)] = CARFAC_Rational_Functions(CF, ear);
	else % reconstruct poles and zeros from section 16.2 in the book
		% this approach requires more work w.r.t. gain
		[r, a0, hc0, g]=deal(CF.ears(ear).CAR_coeffs.r1_coeffs, CF.ears(ear).CAR_coeffs.a0_coeffs, CF.ears(ear).CAR_coeffs.h_coeffs, CF.ears(ear).CAR_coeffs.g0_coeffs);
		% 	relative_undamping = 0;
		% 	g = CARFAC_Stage_g(CF.ears(ear).CAR_coeffs, relative_undamping);

		for m=1:CF.n_ch
			p=r(m)*(a0(m) + j*sin(acos(a0(m))));
			o=a0(m)-hc0(m)/2;
			o=r(m)*(o+j*sin(acos(o)));

			p=[p conj(p)]; % complex conjugate poles and zeros
			o=[o conj(o)];

			a(m,:)=poly(p); % get the a and b polynomials
			b(m,:)=g(m)*poly(o); % apply the gain to the numerator
		end
	end
end
end

function CAR_Design_Test
n_ears=1;
fs=22050;
[~, b, a]=CAR_Design(n_ears, fs);
b=b(:,:,1); % only look at the first ear for now
a=a(:,:,1);
M=size(b,1);
N=fs;
% find the impulse response
x=zeros(N,1);
x(1)=1;
for m=1:M
	y(:,m)=filter(b(m,:),a(m,:),x);
end
Y=fft(y);
f=linspace(0,fs,N+1); f(end)=[];

figure(1); clf
semilogx(f,20*log10(abs(Y))); grid on;
xlabel('f (Hz)'); ylabel('dB')
title('CAR filters')

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
end
