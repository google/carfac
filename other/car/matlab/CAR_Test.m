% Copyright 2012 The CARFAC Authors. All Rights Reserved.
% Author Richard F. Lyon
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

function status = CAR_Test(do_plots)
% CARFAC_TEST returns status = 0 if all tests pass; nonzero if fail,
% and prints messages about tests and failures.  Argument do_plots is
% optional, defaults to 1 (as when executing the file as a script).
% Run CARFAC_Test(0) to suppress plotting.

if nargin < 1, do_plots = 1; end  % Produce plots by default.

% Run tests, and see if any fail (have nonzero status):
status = 0;  % 0 for OK so far; 1 for test fail; 2 for error.
status = status | test_CAR_biquad(do_plots);
report_status(status, 'CAR_Test', 1)
return

function status = test_CAR_biquad(do_plots)
% Test the linear open loop biquadratic implementation of the CAR filters
status = 0; % start in the passing state
addpath ../../../matlab/
n_ears=1;
fs=22050;
[~, b, a]=CAR_Design(n_ears, fs);
b=b(:, :, 1); % only look at the first ear for now
a=a(:, :, 1);
M=size(b, 1);
N=fs; % take a one second response
% find the impulse response
x=zeros(N, 1);
x(1)=1;
y(:, 1)=filter(b(1, :), a(1, :), x);
for m=2:M % ripple through the cascade
	y(:, m)=filter(b(m, :), a(m, :), y(:, m-1));
end

CF=CARFAC_Design(n_ears, fs);
CF.open_loop = 1;  % For measuring impulse response.
CF.linear_car = 1;  % For measuring impulse response.
CF = CARFAC_Init(CF);
[~, CF, bm_initial] = CARFAC_Run_Segment(CF, x);

if do_plots
	Y=fft(y);
	
	f=linspace(0, fs, N+1); f(end)=[];
	
	figure(1); clf
	semilogx(f, 20*log10(abs(Y))); grid on;
	xlabel('f (Hz)'); ylabel('dB')
	title('CAR filters')
	% print -depsc /tmp/CAR.DFT.eps
	
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

tol=1e-10;
if (rms(bm_initial - y) > tol)
	status = 1; % indicate a test fail
	fprintf(1, 'CAR biquad implementation is greater then %e from the cross coupled CAR implementation.\n', tol)
end

report_status(status, 'test_car')
return

function report_status(status, name, extra)
if nargin < 3, extra = 0; end
if extra
  if status
    disp(['FAIL ' name '; at least one test failed.'])
  else
    disp(['PASS ' name '; all tests passed.'])
  end
else
  if status
    disp(['FAIL ' name])
    if status > 1
      disp('(status > 1 => error in test or expected results size)')
    end
  else
    disp(['PASS ' name])
  end
end
return
