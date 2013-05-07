function [data, nFrames, period, nChannels, nSamples, sample_rate] = ...
    AIMCread(filename, strobes_filename)
%[data, nFrames, period, nChannels, nSamples ] = AIMCread( filename)
%
% data ... matrix (or array) of size [nFrames,nChannels,nSamples]
%          in case vertical/horizontal profiles are read, you should use squeeze
% nFrames ... number of frames
% period ... Frame interval in ms
% nChannels ... points on vertical axis of an auditory image
% nSamples ... points on horizontal axis of an auditory image

fid = fopen(filename);

strobes_fid = -1;
if nargin > 1
  strobes_fid = fopen(strobes_filename);
  strobes_raw = fread(strobes_fid, [], 'int32');
end

debug = 0;

nFrames = fread( fid, 1, 'uint32');
period = fread( fid, 1, 'float32'); % Frame period in ms
nChannels = fread( fid, 1, 'uint32'); % vertical axis of an AI
nSamples = fread( fid, 1, 'uint32'); % horizontal axis of an AI
sample_rate = fread(fid, 1, 'float32'); % sample rate of each channel in Hz

if nChannels == 1 % temporal profiles
  data = fread( fid, [nSamples,nFrames], 'float32'); % transposed!
  data = data.';
  data = reshape( data, [nFrames,1,nSamples]); % to have the same dimensions
  % if a 'squeeze' is used, this line has no effect at all
  if debug
   disp('seems to be temporal profiles')
  end
elseif nSamples == 1 % spectral profiles
  data = fread( fid, [nChannels,nFrames], 'float32');  % transposed!
  data = data.';
  %data = reshape( data, [nFrames,nChannels,1]); % has no effect
  if debug
   disp('seems to be spectral profiles')
  end
else % auditory 2d images
  data = zeros(nFrames,nChannels,nSamples);
  for k=1:nFrames % loop since fread cannot return 3d array
    Image = fread(fid, [nSamples,nChannels], 'float32'); % transposed!
    data(k,:,:) = Image.';
  end
  if debug
   disp('seems to be 2d images')
  end
end

fclose(fid);        
