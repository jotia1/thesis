function [ dataset, labels ] = expDecay( starti, endi, xs, ys, ts, ksize, msps, k );
%% EXPDECAY - Given a start and end event, return a matrix of decayed
%           images around each pixel.


%% ALGORITHM
%   Init two 2D array of last spikes in each position
%   Init a [numel(events), 11, 11] dataset result variable
%   Init a [numel(events), 11, 11] labels result variable
%
%   for i in range(numel(xs))
%       add pixel spike time to 2D array
%       Grab an 11x11 around the pixel
%       Add to dataset
%
%       add reverse pixel to other 2D array
%       Grab an 11x11 around it
%       add to labels dataset.

%% CODE
% Constants
DVS_RESOLUTION = 128;
if ~exist('ksize', 'var')
    ksize = 11;
end

% Number of events
if starti > endi;
    disp('Starti cannot be less than endi');
    return
end
nevents = endi - starti;


%   Init two 2D array of last spikes in each position
lastSpikeTimesPast = zeros(DVS_RESOLUTION, DVS_RESOLUTION);
lastSpikeTimesFutr = zeros(DVS_RESOLUTION, DVS_RESOLUTION);

%   Init a [numel(events), 11, 11] dataset and labels result variables
dataset = single(zeros(nevents, ksize, ksize));
labels = single(zeros(nevents, ksize, ksize));

% Ensure I can safely index spike times
offset = floor(ksize / 2);
lastSpikeTimesPast = padarray(lastSpikeTimesPast, [offset offset]);
lastSpikeTimesFutr = padarray(lastSpikeTimesFutr, [offset offset]);

%
%   for i in range(numel(xs))
cspikep = starti;
cspikef = endi;
while cspikep <= endi;
%       add pixel spike time to 2D array
        x = xs(cspikep);
        y = ys(cspikep);
        t = double(ts(cspikep));
        lastSpikeTimesPast(x + offset, y + offset) = t;
%       Grab an 11x11 around the pixel
        xmin = x; xmax = x + ksize - 1;
        ymin = y; ymax = y + ksize - 1;
%       Add to dataset
        tmp = lastSpikeTimesPast( xmin : xmax, ...
                                    ymin : ymax );
        zeroz = tmp == 0; 
        tmp = exp( - abs(t - tmp ) ./ k );
        tmp(zeroz) = 0;
        %if sum(tmp(:)) ~= 1  % Isolated pixel
        dataset(cspikep - starti + 1, :, :) = tmp;
        %end
        
        
        
        %% Now do it backwards
        x = xs(cspikef);
        y = ys(cspikef);
        t = double(ts(cspikef));
        lastSpikeTimesFutr(x + offset, y + offset) = t;
%       Grab an 11x11 around the pixel
        xmin = x; xmax = x + ksize - 1;
        ymin = y; ymax = y + ksize - 1;
%       Add to labels
        tmp = lastSpikeTimesFutr( xmin : xmax, ...
                                    ymin : ymax );
        zeroz = tmp == 0;
        tmp = exp( - abs(t - tmp ) ./ k );
        tmp(zeroz) = 0;
        %if sum(tmp(:)) ~= 1  % Isolated pixel
        labels(cspikef - starti + 1, :, :) = tmp;
        %end

        cspikep = cspikep + 1;
        cspikef = cspikef - 1;

end  % end while loop

end