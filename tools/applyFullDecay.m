function [ dataset, labels ] = applyFullDecay( starti, endi, xs, ys, ts, ksize, msps, stride, decaytype, k );
%% APPLYDECAY - Given a start and end event, return a matrix of decayed
%           images of the FULL image
%
%           NOTE: this is decay over the FULL image as seperate from around
%           distiguished event.
%
%           TODO This is basically a copy paste of applyConvDecay to get
%           running quickly... should really be refactored.


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
    fprintf('Starti (%d) cannot be less than endi(%d)', starti, endi);
    return
end
nevents = endi  - starti;
nsamples = floor((nevents - mod(endi, stride) + (stride - mod(starti, stride))) / stride);

%   Init two 2D array of last spikes in each position
lastSpikeTimesPast = zeros(DVS_RESOLUTION, DVS_RESOLUTION);
lastSpikeTimesFutr = zeros(DVS_RESOLUTION, DVS_RESOLUTION);

%   Init a [numel(events), 11, 11] dataset and labels result variables
dataset = single(zeros(nsamples, ksize, ksize));
labels = single(zeros(nsamples, ksize, ksize));

% Ensure I can safely index spike times
%offset = floor(ksize / 2);
%lastSpikeTimesPast = padarray(lastSpikeTimesPast, [offset offset]);
%lastSpikeTimesFutr = padarray(lastSpikeTimesFutr, [offset offset]);

%
%   for i in range(numel(xs))
cspikep = starti;
cspikef = endi;
datai = 1;
labeli = 1;
p_entries = [];
f_entries = [];
fcount = 1;
while cspikep <= endi || cspikef >= starti;
%       add pixel spike time to 2D array
        x = xs(cspikep);
        y = ys(cspikep);
        t = double(ts(cspikep));
        lastSpikeTimesPast(x, y) = t;
        
        if mod(cspikep, stride) == 0 && cspikep <= endi;  % Only decay is stride is 
            p_entries(datai) = cspikep;
    %       Grab an 11x11 around the pixel
            xmin = x; xmax = x + ksize - 1;
            ymin = y; ymax = y + ksize - 1;
    %       Add to dataset
            %tmp = lastSpikeTimesPast( offset : offset + ksize - 1, ...
            %                            offset : offset + ksize - 1 );
            tmp = lastSpikeTimesPast;
            zeroz = tmp == 0; 

            %tmp = exp( - abs(t - tmp ) ./ k );
            if strcmp(decaytype, 'exp') == 1;
                tmp = expDecay(abs(t - tmp ), k);
            elseif strcmp(decaytype, 'linear') == 1;
                tmp = linearDecay(abs(t - tmp ), k);
            else
               disp('Decaytype not undertood, use linear or exp');
               return;
            end

            tmp(zeroz) = 0;
            %if sum(tmp(:)) ~= 1  % Isolated pixel
            dataset(datai, :, :) = tmp;
            datai = datai + 1;
            %end
        end
        
        
        
        %% Now do it backwards
        x = xs(cspikef);
        y = ys(cspikef);
        t = double(ts(cspikef));
        lastSpikeTimesFutr(x, y) = t;
        
        if mod(cspikef, stride) == 0 && cspikef >= starti;  % Only decay is stride is correct
            f_entries(fcount) = cspikef;
            fcount = fcount + 1;
    %       Grab an 11x11 around the pixel
            xmin = x; xmax = x + ksize - 1;
            ymin = y; ymax = y + ksize - 1;
    %       Add to labels
            %tmp = lastSpikeTimesFutr( xmin : xmax, ...
            %                            ymin : ymax );
            tmp = lastSpikeTimesFutr;
            zeroz = tmp == 0;

            %tmp = exp( - abs(t - tmp ) ./ k );
            if strcmp(decaytype, 'exp') == 1;
                tmp = expDecay(abs(t - tmp ), k);
            elseif strcmp(decaytype, 'linear') == 1;
                tmp = linearDecay(abs(t - tmp ), k);
            else
               disp('Decaytype not undertood, use linear or exp');
               return;
            end

            tmp(zeroz) = 0;
            %if sum(tmp(:)) ~= 1  % Isolated pixel
            labels(labeli, :, :) = tmp;
            labeli = labeli + 1;
            %end
        end


        cspikep = cspikep + 1;
        cspikef = cspikef - 1;
        
end  % end while loop
labels = reshape(flipud(labels), [], ksize * ksize);  %Format data as [samples, kx*ky] vector
dataset = reshape(dataset, [], ksize * ksize); 
end