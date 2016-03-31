function [ ] = preprocess(filename, outDir, decay_type, varargin);
% PREPROCESS Given an Aedat file preprocess it into image files
%  Given an AEDAT file convert it to a folder of images with decay into the
%  past and the future.

%% Algorithm
%   bucket all in a hist
%   for each buck in buckets:
%       if buck > ethres:
%           % Spike. lastSpike to buck -1 is sample
%           applyExpDecay...
%           % while buck > ethres:
%               Move to next buck % exhusting meta data
%           lastSpike = buck
%       else:
%           continue...

%% Hacks and TODOs
    if exist('decay_type', 'var') && strcmp(decay_type, 'exponential') == 0;
        error('decay_type: %s not supported, only exponential ' + ...
            'currently implemented', decay_type);
    end

%% Implementation

    [allAddr, ts] = loadaerdat(filename);
    [xs, ys, ps] = extractRetina128EventsFromAddr(allAddr);
    ts = double(ts);
    
    % Constants
    ETHRES = 100; % Number of events to be considered a spike
    FRAMESPERSPIKE = 6; % Number of frames to skip per spike
    EDGEBUFFER = 3; % How many frames to drop from sample edges
    
    % Variables
    lastSpikeBuck = -1;  % Last bucket that had a spike
    count = 0;  % Used by applyExpDecay 
    
    buckets = hist(ts, (ts(end) - ts(1)) / 3e4);
   
    bucki = 1;
    while bucki < numel(buckets);
        buck = buckets(bucki);
        
        if buck > ETHRES; % Just hit a spike
            startTime = sum(buckets(1 : lastSpikeBuck + EDGEBUFFER));
            endTime = sum(buckets(1 : bucki - EDGEBUFFER));
            count = applyExpDecay(count, startTime, endTime, xs, ys, ...
                ts, filename, outDir, varargin{:});
            
            bucki = bucki + FRAMESPERSPIKE; 
            
            while buckets(bucki) > ETHRES;  % Skip remaining
                bucki = bucki + 1;
            end
            lastSpikeBuck = bucki;
            
        end
        
        bucki = bucki + 1;
    end
   
end
