function [ idx ] = getSampleIndexs( ts )
% GETSAMPLEINDEXS Given some timestamps with standard metaflashes extract the
% indexes of each segments start and stop

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

%% Implementation

    %[allAddr, ts] = loadaerdat(filename);
    %[xs, ys, ps] = extractRetina128EventsFromAddr(allAddr);
    %ts = fixWrapping(ts);
    
    % Constants
    ETHRES = 120; % Number of events to be considered a spike
    FRAMESPERSPIKE = 6; % Number of frames to skip per spike
    EDGEBUFFER = 3; % How many frames to drop from sample edges
    
    % Variables
    lastSpikeBuck = -1;  % Last bucket that had a spike
    count = 0;  % Used by applyExpDecay 
    
    buckets = hist(ts, (ts(end) - ts(1)) / 3e4);
    idx = [];
   
    bucki = 1;
    while bucki < numel(buckets);
        buck = buckets(bucki);
        
        if buck > ETHRES && bucki - lastSpikeBuck >= 6; % Just hit a spike
            startTime = sum(buckets(1 : lastSpikeBuck + EDGEBUFFER));
            endTime = sum(buckets(1 : bucki - EDGEBUFFER));
            idx = [idx; startTime, endTime];
            %count = applyExpDecay(count, startTime, endTime, xs, ys, ...
            %    ts, filename, outDir, varargin{:});
            
            bucki = bucki + FRAMESPERSPIKE; 
            
            while bucki < numel(buckets) && buckets(bucki) > ETHRES;  % Skip remaining
                bucki = bucki + 1;
            end
            lastSpikeBuck = bucki;
            
        end
        
        bucki = bucki + 1;
    end
    % Plus the last segment
    if lastSpikeBuck < numel(buckets);   % Only if haven't already passed end
        startTime = sum(buckets(1 : lastSpikeBuck + EDGEBUFFER));
        endTime = sum(buckets(1 : bucki - EDGEBUFFER));
        idx = [idx; startTime, endTime];
    end
end
