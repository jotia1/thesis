function [ idx ] = getSampleIndexs( filename )
% GETSAMPLEINDEXS Given a file with standard metaflashes extract the
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

    [allAddr, ts] = loadaerdat(filename);
    [xs, ys, ps] = extractRetina128EventsFromAddr(allAddr);
    ts = double(ts);
    
%     sspike = 2000;
%     espike = 672474;
%     xs = xs(sspike:espike);
%     ys = ys(sspike:espike);
%     ts = ts(sspike:espike);
%     ps = ps(sspike:espike);
    
    % Constants
    ETHRES = 100; % Number of events to be considered a spike
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
        
        if buck > ETHRES; % Just hit a spike
            startTime = sum(buckets(1 : lastSpikeBuck + EDGEBUFFER));
            endTime = sum(buckets(1 : bucki - EDGEBUFFER));
            idx = [idx; startTime, endTime];
            %count = applyExpDecay(count, startTime, endTime, xs, ys, ...
            %    ts, filename, outDir, varargin{:});
            
            bucki = bucki + FRAMESPERSPIKE; 
            
            while buckets(bucki) > ETHRES;  % Skip remaining
                bucki = bucki + 1;
            end
            lastSpikeBuck = bucki;
            
        end
        
        bucki = bucki + 1;
    end
    % Plus the last segment
    startTime = sum(buckets(1 : lastSpikeBuck + EDGEBUFFER));
    endTime = sum(buckets(1 : bucki - EDGEBUFFER));
    idx = [idx; startTime, endTime];
end
