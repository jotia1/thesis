function [ ] = nExpDecay( filename, outDir, varargin )
%NEXPDECAY Summary of this function goes here
%   Detailed explanation goes here

    [allAddr, ts] = loadaerdat(filename);
    ts = double(ts);
    [xs, ys, ps] = extractRetina128EventsFromAddr(allAddr);
    
    %% Remove a polarity
    if false
        ps = ps ==1;
        xs = xs(ps);
        ys = ys(ps);
        ts = ts(ps);
    end
    

    %% processing
    
    [poss, counts] = getExpTimes(ts, xs, ys, 30, 75);
    
    sb = -1;  % sample Start Bucket, (init at -1 ready to be init in loop)
    eb = sb + 1;
    count = 1;

    while eb < size(poss, 2)
        sb = sb + 2;
        eb = sb + 1;
        si = sum(counts(1:poss(sb)));  % index of event starting sample
        ei = sum(counts(1:poss(eb)));  % index of event ending the sample
        
        % now for xs, ys, ts apply decay between events si to ei
        count = applyExpDecay(count, si, ei, xs, ys, ts, filename, ...
                                    outDir, varargin{:});
        
    end
    
end

