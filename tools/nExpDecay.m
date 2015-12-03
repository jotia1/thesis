function [ ] = nExpDecay( filename, outDir, varargin )
%NEXPDECAY Summary of this function goes here
%   Detailed explanation goes here

    selectivepol = false;
    if nargin == 4
        if lower(varargin{1}) == 'pol'
            switch lower(varargin{2})
                case 'pos'
                    selectivepol = true;
                    pol = 1;
                case 'neg'
                    selectivepol = true;
                    pol = -1;
                otherwise
                    disp('pol vararg must be pos or neg.');
                    return;
            end
        else
            disp('unrecognised vararg (only pol accepted)')
            return;
        end
    end

    [allAddr, ts] = loadaerdat(filename);
    ts = double(ts);
    [xs, ys, ps] = extractRetina128EventsFromAddr(allAddr);
    
    %% Remove a polarity
    if selectivepol
        ps = ps ==pol;
        xs = xs(ps);
        ys = ys(ps);
        ts = ts(ps);
    end
    

    %% processing
    
    poss = getExpTimes(ts, xs, ys, 30, 75);
    
    %sb = -1;  % sample Start Bucket, (init at -1 ready to be init in loop)
    %eb = sb + 1;
    count = 1;
    bucketindex = 1;

    while bucketindex < size(poss, 2)
        si = poss(bucketindex);  % index of event starting sample
        ei = poss(bucketindex + 1);  % index of event ending the sample
        
        % now for xs, ys, ts apply decay between events si to ei
        count = applyExpDecay(count, si, ei, xs, ys, ts, filename, ...
                                    outDir, varargin{:});
        bucketindex = bucketindex + 2;
    end
    
end

