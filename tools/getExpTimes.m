function [ poss ] = getExpTimes( ts, xs, ys, tpb, maxSpikes )
%GETEXPTIMES Return start and end times of experiemnts in a data file.
%   Return a list of integers detailing the start and end positions of 
%   each experiment in a given data file based on when large spikes 
%   occur.
%   
%   Input: 	ts - times of each spike
%    		xs - x position of all spikes
%   		ys - y position of all spikes
%           tpb - Time (in ms) to store spikes in each bucket
%   		maxSpikes - max spikes per bucket in an experiment
%   			
%   Output:	poss - list of integers corresponding to index's for 
%   				spikes that are the start and end of an experiment

    poss = [];
    cbin = 1;
    bps = 15; % bins per sample (number of bins to have between spikes)
    uspb = tpb * 1e3; %us per bucket - convert ms buckets to us
    if nnz(ts==ts(1)) > 1
       % We have wrap around
       % TODO
       nbins = -1;
       dsip('wrap around exists, problem needs a fix...');
       return;
    else
        nbins = ceil((ts(end) - ts(1)) / uspb); 
        counts = hist(ts, nbins);  % main line, does the bucketing
    end
    metaflashes = counts > maxSpikes;  % logical matrix

    % loop variables
    npzeros = 0;  % number of previous zeros
    lastmf = 1; % last meta flash seen
    slastmf = 1; %start of the last meta flash (used to sum over the flash)
    indata = true;

    % put spikes into buckets based on bucket size
    while cbin < size(counts, 2)
        if (metaflashes(cbin) == 1) && indata % was in data but now am not.
            % TODO below line does not work, need to figure out what to add
            % to make poss a list of indexs rather than bins. 
            %poss(end + 1) = sum(counts(1:poss(lastmf + 2))); %lastmf + 2;
            %poss(end + 1) = sum(counts(1:poss(cbin - 2))); %cbin - 2;
            % poss(end) is the number of spikes up to the last flash start
            % need to add on the number of spikes in the flash...
            samps = lastmf + 7;
            sampe = cbin - 7;
            if size(poss, 2) == 0  % if poss is empty, matlab index issue
                poss(1) = sum(counts(slastmf:samps-1));
            else
                poss(end + 1) = poss(end) + sum(counts(slastmf:samps-1));
            end
            poss(end + 1) = poss(end) + sum(counts(samps:sampe-1));
            slastmf = sampe;  % set for next time around
        end
        % TODO there are still problems here, this code should be considered unstable
        if metaflashes(cbin) == 1
           npzeros = 0;
           lastmf = cbin;
           indata = false;
        else 
            npzeros = npzeros + 1;
        end

        if npzeros > bps && ~indata  % seen at least `bps` good buckets
            indata = true;
        end


        cbin = cbin + 1;
    end


end

