function [  ] = plotSpikeGraph( filename, timeslice )
%PLOTSPIKEGRAPH Plot the given aedat file in buckets of timeslice ms
%   Example usage: plotSpikeGraph('samp1.aedat', 30)

    clc;
    [allAddr, allTs] = loadaerdat(filename);
    [infomatrix1, infomatrix2, infomatrix3] = extractRetina128EventsFromAddr(allAddr);
    allTs = int32(allTs);
    
    cur_spike = 1;
    num_spikes = size(allTs, 1);
    timeslice_us = timeslice * 1000;
    start =  allTs(1);
    last = allTs(end);
    num_buckets = floor((last - start) / timeslice_us) + 1;
    cur_bucket = 1;
    buckets = zeros(1, num_buckets);
    
    while cur_spike < num_spikes
        % If spike belongs in next bucket
        while allTs(cur_spike) > start + (cur_bucket * timeslice_us)
           cur_bucket = cur_bucket + 1;
        end
        buckets(cur_bucket) = buckets(cur_bucket) + 1;
        cur_spike = cur_spike + 1;
    end
    
    plot(buckets)
    xlabel('Time')
    ylabel('# spikes')
    title(sprintf('Spike graph for: %s, %d ms buckets', filename, timeslice));
    %title(strcat(strcat('Spike graph from:  ', filename), strcat(num2str(timeslice), 'ms buckets.')));
    
end

