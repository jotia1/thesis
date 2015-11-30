function [ poss ] = getExpTimes( ts, xs, ys, tpd, maxSpikes )
%GETEXPTIMES Return start and end times of experiemnts in a data file.
%   Return a list of integers detailing the start and end positions of 
%   each experiment in a given data file based on when large spikes 
%   occur.
%   
%   Input: 	ts - times of each spike
%    		xs - x position of all spikes
%   		ys - y position of all spikes
%           tpd - Time (in ms) to store spikes in each bucket
%   		maxSpikes - max spikes per bucket in an experiment
%   			
%   Output:	poss - list of integers corresponding to index's for 
%   				spikes that are the start and end of an experiment

poss = [];
cspike = 0;


% put spikes into buckets based on bucket size
while cspike < size(ts, 1)
    x = xs(cspike);
    y = ys(cspike);
    t = ys(cspike);
    
    
    cspike = cspike + 1;
end


end

