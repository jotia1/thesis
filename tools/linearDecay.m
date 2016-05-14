function [ decayed ] = linearDecay( time_difs, k )
%% LINEARDECAY - Given an image decay it linearly using the constant k
%   
    decayed = (-1/k) .* time_difs + 1;
    decayed(decayed < 0) = 0;
end