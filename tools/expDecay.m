function [ decayed ] = expDecay( time_difs, k )
%% EXPDECAY - Given the diffs from a recording and a k apply 
%               expontential decay

    decayed = exp( - time_difs ./ k );

end