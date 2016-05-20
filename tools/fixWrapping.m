function [ ts ] = fixWrapping( ts )
%% FIXWRAPPING - Given a list of timestamps that potentially wrap fix fix 
%               so they are monotonically increasing.

    % Indexs where timestamps decrease (last number before decrease)
    idxs = find( ts(2 : end) < ts(1 : end -1) );
    
    for i = 1 : numel(idxs)
        sec_start = idxs(i) + 1;
        if numel(idxs) == 1 || i == numel(idxs)
            sec_end = numel(ts);
        else
            sec_end = idxs( i + 1 ); 
        end
        ts(sec_start : sec_end) = ts(sec_start : sec_end) + (2^32)*i;
    end

end