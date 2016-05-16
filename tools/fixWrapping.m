function [ ts ] = fixWrapping( ts )
%% FIXWRAPPING - Given a list of timestamps that potentially wrap fix fix 
%               so they are monotonically increasing.

    % Indexs where timestamps decrease (last number before decrease)
    idxs = find( ts(2 : end) < ts(1 : end -1) );
    cum_time = 0;
    
    for i = 2 : numel(idxs)
        sec_start = idxs(i -1) + 1;
        sec_end = idxs( i ); % idxs(i) is first of next section, so -1
        ts(sec_start : sec_end) = ts(sec_start : sec_end) + cum_time;
        time_between_sections = ts( sec_end ) + (2^32 - ts(sec_end - 1));
        cum_time = cum_time + time_between_sections;
    end
    % Clean up last section if there were problems
    if numel(idxs) > 0
        ts(idxs(end) + 1 : end) = ts(idxs(end) + 1 : end) + cum_time;
    end
    % If there are wraps and its near start
    if numel(idxs) > 0 && idxs(1) < 2000
        ts = ts(idxs(1) + 1 : end);  % Remove initial junk
    end
end