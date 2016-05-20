function [ xs, ys, ts, ps, seps ] = trimEvents(infile) 
%% Trim events - Assemble data and trim out metaflashes



[allAddr, tts] = loadaerdat(infile);
[txs, tys, tps] = extractRetina128EventsFromAddr(allAddr);
tts = fixWrapping(tts);

%infile = 'data/8ang1.aedat';
idxs = getSampleIndexs(tts);

sspike = idxs(2, 1) - 1;
espike = idxs(end - 1, 2);
txs = txs(sspike:espike) + 1;  % Setup for matlab index (from 1)
tys = tys(sspike:espike) + 1;
tts = tts(sspike:espike);
tps = tps(sspike:espike);

idxs = getSampleIndexs(tts);  % Get new seps now its trimmed.

xs = [];
ys = [];
ts = [];
ps = [];
cum_timelost = 0;
lhidx = -1; % last half index
lastStart = 0;
seps = zeros(size(idxs));
for i = 1:size(idxs, 1);  % Skip first and last section 
   sidx = idxs(i, 1);
   eidx = idxs(i, 2);
   %hidx = int32((eidx - sidx)) + sidx; % used to be used to reduce data
   %[i, sidx, hidx, eidx, numel(txs)];
   xs = [ xs; txs( sidx : eidx ) ];
   ys = [ ys; tys( sidx : eidx ) ];
   %if i > 2
   %    cum_timelost = cum_timelost + (tts(sidx) - tts(lhidx));
   %end
   ts = [ ts; tts( sidx : eidx ) ];
   ps = [ ps; tps( sidx : eidx ) ];
   %fprintf('sidx: %d, eidx: %d, hidx: %d, lhidx: %d, cum_time: %d\n', ...
   %sidx, eidx, hidx, lhidx, cum_timelost);
   %lhidx = hidx;
   seps(i, :) = [lastStart + 1, numel(xs)];
   lastStart = numel(xs);
end

%seps = seps(2 : end -1, :);

end