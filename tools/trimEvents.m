function [ xs, ys, ts, ps, seps ] = trimEvents(infile) 
%% Trim events - Assemble data and trim out metaflashes

%infile = 'data/8ang1.aedat';
idxs = getSampleIndexs(infile);

[allAddr, tts] = loadaerdat(infile);
[txs, tys, tps] = extractRetina128EventsFromAddr(allAddr);
tts = fixWrapping(tts);

sspike = idxs(2, 1) - 1;
espike = idxs(end - 1, 2);
txs = txs(sspike:espike) + 1;
tys = tys(sspike:espike) + 1;
tts = tts(sspike:espike);
tps = tps(sspike:espike);

xs = [];
ys = [];
ts = [];
ps = [];
cum_timelost = 0;
lhidx = -1; % last half index
lastStart = 1;
seps = zeros(size(idxs));
for i = 2:size(idxs, 1) - 1;  % Skip first and last section 
   sidx = idxs(i, 1) - sspike;
   eidx = idxs(i, 2) - sspike;
   hidx = int32((eidx - sidx)) + sidx; % used to be used to reduce data
   %[i, sidx, hidx, eidx, numel(txs)];
   xs = [ xs; txs( sidx : hidx ) ];
   ys = [ ys; tys( sidx : hidx ) ];
   if i > 2
       cum_timelost = cum_timelost + (tts(sidx) - tts(lhidx));
   end
   ts = [ ts; tts( sidx : hidx ) - cum_timelost ];
   ps = [ ps; tps( sidx : hidx ) ];
   %fprintf('sidx: %d, eidx: %d, hidx: %d, lhidx: %d, cum_time: %d\n', ...
   %sidx, eidx, hidx, lhidx, cum_timelost);
   lhidx = hidx;
   seps(i, :) = [lastStart, numel(xs)];
   lastStart = numel(xs);
end

seps = seps(2 : end -1, :);

end
%gpuEvolveKerns;
