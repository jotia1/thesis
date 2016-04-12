%% Run batch - Assemble data and evolve kernels

infile = 'data/8ang1.aedat';
idxs = getSampleIndexs(infile);

[allAddr, tts] = loadaerdat(infile);
[txs, tys, tps] = extractRetina128EventsFromAddr(allAddr);

sspike = 8200;
espike = 68200;
txs = txs(sspike:espike);
tys = tys(sspike:espike);
tts = tts(sspike:espike);
tps = tps(sspike:espike);

xs = [];
ys = [];
ts = [];
ps = [];

for i = 1:size(idxs, 1);
    disp(idxs(i, :))
   xs = [ xs; txs( idxs(i, 1) : idxs(i, 2) ) ];
   ys = [ ys; tys( idxs(i, 1) : idxs(i, 2) ) ];
   ts = [ ts; tts( idxs(i, 1) : idxs(i, 2) ) ];
   ps = [ ps; tps( idxs(i, 1) : idxs(i, 2) ) ];
end



gpuEvolveKerns;