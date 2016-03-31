aedatFile = 'data/dvs_gt5.aedat';
outputFolder = 'gt5';
msin = 30;
k = 0.5;


mkdir(outputFolder);
preprocess(aedatFile, outputFolder, 'exponential', 'msin', msin, 'k', k);
