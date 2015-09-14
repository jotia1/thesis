function [allTs, infomatrix1, infomatrix2, infomatrix3] = openaedat(filename)

%loads all the addresses and times into matrices
[allAddr, allTs] = loadaerdat(filename);
%times are in us
allTs = int32(allTs);

%extracts all the information from the address matrices in the form
%[xcoordinate, ycoordinate, polarity]
[infomatrix1, infomatrix2, infomatrix3] = extractRetina128EventsFromAddr(allAddr);

end
    