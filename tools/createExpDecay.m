function [ ] = createExpDecay( filename, outDir, evtsPimg, k )
%CREATEEXPDECAY Convert a given aedat file to a data set with exponential decay
%   Open the given aedat file and apply decay with the given parameters
%   saving an image of the decay to the out
%   Example: createExpDecay('ball.aedat', 'play', 5000, 0.5)


    [allAddr, allTs] = loadaerdat(filename);
    %extracts all the information from the address matrices in the form
    %[xcoordinate, ycoordinate, polarity]
    [infomatrix1, infomatrix2, infomatrix3] = extractRetina128EventsFromAddr(allAddr);
    %times are in us
    allTs = int32(allTs);
    k = k * 5e5;  % 1 second is 1e6
    
    DVS_RESOLUTION = 128;
    total_spikes = size(allTs);
    total_spikes = total_spikes(1);
    starttime = allTs(1);
    
    lastSpikeTimes = int32(zeros(DVS_RESOLUTION, DVS_RESOLUTION));
    cur_spike = 1;
    lastSave = 1;
    count = 1;
    figure
    while cur_spike < total_spikes
        x = infomatrix1(cur_spike)+1; %offset as DVS numbers from 0
        y = infomatrix2(cur_spike)+1;
        %pol = infomatrix3(cur_spike); %unused atm
        time = allTs(cur_spike);
        
        lastSpikeTimes(x, y) = time;
        cur_spike = cur_spike + 1;
        disp(cur_spike - lastSave);
        if cur_spike - lastSave > evtsPimg
            timeDif = time - lastSpikeTimes;
            res = 1./(1 + exp(-double(timeDif)./k));
            
            outfile = sprintf('%s/%s_%devts_%05d.png', ...
                    outDir, evtsPimg, filename, count);
            
            imshow(flipud(res.'), 'InitialMagnification', 300)%, outfile);
            count = count + 1;
            lastSave = cur_spike;
        end
    end
    
    
 
end

