function [ ] = createExpDecay( filename, outDir, evtsPimg, k )
%CREATEEXPDECAY Convert a given aedat file to a data set with exponential decay
%   Open the given aedat file and apply decay with the given parameters
%   saving an image of the decay to the out
%   k is how long to apply decay over, value of 1 will give 0.5 seconds
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
    while 1 == 0
    %while cur_spike < total_spikes
        x = infomatrix1(cur_spike)+1; %offset as DVS numbers from 0
        y = infomatrix2(cur_spike)+1;
        %pol = infomatrix3(cur_spike); %unused atm
        time = allTs(cur_spike);
        
        lastSpikeTimes(x, y) = time;
        cur_spike = cur_spike + 1;
        %disp(cur_spike - lastSave);
        if cur_spike - lastSave > evtsPimg
            timeDif = time - lastSpikeTimes;
            res = 1./(1 + exp(-double(timeDif)./k));
            
            outfile = sprintf('%s/%s_%devts_%05d_fwds.png', ...
                    outDir, filename, evtsPimg, count);
            
            imshow(flipud(res.'), 'InitialMagnification', 300)%, outfile);
            count = count + 1;
            lastSave = cur_spike;
        end
    end
    cur_spike = size(allTs, 1);
    count = 1000;
    disp('BACKWARDS');
    % cur_spike, count, are at the end
    lastSpikeTimes = int32(zeros(DVS_RESOLUTION, DVS_RESOLUTION));
    lastSave = cur_spike; % in case some events were left out from fwds
    % Now we go backwards
    while cur_spike > 1
        %disp('in');
        x = infomatrix1(cur_spike)+1; %offset as DVS numbers from 0
        y = infomatrix2(cur_spike)+1;
        %pol = infomatrix3(cur_spike); %unused atm
        time = allTs(cur_spike);
        
        lastSpikeTimes(x, y) = time;
        cur_spike = cur_spike - 1;
        %disp(cur_spike - lastSave);
        if lastSave - cur_spike > evtsPimg
            %disp('inner if');
            timeDif = lastSpikeTimes - time;
            timeDif(timeDif == -time) = time;
            res = 1./(1 + exp(-double(timeDif)./k));
            
            outfile = sprintf('%s/%s_%devts_%05d_bkwd.png', ...
                    outDir, filename, evtsPimg, count);
            
            imshow(flipud(res.'), 'InitialMagnification', 300)%, outfile);
            count = count - 1;
            lastSave = cur_spike;
        end
    end
    
    
 
end

