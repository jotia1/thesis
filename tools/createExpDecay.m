function [ ] = createExpDecay( filename, outDir, msin, kin )
%CREATEEXPDECAY Convert a given aedat file to a data set with exponential decay
%   Open the given aedat file and apply decay with the given parameters
%   saving an image of the decay to the out
%   k is how long to apply decay over, value of 1 will give 0.5 seconds
%   Example: createExpDecay('ball.aedat', 'play', 5000, 0.5)


    disp('start loading aedat file');
    [allAddr, allTs] = loadaerdat(filename);
    %extracts all the information from the address matrices in the form
    %[xcoordinate, ycoordinate, polarity]
    [infomatrix1, infomatrix2, infomatrix3] = extractRetina128EventsFromAddr(allAddr);
    %times are in us
    allTs = int32(allTs);
    k = kin * 5e5;  % 1 second is 1e6 so just do half second (5e5)
    timesliceus = msin * 1e3;
    
    DVS_RESOLUTION = 128;
    total_spikes = size(allTs);
    total_spikes = total_spikes(1);
    starttime = allTs(1);
    
    lastSpikeTimes = int32(zeros(DVS_RESOLUTION, DVS_RESOLUTION));
    cur_spike = 1;
    lastSave = 1;
    count = 1;

    % Write meta data to text file
    fileID = fopen(sprintf('%s/readme.txt', outDir), 'w');
    fprintf(fileID, 'Exponential Decay: filename outdir eventsPerImage k\n');
    fprintf(fileID, '%s %s %d %d\n', filename, outDir, msin, kin);
    fclose(fileID);
    
    % process data
%     disp('start pre-process');
%     spike = 1;
%     freq = 0;
%     last_check = 1;
%     nallTs = [];
%     while spike < size(allTs, 1)
%         if allTs(spike) - allTs(last_check)
%             
%         end
%         spike = spike + 1;
%     end
    
    
    
    
    %DEBUG: while 1 == 0
    disp('start forward pass (into past)');
    freq = 0;
    while cur_spike < total_spikes
        x = infomatrix1(cur_spike)+1; %offset as DVS numbers from 0
        y = infomatrix2(cur_spike)+1;
        %pol = infomatrix3(cur_spike); %unused atm
        time = allTs(cur_spike);
        cur_spike = cur_spike + 1;
        
        lastSpikeTimes(x, y) = time;
        
        
        %disp(cur_spike - lastSave);
        if allTs(cur_spike) - allTs(lastSave) > timesliceus
            
            timeDif = time - lastSpikeTimes;
            
       
            % THIS IS EXP DECAY
            res = 1./(1 + exp(-double(timeDif)./k));
            
            
            
            outfile = sprintf('%s/%s_%dms_%05d_past.png', ...
                    outDir, filename, msin, count);
            
            imwrite(flipud(res.'), outfile); % 'InitialMagnification', 300
            count = count + 1;
            lastSave = cur_spike;
        end
    end
    
    
    %DEBUG: cur_spike = size(allTs, 1);
    %DEBUG: count = 1000;
    disp('Start backwards pass (into future)');
    % cur_spike, count, are at the end
    lastSpikeTimes = int32(zeros(DVS_RESOLUTION, DVS_RESOLUTION));
    lastSave = timesliceus + lastSave;%cur_spike; % in case some events were left out from fwds
    %cur_spike = lastSave
    % Now we go backwards
    while cur_spike > 1
        %disp('in');
        x = infomatrix1(cur_spike)+1; %offset as DVS numbers from 0
        y = infomatrix2(cur_spike)+1;
        %pol = infomatrix3(cur_spike); %unused atm
        time = allTs(cur_spike);
        cur_spike = cur_spike - 1;

        lastSpikeTimes(x, y) = time;

        %disp(cur_spike - lastSave);
        if lastSave > size(allTs,1) || allTs(lastSave) - allTs(cur_spike) > timesliceus
            timeDif = lastSpikeTimes - time;
            timeDif(timeDif == -time) = time;
            res = 1./(1 + exp(-double(timeDif)./k));
            
            outfile = sprintf('%s/%s_%dms_%05d_futr.png', ...
                    outDir, filename, msin, count);
            imwrite(flipud(res.'), outfile); %'InitialMagnification', 300
            count = count - 1;
            lastSave = cur_spike;
        end
    end
    
    
 
end

