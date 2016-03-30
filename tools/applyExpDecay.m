function [ count ] = applyExpDecay( count, starti, endi, xs, ys, ts, filename, outDir, varargin )
%APPLYEXPDECAY Summary of this function goes here
%   Detailed explanation goes here
    fprintf('%d, %d, diff: %d\n', starti, endi, endi-starti);
    msin = 30; %30ms
    k = 0.5; % A decent default
    for arg = 1:2:(nargin - 8)
        
       switch lower(varargin{arg})
           case 'msin'
               msin = varargin{arg + 1};
           case 'k'
               k = varargin{arg + 1};
           case 'pol'
               continue;
           otherwise
               disp('unrecognised arg');
               return;
       end
           
    end
    
    % set up variables
    DVS_RESOLUTION = 128;
    cur_spike = starti;
    lastSave = starti;
    kin = k;
    k = kin * 5e5;  % 1 second is 1e6 so just do half second (5e5)
    timesliceus = msin * 1e3;
    lastSpikeTimes = double(zeros(DVS_RESOLUTION, DVS_RESOLUTION));
    tmpcount = count;
        
    %% Write meta data to text file
    fileID = fopen(sprintf('%s/readme.txt', outDir), 'w');
    fprintf(fileID, 'Created at: %s\n', datestr(datetime('now')));
    fprintf(fileID, 'Exponential Decay: filename outdir msPerImage k\n');
    fprintf(fileID, '%s %s %d %d\n', filename, outDir, msin, kin);
    fclose(fileID);
    
    sigspikes = []; % index's of spikes at which images were made
    
    while cur_spike < endi
        x = xs(cur_spike)+1; %offset as DVS numbers from 0
        y = ys(cur_spike)+1;

        time = ts(cur_spike);
        
        lastSpikeTimes(x, y) = time;
        if time - ts(lastSave) > timesliceus
            sigspikes(end + 1) = cur_spike;
            timeDif = time - lastSpikeTimes;
            %zeroz = lastSpikeTimes == 0);
            % THIS IS EXP DECAY
            res = 1./(1 + exp(-double(timeDif)./k));
            res(lastSpikeTimes == 0) = 1;
            outfile = sprintf('%s/%s_%dms_k%0.2f_%05d_past.pgm', ...
                    outDir, filename, msin, kin, tmpcount);
            imwrite(flipud(res.'), outfile);%'InitialMagnification', 300); %
            tmpcount = tmpcount + 1;
            %waitforbuttonpress;
            %return;
            lastSave = cur_spike;
        end
        cur_spike = cur_spike + 1;
    end
    
    count = tmpcount;
    
    lastSpikeTimes = double(zeros(DVS_RESOLUTION, DVS_RESOLUTION));
    lastSave = timesliceus + lastSave;
    spikei = size(sigspikes, 2);  % index of next spike to create image at
    while cur_spike > starti
        x = xs(cur_spike)+1; %offset as DVS numbers from 0
        y = ys(cur_spike)+1;
        %pol = infomatrix3(cur_spike); %unused atm
        time = ts(cur_spike);
        
        lastSpikeTimes(x, y) = time;

        %if lastSave > size(ts,1) || ts(lastSave) - ts(cur_spike) > timesliceus
        if spikei > 0 && cur_spike == sigspikes(spikei)
            spikei = spikei - 1;
            timeDif = lastSpikeTimes - time;
            timeDif(timeDif == -time) = time;
            %zeroz = find(lastSpikeTimes == 0);
            % THIS IS EXP DECAY
            res = 1./(1 + exp(-double(timeDif)./k));
            res(lastSpikeTimes == 0) = 1;
            outfile = sprintf('%s/%s_%dms_k%0.2f_%05d_futr.pgm', ...
                    outDir, filename, msin, kin, tmpcount);
            imwrite(flipud(res.'), outfile);%'InitialMagnification', 300); %
            tmpcount = tmpcount - 1;
            %waitforbuttonpress;
            %return;
            lastSave = cur_spike;
        end
        cur_spike = cur_spike - 1;
    end


end

