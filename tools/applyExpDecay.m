function [ count ] = applyExpDecay( count, starti, endi, xs, ys, ts, varargin )
%APPLYEXPDECAY Summary of this function goes here
%   Detailed explanation goes here
    fprintf('%05f, %d, diff: %d\n', ts(starti)/30000, endi, endi-starti);
    msin = 3e4; % 3e4 is 30000us or 30ms
    k = 0.5; % A decent default
    for arg = 1:2:(nargin - 6)
        
       switch lower(varargin{arg})
           case 'msin'
               msin = varargin{arg + 1};
           case 'k'
               k = varargin{arg + 1};
           case 'outdir'
               outDir = varargin{arg + 1};
           case 'filename'
               filename = varargin{arg + 1};
           otherwise
               disp('unrecognised arg');
               return;
       end
           
    end
        

    DVS_RESOLUTION = 128;
    cur_spike = starti;
    lastSave = starti;
    kin = k;
    k = kin * 5e5;  % 1 second is 1e6 so just do half second (5e5)
    timesliceus = msin * 1e3;
    lastSpikeTimes = double(zeros(DVS_RESOLUTION, DVS_RESOLUTION));
    tmpcount = count;
    while cur_spike < endi
        x = xs(cur_spike)+1; %offset as DVS numbers from 0
        y = ys(cur_spike)+1;
        %pol = infomatrix3(cur_spike); %unused atm
        time = ts(cur_spike);
        
        lastSpikeTimes(x, y) = time;
        if time - ts(lastSave) > timesliceus
            timeDif = time - lastSpikeTimes;
            % THIS IS EXP DECAY
            res = 1./(1 + exp(-double(timeDif)./k));
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
    while cur_spike > starti
        x = xs(cur_spike)+1; %offset as DVS numbers from 0
        y = ys(cur_spike)+1;
        %pol = infomatrix3(cur_spike); %unused atm
        time = ts(cur_spike);
        
        lastSpikeTimes(x, y) = time;

        if lastSave > size(ts,1) || ts(lastSave) - ts(cur_spike) > timesliceus
            timeDif = lastSpikeTimes - time;
            timeDif(timeDif == -time) = time;
            % THIS IS EXP DECAY
            res = 1./(1 + exp(-double(timeDif)./k));
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

