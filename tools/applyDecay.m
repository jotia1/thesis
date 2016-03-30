function [ output_args ] = applyDecay( filename, outDir, decay_type, varargin )
%APPLYDECAY Given an aedat file and parameters of decay, generate images
%   Given input parameters for decay apply that decay to the given input
%   file and save series of those images to the specified outdir. 
%   Example: applyDecay('samp3.aedat', 'play', 'exponential')

    [allAddr, ts] = loadaerdat(filename);
    [xs, ys, ps] = extractRetina128EventsFromAddr(allAddr);
    ts = double(ts);
    total_spikes = size(ts, 1);
    count = 1;
    
    ethres = 100;   % TODO This needs to be experimentally determined
    tthres = 3e4;  % 3e4 is 30ms
    startp = 1;
    endp = startp + ethres;
    starti = 1;
    indata = true;
    last_trigger = 1;
    
    % capture one frame then call a function to process it
    while endp < total_spikes   % for each spike
        
        % TODO add a catch in here to circular time stuff (fix wrapping
        % problem)
        
        time_distance = ts(endp) - ts(startp);
        % if not in data is triggered then delay 60ms
        if ~indata && ts(endp) - ts(last_trigger) < tthres *2
            startp = startp + 1;
            endp = endp + 1;
            continue;
        end
            
        if time_distance > tthres && ~indata % in data and was in meta
            indata = true;
            starti = endp;
            endi = endp;
        elseif time_distance > tthres && indata % in data and still in data
            indata = true;
            endi = endp;
        elseif time_distance < tthres && indata % in meta but was in data before
            % starti and endi are now set. 
            count = applyExpDecay(count, starti, startp - ethres * 2, xs, ...
                            ys, ts, filename, outDir, varargin{:});
            startp = endp;
            endp = startp + ethres;
            starti = startp;
            endi = endp;
            indata = false;
            last_trigger = startp - ethres;
        elseif time_distance < tthres && ~indata % Else in meta and was in meta before
            %fprintf('METAMETA: Start: %d, end: %d, diff: %d\n', starti, endi, endi-starti);
            startp = endp;
            endp = startp + ethres;
            starti = startp;
            endi = endp;
            indata = false;
            last_trigger = startp;
        end
        
        startp = startp + ethres;
        endp = endp + ethres;
    end
    % Decay last section
    %applyExpDecay(count, starti, startp, xs, ...
    %                        ys, ts, 'filename', filename, 'outdir', outDir, varargin{:});
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    %% OLD STUFF
    
    
%     
%     % process data
% %     disp('start pre-process');
% %     spike = 1;
% %     freq = 0;
% %     last_check = 1;
% %     nallTs = [];
% %     while spike < size(allTs, 1)
% %         if allTs(spike) - allTs(last_check)
% %             
% %         end
% %         spike = spike + 1;
% %     end
%     
%     
%     
%     
%     %DEBUG: while 1 == 0
%     disp('start forward pass (into past)');
%     freq = 0;
%     lastlastSpikeTimes = int32(zeros(DVS_RESOLUTION, DVS_RESOLUTION));
%     while cur_spike < total_spikes
%         x = infomatrix1(cur_spike)+1; %offset as DVS numbers from 0
%         y = infomatrix2(cur_spike)+1;
%         %pol = infomatrix3(cur_spike); %unused atm
%         time = allTs(cur_spike);
%         cur_spike = cur_spike + 1;
%         
%         lastSpikeTimes(x, y) = time;
%         freq = freq + 1;
%         
%         
%         %disp(cur_spike - lastSave);
%         if allTs(cur_spike) - allTs(lastSave) > timesliceus
%             count = count + 1;
%             lastSave = cur_spike;  
%             
%             if freq > thres
%                 lastSpikeTimes = lastlastSpikeTimes; % remove flash data
%                 freq = 0;
%                 continue;
%             end
%             lastlastSpikeTimes = lastSpikeTimes; % We aren't seeing a flash
%             freq = 0;
%             
%             timeDif = time - lastSpikeTimes;
% 
%             % THIS IS EXP DECAY
%             res = 1./(1 + exp(-double(timeDif)./k));
%             
%             outfile = sprintf('%s/%s_%dms_%05d_past.png', ...
%                     outDir, filename, msin, count);
%             
%             imwrite(flipud(res.'), outfile); % 'InitialMagnification', 300
% 
%         end
%     end
%     
%     
%     %DEBUG: cur_spike = size(allTs, 1);
%     %DEBUG: count = 1000;
%     disp('Start backwards pass (into future)');
%     % cur_spike, count, are at the end
%     lastSpikeTimes = int32(zeros(DVS_RESOLUTION, DVS_RESOLUTION));
%     lastSave = timesliceus + lastSave;%cur_spike; % in case some events were left out from fwds
%     %cur_spike = lastSave
%     % Now we go backwards
%     while cur_spike > 1
%         %disp('in');
%         x = infomatrix1(cur_spike)+1; %offset as DVS numbers from 0
%         y = infomatrix2(cur_spike)+1;
%         %pol = infomatrix3(cur_spike); %unused atm
%         time = allTs(cur_spike);
%         cur_spike = cur_spike - 1;
% 
%         lastSpikeTimes(x, y) = time;
% 
%         %disp(cur_spike - lastSave);
%         if lastSave > size(allTs,1) || allTs(lastSave) - allTs(cur_spike) > timesliceus
%             timeDif = lastSpikeTimes - time;
%             timeDif(timeDif == -time) = time;
%             res = 1./(1 + exp(-double(timeDif)./k));
%             
%             outfile = sprintf('%s/%s_%dms_%05d_futr.png', ...
%                     outDir, filename, msin, count);
%             imwrite(flipud(res.'), outfile); %'InitialMagnification', 300
%             count = count - 1;
%             lastSave = cur_spike;
%         end
%     end
%     
% % %     
% % %     
% % % 
% % %     if strcmp(decay_type, 'linear') == 1
% % %         
% % %     elseif strcmp(decay_type, 'exponential') == 1
% % %             
% % %     elseif  strcmp(decay_type, 'sigmoid') == 1
% % %         
% % %     end
% % % 
% % % 
% % %     if nargin < 1
% % %        disp('Not enough input arguments.'); 
% % %     end
% % %     
% % %     
% % %     for arg = 1:2:(nargin -1)
% % %        switch lower(varargin{arg})
% % %            case 'k'
% % %                disp(varargin{arg + 1});
% % %            otherwise
% % %                disp('unrecognised arg');
% % %            end
% % %         end
% % % end
% % %     

    
    
    
    
 
end




