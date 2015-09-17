function [  ] = aedat2avi( filename )
%AEDAT2AVI Converts an aedat file to avi format
%   Given an aedat filename as a string convert the file to an avi video
%   based on 30ms time slices and save as filename.avi
%   Example Usage: aedat2avi('samp1.aedat')

    clc;
    [allAddr, allTs] = loadaerdat(filename);
    %extracts all the information from the address matrices in the form
    %[xcoordinate, ycoordinate, polarity]
    [infomatrix1, infomatrix2, infomatrix3] = extractRetina128EventsFromAddr(allAddr);
    %times are in us
    allTs = int32(allTs);
    
    DVS_RESOLUTION = 128;
    %CMAX = 10;
    f = figure;
    total_spikes = size(allTs);
    total_spikes = total_spikes(1);

    start = allTs(1);
    finish = allTs(end);
    total_time = finish - start;
    %frames = total_time/h.timestep;
    cur_frame = 1;
    num_frames = 1;
    
    im = zeros(DVS_RESOLUTION, DVS_RESOLUTION);
    cur_spike = 1;
    frame_len = 33 * 1000; % 33 ms converted to microseconds (
    
    while cur_spike < total_spikes;  % not finished
       im = zeros(DVS_RESOLUTION, DVS_RESOLUTION);
       while allTs(cur_spike) <  cur_frame * frame_len + start; %do frame\
           x = infomatrix1(cur_spike)+1;
           y = infomatrix2(cur_spike)+1;
           pol = infomatrix3(cur_spike);
           if pol == 1 % pixel went white
               im(x, y) = 1;
           else
               im(x, y) = 0;
           end
           cur_spike = cur_spike + 1;
           if cur_spike == total_spikes;
              break; 
           end
       end
       cur_frame = cur_frame + 1;
       imagesc(flipud(im.'), [-1, 1]);
       colormap(gray(256))
       M(cur_frame) = getframe(gcf);
       nvar = 10;
    end
    num_frames = cur_frame;
    cur_frame = 1;
    vidObj = VideoWriter('test3.avi', 'Motion JPEG AVI');
    open(vidObj);
    
    for k = 1:num_frames;
        disp(k)
        writeVideo(vidObj, M(k));
    end
    vidObj.FrameRate = 30;

    close(vidObj);
    movie(M, 2, 30);
end

