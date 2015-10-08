function [ ] = aedat2img( filename, outDir, time_slices )
%AEDAT2IMG Convert a given aedat file to a series of images 
%   Given a filename for an aedat file create images representing buckets
%   of time time_slices us and save images to outDir.
%   e.g. aedat2img('samp1.aedat', 'imgout', 30) -> imgout/ will be full of
%   images.

    [allAddr, allTs] = loadaerdat(filename);
    %extracts all the information from the address matrices in the form
    %[xcoordinate, ycoordinate, polarity]
    [infomatrix1, infomatrix2, infomatrix3] = extractRetina128EventsFromAddr(allAddr);
    %times are in us
    allTs = int32(allTs);
    
    DVS_RESOLUTION = 128;
    total_spikes = size(allTs);
    total_spikes = total_spikes(1);

    start = allTs(1);
    finish = allTs(end);
    total_time = finish - start;
    cur_frame = 1;
    num_frames = 1;
    
    im = zeros(DVS_RESOLUTION, DVS_RESOLUTION);
    cur_spike = 1;
    frame_len = time_slices * 1000; % time_slices ms converted to microseconds
    
    while cur_spike < total_spikes;  % not finished
       im = zeros(DVS_RESOLUTION, DVS_RESOLUTION);
       bucket_start = cur_spike;
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
       % Save images to file
       s = sprintf(strcat(outDir, '/img%d.png'), cur_frame);
       if (cur_spike - bucket_start) < 46  % 46 is thres for 30ms
            imwrite(flipud(im.'), s);
       end
       if mod(cur_frame, 50) == 0
          disp(sprintf('Image number %d saved.', cur_frame)); 
       end
    end
end

