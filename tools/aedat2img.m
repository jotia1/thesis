function [ ] = aedat2img( filename, outDir, time_slices )
%AEDAT2IMG Convert a given aedat file to a series of images 
%   Given a filename for an aedat file create images representing buckets
%   of time time_slices and save images to outDir

       % Save images to file
       s = sprintf(strcat(outDir, '/2img%d.png'), cur_frame);
       imwrite(flipud(im.'), s);


end

