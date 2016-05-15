%% Batch aedat2NetIn
% Process many aedat files to network input files sequentially

k_values = [5*1e4, 1*1e5, 25*1e4, 1*1e5];
speed_values = [2, 4, 6, 8];
size_values = [4, 6, 8];
decay_values = {'linear', 'exp'};

for size_i = 1 : numel(size_values);
    for speed_i = 1 : numel(speed_values);
        for k_i = 1 : numel(k_values);
            for decay_i = 1 : numel(decay_values)
            
                k = k_values(k_i);
                speed = speed_values(speed_i);
                size = size_values(size_i);
                decay = decay_values{decay_i};

                %fprintf('k: %d, speed: %d, size: %d\n', k, speed, size );
                infilename = sprintf('../data/8AD_samp/onight_%d_%d.aedat', speed, size);
                outfilename = sprintf('%d_%d_%dk_%s', speed, size, k/1000, decay);
                
                fprintf('infile: %s, outfile: %s, K: %d, decay: %s, \n', infilename, outfilename, k, decay);
                %aedat2NetIn(infilename, outfilename, 128, 128, 1, 30, false, decay, k)
            end
        end
    end
end