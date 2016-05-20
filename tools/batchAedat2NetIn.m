%% Batch aedat2NetIn
% Process many aedat files to network input files sequentially
l5 = log(0.5);
k_values = [1, 16, 33, 100];
lk_values = [2*1e3, 32*1e3, 66*1e3, 100*1e3];
ek_values = [(-1/l5)*1e3, (-16/l5)*1e3, (-33/l5)*1e3, (-100/l5)*1e3];
speed_values = [2, 4, 6, 8];
size_values = [4, 6, 8];
decay_values = {'linear', 'exp'};
sep_sizes = [];

for size_i = 1 : numel(size_values);
    for speed_i = 1 : numel(speed_values);
        for k_i = 1 : numel(ek_values);
            for decay_i = 1 : numel(decay_values)
                
                if decay_i == 1
                    k = lk_values(k_i);
                else
                    k = ek_values(k_i);
                end
                
                speed = speed_values(speed_i);
                dotSize = size_values(size_i);
                 decay = decay_values{decay_i};

                %fprintf('k: %d, speed: %d, size: %d\n', k, speed, size );
                infilename = sprintf('../data/8AD/recordings/onight_%d_%d.aedat', dotSize, speed);
                outfilename = sprintf('sec_%d_%d_%dk_%s', dotSize, speed, k_values(k_i), decay);
                %[ xs, ys, ts, ~, seps ] = trimEvents(infilename);
                %fprintf('infile: %s, size_seps: %d\n', infilename, size(seps,1));
                
                fprintf('infile: %s, outfile: %s, K: %d, decay: %s, \n', infilename, outfilename, k, decay);
                aedat2NetIn(infilename, outfilename, 128, 128, 1, 30, false, decay, k)
             end
         end
        sep_sizes(end + 1) = size(seps, 1);
    end
end
sep_sizes;   % Left in incase i want to explore this later