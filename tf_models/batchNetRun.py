import os as os
import shutil as shutil

k_values = [1, 16, 33, 100, 165]
speed_values = [2, 4, 6, 8]
size_values = [4, 6, 8]
decay_values = ['linear', 'exp']

params = {}

net = 'PILOT'  # Options: AE, ATTN, CONV, PILOT
if net == "ATTN":
    import net4 as net
    results_root = 'attn_1m_batch_results'
    model_base = '8AD_attn'
elif net == "CONV":
    import cnet1 as net
    results_root = 'attConv64_batch_results'
    model_base = '8AD_attn'
    params = {'num_features': 9,
                'conv_size': 6,
                'fc_units': 64,
                }
elif net == "PILOT":
    import net2 as net
    results_root = 'pilotA_batch_results'
    model_base = 'AAD'
elif net == "AE":
    import autoEnc as net
    results_root = 'AE64h_batch_results'
    model_base = '8AD_attn'
    params = {'num_hidden': 64}
else:
    raise Exception("Net not known:" + net)

# File system constants
data_dir = '../data/AAD/processed'

# If previous work exists, remove it
if os.path.exists(results_root):
    #subprocess.call('rm -rf {}'.format(results_root), shell=True) 
    shutil.rmtree(results_root)
os.makedirs(results_root)

#  NETWORK PARMAS  
load_model = False
save_model = True
write_images = True
batch_size = 100
total_steps = 50001
learning_rate = 0.5
params.update({})  # any new consistant variables here
other_params = params

# Write parameters to file
f = open(os.path.join(results_root, 'params.txt'), 'w')
header = "root = {}\n-------------------\n \
load_model = {}\nsave_model = {}\nwrite_imgs = {}\n \
batch_size = {}\ntotal_steps = {}\nlearning_rate = {}\n \
model_base = {}\nother_params = {}\n"
f.write(header.format(results_root, load_model, save_model,
        write_images, batch_size, total_steps, learning_rate, model_base, 
        other_params))
f.close()

for size in size_values:
    size_dir = os.path.join(results_root, str(int(size)))
    os.makedirs(size_dir)
    open(os.path.join(results_root, 'size'), 'w').close() # Leave a file so we know
    for speed in speed_values:
        speed_dir = os.path.join(size_dir, str(int(speed)))
        os.makedirs(speed_dir)
        open(os.path.join(size_dir, 'speed'), 'w').close() # Leave a file so we know
        for k in k_values:
            k_dir = os.path.join(speed_dir, str(int(k)))
            os.makedirs(k_dir)
            open(os.path.join(speed_dir, 'k'), 'w').close() # Leave a file so we know
            for decay in decay_values:
                decay_dir = os.path.join(k_dir, decay)
                os.makedirs(decay_dir)
                
                model_id = model_base + "_{0}_{1}_{2}k_{3}".format(size, speed, int(k), decay)
                datafile = os.path.join(data_dir,  model_id) + ".mat"
                tensorboard_dir = decay_dir
                save_dir = decay_dir
                image_dir = tensorboard_dir
                
                print("current model:", os.path.join(decay_dir, datafile))
                net.runNet(datafile=datafile, tensorboard_dir=tensorboard_dir, save_dir=save_dir,
                    model_id=model_id, image_dir=image_dir, load_model=load_model,
                    save_model=save_model, write_image=write_images, batch_size=batch_size,
                    total_steps=total_steps, learning_rate=learning_rate, 
                    other_params=other_params)
