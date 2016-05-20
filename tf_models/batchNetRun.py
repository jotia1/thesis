import os as os
import shutil as shutil
import net2 as net

k_values = [5*1e4, 1*1e5, 2.5*1e5]
speed_values = [2, 4, 6, 8]
size_values = [4, 6, 8]
decay_values = ['linear', 'exp']

# File system constants
results_root = 'net2_batch_results'
data_dir = '../data/8AD/processed'

# If previous work exists, remove it
if os.path.exists(results_root):
    #subprocess.call('rm -rf {}'.format(results_root), shell=True) 
    shutil.rmtree(results_root)
os.makedirs(results_root)

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
                
                model_id = "{0}_{1}_{2}k_{3}".format(size, speed, int(k/1000), decay)
                datafile = os.path.join(data_dir,  model_id) + ".mat"
                #datafile = '8_8_exp.mat'
                tensorboard_dir = decay_dir
                save_dir = decay_dir
                image_dir = tensorboard_dir
                load_model = False
                save_model = True
                write_images = True
                batch_size = 100
                total_steps = 500001
                learning_rate = 0.5
                
                print("current model:", os.path.join(decay_dir, datafile))
                net.runNet(datafile=datafile, tensorboard_dir=tensorboard_dir, save_dir=save_dir,
                    model_id=model_id, image_dir=image_dir, load_model=load_model,
                    save_model=save_model, write_image=write_images, batch_size=batch_size,
                    total_steps=total_steps, learning_rate=learning_rate)
