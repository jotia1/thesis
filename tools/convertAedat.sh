## Usage: ./convertAedat.sh
# Execute matlab to preprocess an AEDAT file to pgm images to be input to 
# a neural network.

## TODO List
#   Find a way to sort data into seperate folders after processing so to be 
#       completely ready to run the neural network.


matlab -nodisplay -r 'runpreprocess, exit'

