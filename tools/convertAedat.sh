## Usage: ./convertAedat.sh
# Execute matlab to preprocess an AEDAT file to pgm images to be input to 
# a neural network.

## TODO List
#   Find a way to sort data into seperate folders after processing so to be 
#       completely ready to run the neural network.

AEDATFILE='data/dvs_gt4.aedat'
OUTFOLDER='gt4'
MSIN='30'
K='0.5'

# Delete output folder if it already exists and recreate
if [ -d $OUTFOLDER ]; then
    rm -rf $OUTFOLDER
fi

mkdir $OUTFOLDER
mkdir $OUTFOLDER/data

#matlab -nodisplay -r 'runpreprocess, exit'
matlab -nosplash -nodisplay -r "preprocess('$AEDATFILE', '$OUTFOLDER', 'exponential', 'msin', $MSIN, 'k', $K), exit"
