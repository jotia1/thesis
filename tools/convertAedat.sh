## Usage: ./convertAedat.sh
# Execute matlab to preprocess an AEDAT file to pgm images to be input to 
# a neural network.

## TODO List
#   Find a way to sort data into seperate folders after processing so to be 
#       completely ready to run the neural network.

AEDATFILE='data/dvs_gt5.aedat'
OUTFOLDER='gt5'
MSIN='30'
K="240000" # MSIN * 1e3 * 8
MATLAB_LINE="preprocess('$AEDATFILE', '$OUTFOLDER', 'exponential', 'msin', $MSIN, 'k', $K), exit"

# Delete output folder if it already exists and recreate
if [ -d $OUTFOLDER ]; then
    rm -rf $OUTFOLDER
fi

mkdir $OUTFOLDER
mkdir $OUTFOLDER/data

matlab -nosplash -nodisplay -r "$MATLAB_LINE" > $OUTFOLDER/matlab.log

## HACK TO FIX NUMBERING
ls $OUTFOLDER/data | head -n 1 | xargs -I {} rm $OUTFOLDER/data/{}
ls $OUTFOLDER/data | tail -n 1 | xargs -I {} rm $OUTFOLDER/data/{}

