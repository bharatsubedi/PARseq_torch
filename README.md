# PARseq_torch
Torch version of parseq module
- I have convert parseq module into torch version and torch version accuracy is higher than pytorch lighting version.
  - I also prepare for onnx and tensorrt conversion. 
  - please test using corresponding python file


## data preparation
Check data stucture in `DATA.md` file 

## Train and test
- `bash train.sh`
- `bash test.sh`

## configuration
Change the configuration and path to fit your model at `config.yaml` file

## character set 
if you want to train with your own characterset, change the `charset.txt` file 
