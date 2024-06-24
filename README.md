# Instructions for usage 

1. Create a directory to place your data.
2. Specify the data path at the bottom of the script.

This will convert LabelMe JSON to YOLOv5 PyTorch format and split them into validation, testing, and training
datasets. The default split is 10% validation and 10% testing, but you may specify a different 
split in the split_data function. 
