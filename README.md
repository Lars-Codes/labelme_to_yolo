# Labelme JSON annotations to YOLOv5PyTorch annotation format  

## Parameters 
**--labelme_dir** - Path to directory with LabelmeJSON files. It is recommended to remove the image files for organization purposes, but not required. 
**--delete_json** - Default value is "no." Type 'y' to delete LabelMeJSON files after conversion.
**--val_split** - Default value is 0.10. Type percentage, in decimal form, of data to be in validation set. 
**--test_split** - Default value is 0.10. Type percentage, in decimal form, of data to be in validation set. 

`
  python3 converter.py --labelme_dir /path/to/labelme/dir/ --delete-json n --val_split 0.10 --test_split 0.10 
`
## Constraints 
Val split and test split must sum to 1. 

