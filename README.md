# Visual field prediction from SD-OCT
![](https://github.com/climyth/VFbySD-OCT/blob/master/example/FigExamplpes.jpg?raw=true)

### Features
- Inception V3 backboned deep learning model
- Predicts Humphrey's visual field 24-2 total threshold values from Zeiss SD-OCT
- Predicts entire picture of visual field
- Uses combined OCT images (macula scan + ONH scan);
- Mean prediction error is 4.79 dB

### Prerequisites
- python 3.6
- tensorflow >= 1.6.0
- keras 2.2.4

### How can I test OCT image?
1. Download all files from github
2. You can download weight file here: https://drive.google.com/open?id=1MEBzcT6MG9OfFdot_6mwhnIPsHdjVCnQ
3. Open TestModel.py
4. Modify "Setup"
```python
# Setup ====================================================================================
image_file = "testset/OCT016.jpg"  # combined OCT image
vf_file = "testset/VFTest.xlsm"   # ground truth visual field file
vf_sheet = "Test"  # data sheet in excel file
oct_filename_col = 0   # column number of OCT filename (minimum = 0)
thv_start_col = 6   # column number where THV values start (minimum = 0)
weight_file = "weights/inceptionV3FinalTrained.hdf5"  # trained model
# ===========================================================================================
```
5. Run TestModel.py;
6. You can see the popup window like below.
![](https://github.com/climyth/VFbySD-OCT/blob/master/example/TestWindow.JPG?raw=true)

### How can I make "combined OCT" image?
1. Download "panomaker.exe" in "utils" folder
2. In utils folder, there are sample OCT images to generate combined OCT image.<br/>
   You need 2 OCT images in pair like below. (1) macular OCT (2) ONH OCT<br/>
   ![](https://github.com/climyth/VFbySD-OCT/blob/master/example/oct_example.jpg?raw=true)
   <br/>
3. Image file name must follow the rule:<br/>
   (1) macular OCT: patientID_examdate_1.jpg  (ex. 012345678_20180403_1.jpg)<br/>
   (2) ONH OCT: patientID_examdate_2.jpg   (ex. 012345678_20180403_2.jpg)<br/>
   Note: Two images must have the same name (the only difference is last number _1 or _2)
4. Run "panomaker.exe"<br/><br/>
![](https://github.com/climyth/VFbySD-OCT/blob/master/example/panomaker.png?raw=true)
<br/><br/>
5. set source folder and output folder
6. press Start button. That's it!

### How can I train model with my own OCT iamges?
1. Prepare your own OCT iamges and visual field data (excel file)
2. Generate "combined OCT" images from your train set
3. In visual field excel file, <br/>
   (1) your data must be in "Train" data sheet
   (2) the first column must be combined OCT image file paths
   (3) visual field total threshold values must begin at 7th column by default (otherwise, you need to modify "LoadData")
4. Modify 'Setup' in VFOCT_Train.py
```python
# Setup ====================================================================
image_folder = ""   # root image folder for train set (combined OCT images)
vf_file = "VFTrain.xlsm"   # visual field data excel file
weight_save_folder = "Weights"
graph_save_folder = ""   # model graph output folder
pretrained_weights = ""   # if no pretrained weight, just leave ""
tensorboard_log_folder = "logs"
# ==========================================================================
```
5. Run the VFOCT_Train.py
6. You can monitor loss trend in tensorboard. This is our trend curve for example.<br/><br/>
![](https://github.com/climyth/VFbySD-OCT/blob/master/example/train_log_1.jpg?raw=true)
<br/><br/>
7. To prevent overfitting, we used "repeated random sub-sampling cross validation method". To do this, just repeat to run VFOCT_Train.py. In each run, you can set "pretrained_weights" to continue the training from last weight file.


### For Research Use Only
The performance characteristics of this product have not been evaluated by the Food and Drug Administration and is not intended for commercial use or purposes beyond research use only.
