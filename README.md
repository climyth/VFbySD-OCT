# Visual field prediction from SD-OCT

### Features
- Inception V3 backboned deep learning model
- Predicts Humphrey's visual field 24-2 total threshold values from Zeiss SD-OCT
- Predicts entire picture of visual field
- Uses combined OCT images (macula scan + ONH scan);
- Mean prediction error is 4.79 dB

### How can I test OCT image?
1. Download all files and test images from github
2. You can download weight file here: https://drive.google.com/open?id=1MEBzcT6MG9OfFdot_6mwhnIPsHdjVCnQ
3. Open TestModel.py
4. Modify "Setup"
.. image:: https://github.com/climyth/VFbySD-OCT/tree/master/example/TestModelSetup.PNG?raw=true
![](https://github.com/climyth/VFbySD-OCT/tree/master/example/TestModelSetup.PNG)
5. Run TestModel.py
6. You can see the popup window like below.
![]()
