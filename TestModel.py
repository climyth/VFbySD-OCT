from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.models import Model
from keras.utils.vis_utils import plot_model
from keras.layers import Dense, GlobalAveragePooling2D
import xlrd
import numpy as np
from VFOCT_DataLoad import *
from VFOCT_Model import *
from matplotlib import pyplot as plt
from pathlib import Path


# Setup ====================================================================================
image_file = "testset/OCT016.jpg"  # combined OCT image
vf_file = "testset/VFTest.xlsm"   # ground truth visual field file
vf_sheet = "Test"  # data sheet in excel file
oct_filename_col = 0   # column number of OCT filename (minimum = 0)
thv_start_col = 6   # column number where THV values start (minimum = 0)
weight_file = "weights/inceptionV3FinalTrained.hdf5"  # trained model
# ===========================================================================================


def read_visual_field(excel_file, sheet_name, oct_filename, filename_col=0, thv_col=135):
    worksheet = xlrd.open_workbook(excel_file).sheet_by_name(sheet_name)
    nrows = worksheet.nrows - 1
    y_data = np.empty([52])

    for r in range(1, nrows):
        cur_filename = worksheet.cell_value(r, filename_col)
        if Path(cur_filename).name == Path(oct_filename).name:
            c1 = 0
            for c in range(0, 54):
                if c != 25 and c != 34:  # physiological scotomas excluded
                    y_data[c1] = worksheet.cell_value(r, c + thv_col)
                    c1 = c1+1
            break
    return y_data


def draw_visual_field(field_values, mplt, start_pos=(0, 0), rect_size=(35, 25)):
    #  ===== config =====================================================
    rect_pos = [                  [3,0],[4,0],[5,0],[6,0],
                            [2,1],[3,1],[4,1],[5,1],[6,1],[7,1],
                      [1,2],[2,2],[3,2],[4,2],[5,2],[6,2],[7,2],[8,2],
                [0,3],[1,3],[2,3],[3,3],[4,3],[5,3],[6,3],      [8,3],
                [0,4],[1,4],[2,4],[3,4],[4,4],[5,4],[6,4],      [8,4],
                      [1,5],[2,5],[3,5],[4,5],[5,5],[6,5],[7,5],[8,5],
                            [2,6],[3,6],[4,6],[5,6],[6,6],[7,6],
                                  [3,7],[4,7],[5,7],[6,7]]
    min_vf = 5   # for color range calculation
    max_vf = 35  # for color range calculation
    # ====================================================================

    for i in range(0, 52):
        x = start_pos[0] + rect_pos[i][0] * rect_size[0]
        y = start_pos[1] + rect_pos[i][1] * rect_size[1]
        vf = field_values[i]
        bg_col = (vf - min_vf) / (max_vf - min_vf)   # max = 35, min = 5
        if bg_col < 0:
            bg_col = 0
        if bg_col > 1.0:
            bg_col = 1.0
        txt_color = 'black'
        if bg_col < 0.5:
            txt_color = 'white'
        rect = mplt.Rectangle((x, y), rect_size[0], rect_size[1], fill=True, fc=(bg_col, bg_col, bg_col))
        mplt.gca().add_patch(rect)
        mplt.gca().text(x+rect_size[0]/2, y+rect_size[1]/2, "{:.1f}".format(vf), fontsize=6,
                        horizontalalignment='center', verticalalignment='center', color=txt_color)


# make deep learning model
model = GetModel()
model.load_weights(weight_file)

# load OCT image
img = load_img(image_file)
x_img = img_to_array(img)
x_img = x_img.reshape((1,) + x_img.shape)

# load ground-truth values
truth = read_visual_field(vf_file, vf_sheet, image_file, oct_filename_col, thv_start_col)

# get visual field prediction
pred = model.predict(x_img)

# draw prediction
plt.clf()
plt.imshow(img)   # draw OCT image
# draw ground-truth visual field
plt.gca().text(10, 180, "Ground truth visual field", fontsize=6, color='black')
draw_visual_field(truth, plt, start_pos=(5, 190))
# draw predicted visual field
plt.gca().text(10, 410, "Predicted visual field", fontsize=6, color='black')
draw_visual_field(pred[0], plt, start_pos=(5, 420))
plt.axis("scaled")
plt.show()
