"""
Some configurations.
Copyright (c) 2019 Netease Youdao Information Technology Co.,Ltd.
Licensed under the GPL License 
Written by Yu Qian
"""
import numpy as np

# font
font_size = [170, 175]  # [25, 40]
underline_rate = 0.01
strong_rate = 0.07
oblique_rate = 0.02
font_dir = 'data/fonts/plate_ttf/'  #'/home/huyj/SRNet-master/datasets/fonts/english_ttf/'#
# standard_font_path = 'data/fonts/englisth_ttf/arial.ttf'#'/home/huyj/SRNet-master/datasets/fonts/english_ttf/OpenSans-Regular.ttf'#
standard_font_path = 'data/fonts/plate_ttf/platech.ttf'#'/home/huyj/SRNet-master/datasets/fonts/english_ttf/OpenSans-Regular.ttf'#

# text
text_dir = '/home/huyj/SRNet-master/SRNet-Datagen/Synthtext/data/newsgroup/'
text_filepath = 'data/lly_text.txt'#'/home/huyj/SRNet-master/SRNet-Datagen/Synthtext/data/texts.txt'#
capitalize_rate = 0.1
uppercase_rate = 0.04

# background
bg_filepath = 'data/imnames.cp'
temp_bg_path = '/data2/lily/datasets/bg_img/'

## background augment
brightness_rate = 0.8
brightness_min = 0.7
brightness_max = 1.5
color_rate = 0.8
color_min =0.7
color_max = 1.3
contrast_rate = 0.8
contrast_min = 0.7
contrast_max = 1.3

# curve
is_curve_rate = 0.05
curve_rate_param = [0.1, 0] # scale, shift for np.random.randn()

# perspective
rotate_param = [1, 0] # scale, shift for np.random.randn()
zoom_param = [0.1, 1] # scale, shift for np.random.randn()
shear_param = [2, 0] # scale, shift for np.random.randn()
perspect_param = [0.0005, 0] # scale, shift for np.random.randn()

# render

## surf augment
elastic_rate = 0.001
elastic_grid_size = 4
elastic_magnitude = 2

## colorize
padding_ud = [0, 10]
padding_lr = [0, 20]
is_border_rate = 0.02
is_shadow_rate = 0.02
shadow_angle_degree = [1, 3, 5, 7] # shift for shadow_angle_param
shadow_angle_param = [0.5, None] # scale, shift for np.random.randn()
shadow_shift_param = np.array([[0, 1, 3], [2, 7, 15]], dtype = np.float32) # scale, shift for np.random.randn()
shadow_opacity_param = [0.1, 0.5] # shift for shadow_angle_param
color_filepath = 'data/colors.cp'
use_random_color_rate = 0.5
