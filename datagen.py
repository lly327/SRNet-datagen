import os
import cv2
import cfg
from Synthtext.gen import datagen, multiprocess_datagen
import json

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def main():
    
    i_t_dir = os.path.join(cfg.data_dir, cfg.i_t_dir)
    i_s_dir = os.path.join(cfg.data_dir, cfg.i_s_dir)
    t_sk_dir = os.path.join(cfg.data_dir, cfg.t_sk_dir)
    t_t_dir = os.path.join(cfg.data_dir, cfg.t_t_dir)
    t_b_dir = os.path.join(cfg.data_dir, cfg.t_b_dir)
    t_f_dir = os.path.join(cfg.data_dir, cfg.t_f_dir)
    mask_t_dir = os.path.join(cfg.data_dir, cfg.mask_t_dir)
    
    makedirs(i_t_dir)
    makedirs(i_s_dir)
    makedirs(t_sk_dir)
    makedirs(t_t_dir)
    makedirs(t_b_dir)
    makedirs(t_f_dir)
    makedirs(mask_t_dir)


    mp_gen = datagen()
    # mp_gen.multiprocess_runningqueue()
    digit_num = len(str(cfg.sample_num)) - 1
    temp1_dict = {}
    temp2_dict = {}
    for idx in range(cfg.sample_num):
        print("Generating step {:>6d} / {:>6d}".format(idx + 1, cfg.sample_num))
        i_t, i_s, t_sk, t_t, t_b, t_f, mask_t, bounding_box1, bounding_box2,\
        font_type, font_size, num1, text1, num2, text2 = mp_gen.gen_srnet_data_with_background()
        i_t_path = os.path.join(i_t_dir, str(idx).zfill(digit_num) + '.png')
        i_s_path = os.path.join(i_s_dir, str(idx).zfill(digit_num) + '.png')
        t_sk_path = os.path.join(t_sk_dir, str(idx).zfill(digit_num) + '.png')
        t_t_path = os.path.join(t_t_dir, str(idx).zfill(digit_num) + '.png')
        t_b_path = os.path.join(t_b_dir, str(idx).zfill(digit_num) + '.png')
        t_f_path = os.path.join(t_f_dir, str(idx).zfill(digit_num) + '.png')
        pic1, pic2 = [], []
        pic1.append(bounding_box1)
        pic1.append([font_type, font_size, num1, text1])
        pic2.append(bounding_box2)
        pic2.append([font_type, font_size, num2, text2])
        temp1_dict[str(idx).zfill(digit_num)] = pic1
        temp2_dict[str(idx).zfill(digit_num)] = pic2

        mask_t_path = os.path.join(cfg.data_dir, cfg.mask_t_dir, str(idx).zfill(digit_num) + '.png')
        cv2.imwrite(i_t_path, i_t, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        cv2.imwrite(i_s_path, i_s, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        cv2.imwrite(t_sk_path, t_sk, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        cv2.imwrite(t_t_path, t_t, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        cv2.imwrite(t_b_path, t_b, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        cv2.imwrite(t_f_path, t_f, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        cv2.imwrite(mask_t_path, mask_t, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
    temp1_json = json.dumps(temp1_dict, indent=4, ensure_ascii=False)
    temp2_json = json.dumps(temp2_dict, indent=4, ensure_ascii=False)
    f1 = open('bounding_box1.json', 'w')
    f2 = open('bounding_box2.json', 'w')
    f1.write(temp1_json)
    f2.write(temp2_json)
    f1.close()
    f2.close()
    # mp_gen.terminate_pool()

if __name__ == '__main__':
    main()
