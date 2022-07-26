import cv2
import pygame
from pygame import freetype
import numpy as np
import os
import json
import pygame.locals
import tqdm


def render_normal(font, text):
    line_spacing = font.get_sized_height() + 1
    line_bounds = font.get_rect(text)
    fsize = (round(2.0 * line_bounds.width), round(1.25 * line_spacing))
    surf = pygame.Surface(fsize, pygame.locals.SRCALPHA, 32)
    x, y = 0, line_spacing

    rect = font.render_to(surf, (x, y), text)
    rect.x = x + rect.x
    rect.y = y - rect.y

    surf = pygame.surfarray.pixels_alpha(surf).swapaxes(0, 1)
    loc = np.where(surf > 20)
    miny, minx = np.min(loc[0]), np.min(loc[1])
    maxy, maxx = np.max(loc[0]), np.max(loc[1])
    return surf[miny:maxy + 1, minx:maxx + 1], rect


def make_standard_text(font_path, text, shape, padding = 0.1, color = (0, 0, 0), init_fontsize = 25):
    font = freetype.Font(font_path)
    font.antialiased = True
    font.origin = True
    fontsize = init_fontsize
    font.size = fontsize
    pre_remain = None
    if padding < 1:
        border = int(min(shape) * padding)
    else:
        border = int(padding)
    # target_shape = tuple(np.array(shape) - 2 * border)
    target_shape = tuple(np.array(shape))
    # while True:
    #     rect = font.get_rect(text)
    #     res_shape = tuple(np.array(rect[1:3]))
    #     remain = np.min(np.array(target_shape) - np.array(res_shape))
    #     if pre_remain is not None:
    #         m = pre_remain * remain
    #         if m <= 0:
    #             if m < 0 and remain < 0:
    #                 fontsize -= 1
    #             if m == 0 and remain != 0:
    #                 if remain < 0:
    #                     fontsize -= 1
    #                 elif remain > 0:
    #                     fontsize += 1
    #             break
    #     if remain < 0:
    #         if fontsize == 2:
    #             break
    #         fontsize -= 1
    #     else:
    #         fontsize += 1
    #     pre_remain = remain
    #     font.size = fontsize

    surf, rect = render_normal(font, text)
    if np.max(np.array(surf.shape) - np.array(target_shape)) > 0:
        scale = np.min(np.array(target_shape, dtype = np.float32) / np.array(surf.shape, dtype = np.float32))
        to_shape = tuple((np.array(surf.shape) * scale).astype(np.int32)[::-1])
        surf = cv2.resize(surf, to_shape)
    canvas = np.zeros(shape, dtype = np.uint8)
    tly, tlx = int((shape[0] - surf.shape[0]) // 2), int((shape[1] - surf.shape[1]) // 2)
    canvas[tly:tly+surf.shape[0], tlx:tlx+surf.shape[1]] = surf
    canvas = ((1. - canvas.astype(np.float32) / 255.) * 127.).astype(np.uint8)
    bounding_box = [tlx-1, tly-1, tlx+surf.shape[1]+1, tly+surf.shape[0]+1]

    return cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB), bounding_box



if __name__ == '__main__':
    '''
    1. text1是json1文件中的[1][-1]
    2. surf_h, surf_w 和 i_s原图一样
    3. font.size 是json1文件中的[1][1]
    '''
    pygame.init()
    freetype.init()
    standard_font_path = '/home/lily/SRNet-datagen/Synthtext/data/fonts/cn_ttf/Deng.ttf'
    f1 = open('/home/lily/SRNet-datagen/json1.json', 'r')
    content1 = f1.read()
    a = json.loads(content1)
    f1.close()
    temp_dict = {}

    # text1 = a['0000'][1][-1]
    # img_0 = cv2.imread(f'/home/lily/SRNet-datagen/i_s/0000.png')
    # surf_h_w = img_0.shape
    # i_t, bounding_box = make_standard_text(standard_font_path, text1, (surf_h_w[0], surf_h_w[1]),
    #                          init_fontsize=a['0000'][1][1])
    # cv2.imwrite(f'/home/lily/SRNet-datagen/i_o/0002.png', i_t, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
    # print(bounding_box)
    # a_tqdm = tqdm(a)
    for k, v in tqdm.tqdm(a.items(), total=len(a)):
        text1 = a[k][1][-1]
        img_0 = cv2.imread(f'/data2/lily/output/srnet_datagen/i_s/{k}.png')
        surf_h_w = img_0.shape
        i_t, bounding_box = make_standard_text(standard_font_path, text1, (surf_h_w[0], surf_h_w[1]),
                                 init_fontsize=a[k][1][1])
        cv2.imwrite(f'/data2/lily/output/srnet_datagen/o_t/{k}.png', i_t, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        temp_dict[k] = bounding_box
    temp1_json = json.dumps(temp_dict, indent=4, ensure_ascii=False)
    f1 = open('bounding_box_ot.json', 'w')
    f1.write(temp1_json)
    f1.close()

    print("DONE!")
