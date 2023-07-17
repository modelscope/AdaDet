# Copyright (c) Alibaba, Inc. and its affiliates.
import math

import cv2
import numpy as np
from adadet.utils.constants import StateCode
from adadet.utils.visualization import get_color

from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks, ThirdParty
from modelscope.utils.logger import get_logger
from .deploy import DEPLOYS, Deploy

logger = get_logger()


@DEPLOYS.register_module('adadet_deploy', 'FaceRetouch')
class FaceRetouch(Deploy):

    def __init__(self, model_id, rules) -> None:
        super().__init__()
        self.rules = rules
        self.is_video = self.rules.get('is_video', False)
        self.filter_style = self.rules.get('filter_style', 'none')
        self.strength_smooth = self.rules.get('smooth', 60)
        self.strength_white = self.rules.get('white', 60)
        self.strength_enlarge_eye = self.rules.get('enlarge_eye', 60)
        self.strength_slim_face = self.rules.get('slim_face', 60)

        self.filter_map = None
        self.filter_map_sun = cv2.imread('adadet/resource/040_sun.png')
        self.filter_map_japan = cv2.imread('adadet/resource/038_japan.png')
        self.filter_map_marshmallow = cv2.imread(
            'adadet/resource/020_marshmallow.png')
        self.filter_map_magic = cv2.imread('adadet/resource/007_magic.png')
        self.filter_map_film = cv2.imread('adadet/resource/058_film.png')
        self.filter_map_white = cv2.imread('adadet/resource/whiten.jpg')

        self.pipeline = pipeline(
            Tasks.face_2d_keypoints,
            model=model_id,
            third_party=getattr(ThirdParty, 'ADADET', 'adadet'))

    def __call__(self, input) -> dict:

        self.set_params(self.filter_style, self.strength_smooth,
                        self.strength_white, self.strength_enlarge_eye,
                        self.strength_slim_face)

        img = cv2.imread(input)
        _h, _w, _c = img.shape
        h, w = self.get_size(_h, _w, 720)
        image = cv2.resize(img, dsize=(w, h))

        res = self.pipeline(image)
        keypoints = res[OutputKeys.KEYPOINTS]

        output = self.face_retouch(image, keypoints)

        return output

    def visualize(self, input_video, res, save_path) -> None:
        cv2.imwrite(save_path, res)
        return save_path

    def get_size(self, h, w, max=720):
        if min(h, w) > max:
            if h > w:
                h, w = int(max * h / w), max
            else:
                h, w = max, int(max * w / h)
        return h, w

    def get_filter_map(self):
        if self.filter_style == 'none':
            self.filter_map = None
        elif self.filter_style == 'sun':
            self.filter_map = self.filter_map_sun
        elif self.filter_style == 'japan':
            self.filter_map = self.filter_map_japan
        elif self.filter_style == 'marshmallow':
            self.filter_map = self.filter_map_marshmallow
        elif self.filter_style == 'magic':
            self.filter_map = self.filter_map_magic
        elif self.filter_style == 'film':
            self.filter_map = self.filter_map_film
        else:
            self.filter_map = None

    def set_params(self, filter_style, smooth, white, enlarge_eye, slim_face):
        self.filter_style = filter_style
        self.strength_smooth = smooth / 100.0
        self.strength_white = white / 100.0
        self.strength_enlarge_eye = enlarge_eye / 100.
        self.strength_slim_face = slim_face / 100.0

        self.get_filter_map()

    def lookuptable(self, img, lut, strength):
        h, w, c = img.shape
        img = img.astype(np.float32)
        for i in range(w):
            for j in range(h):
                pixel_b, pixel_g, pixel_r = img[j, i]
                val_b = pixel_b * 0.24705882
                quad1_x = math.floor(math.floor(val_b) / 8.0)
                quad1_y = math.floor(val_b) - quad1_x * 8.0
                quad2_x = math.floor(math.ceil(val_b) / 8.0)
                quad2_y = math.ceil(val_b) - quad2_x * 8.0

                texpos1_y = quad1_y * 0.125 + 0.5 / 511.0 + (
                    0.125 - 1.0 / 511.0) * float(pixel_r) / 255.0
                texpos1_x = quad1_x * 0.125 + 0.5 / 511.0 + (
                    0.125 - 1.0 / 511.0) * float(pixel_g) / 255.0
                texpos1_x = int(texpos1_x * 511)
                texpos1_y = int(texpos1_y * 511)
                if texpos1_x > 511:
                    texpos1_x = 511
                if texpos1_y > 511:
                    texpos1_y = 511

                texpos2_y = quad2_y * 0.125 + 0.5 / 511.0 + (
                    0.125 - 1.0 / 511.0) * float(pixel_r) / 255.0
                texpos2_x = quad2_x * 0.125 + 0.5 / 511.0 + (
                    0.125 - 1.0 / 511.0) * float(pixel_g) / 255.0
                texpos2_x = int(texpos2_x * 511)
                texpos2_y = int(texpos2_y * 511)
                if texpos2_x > 511:
                    texpos2_x = 511
                if texpos2_y > 511:
                    texpos2_y = 511

                alpha = val_b - math.floor(val_b)

                new_color1 = lut[texpos1_x, texpos1_y]
                new_color2 = lut[texpos2_x, texpos2_y]

                pix_b = (1.0 - alpha) * new_color1[0] + alpha * new_color2[0]
                pix_g = (1.0 - alpha) * new_color1[1] + alpha * new_color2[1]
                pix_r = (1.0 - alpha) * new_color1[2] + alpha * new_color2[2]

                pix_b = pix_b * strength + pixel_b * (1.0 - strength)
                pix_g = pix_g * strength + pixel_g * (1.0 - strength)
                pix_r = pix_r * strength + pixel_r * (1.0 - strength)

                img[j, i] = [pix_b, pix_g, pix_r]

        img = img.astype(np.uint8)
        return img

    def filter(self, img, lut, strength):
        if strength < 0.01 or lut is None:
            return img

        img = self.lookuptable(img, lut, strength)
        return img

    def blend(self, a, iter=3):
        val = a
        for i in range(iter):
            if val <= 0.5:
                val = 2.0 * val * val
            else:
                val = 1.0 - (1.0 - val) * (1.0 - val) * 2.0
        return val

    def smooth(self, img, strength):
        if strength < 0.01:
            return img

        h, w, c = img.shape
        b, g, r = cv2.split(img)
        small_g = cv2.resize(g, dsize=None, fx=0.5, fy=0.5)
        gauss_g = cv2.GaussianBlur(small_g, (9, 9), 5, 5, cv2.BORDER_DEFAULT)
        gauss_g = cv2.resize(gauss_g, dsize=(w, h))

        sample = np.zeros_like(gauss_g)
        highpass = np.zeros_like(gauss_g)
        sample = sample.astype(np.float32)
        highpass = highpass.astype(np.float32)
        gauss_g = gauss_g.astype(np.float32)
        img = img.astype(np.float32)

        for i in range(w):
            for j in range(h):
                pix_g = g[j, i]
                pix_gauss_g = gauss_g[j, i]
                val = (pix_g - pix_gauss_g + 127.0) / 255.0
                val = np.clip(val, 0.0, 1.0)
                highpass[j, i] = val
                val = self.blend(val, 3)
                sample[j, i] = val

        for i in range(w):
            for j in range(h):
                pixel_b, pixel_g, pixel_r = img[j, i]
                pixel_b /= 255.0
                pixel_g /= 255.0
                pixel_r /= 255.0
                val_sample = sample[j, i]
                val_hp = highpass[j, i]
                val_gauss = gauss_g[j, i]
                val_gauss /= 255.0

                aa = 1.0 + math.pow(val_gauss, 0.3) * 0.09
                val_b = pixel_b * aa - val_sample * (aa - 1.0)
                val_g = pixel_g * aa - val_sample * (aa - 1.0)
                val_r = pixel_r * aa - val_sample * (aa - 1.0)
                val_b = np.clip(val_b, 0.0, 1.0)
                val_g = np.clip(val_g, 0.0, 1.0)
                val_r = np.clip(val_r, 0.0, 1.0)

                bb = math.pow(val_hp, 0.33)
                cc = math.pow(val_hp, 0.39)
                val_b = pixel_b * (1.0 - bb) + val_b * bb
                val_b = pixel_b * (1.0 - cc) + val_b * cc
                val_g = pixel_g * (1.0 - bb) + val_g * bb
                val_g = pixel_g * (1.0 - cc) + val_g * cc
                val_r = pixel_r * (1.0 - bb) + val_r * bb
                val_r = pixel_r * (1.0 - cc) + val_r * cc

                val_b = pixel_b * (1.0 - strength) + val_b * strength
                val_g = pixel_g * (1.0 - strength) + val_g * strength
                val_r = pixel_r * (1.0 - strength) + val_r * strength

                val_b = np.clip(val_b * 255.0, 0.0, 2550.)
                val_g = np.clip(val_g * 255.0, 0.0, 2550.)
                val_r = np.clip(val_r * 255.0, 0.0, 2550.)

                img[j, i] = [val_b, val_g, val_r]

        img = img.astype(np.uint8)
        return img

    def calc_distance(self, p1, p2):
        dis = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        return dis

    def enlarge_eye(self, img, lmk, strength):
        if strength < 0.01:
            return img

        fratio = strength
        h, w, c = img.shape
        img = img.astype(np.float32)
        dst = img.copy()

        centerx = int(lmk[74][0])
        centery = int(lmk[74][1])
        radius = int(self.calc_distance(lmk[74], lmk[66]) * 1.3)
        left = max(centerx - radius, 0)
        top = max(centery - radius, 0)
        right = min(centerx + radius, w)
        bottom = min(centery + radius, h)
        squareradius = radius * radius
        for x in range(left, right):
            offsetx = x - centerx
            for y in range(top, bottom):
                offsety = y - centery
                dis = offsetx * offsetx + offsety * offsety
                fscaleratio = (1.0
                               - np.sqrt(dis * 1.0 / squareradius)) * fratio

                if dis <= squareradius:
                    scalefactor = 1.0 - dis * 1.0 / squareradius
                    scalefactor = 1.0 - fscaleratio * scalefactor
                    scalefactor = np.clip(scalefactor, 0.0, 1.0)

                    x_new = (offsetx * scalefactor * 0.99) + centerx
                    y_new = (offsety * scalefactor) + centery

                    xx = math.floor(x_new)
                    yy = math.floor(y_new)

                    u = y_new - yy
                    v = x_new - xx

                    xx = np.clip(xx, 0, w - 2)
                    yy = np.clip(yy, 0, h - 2)

                    pix_00 = img[yy, xx]
                    pix_10 = img[yy, xx + 1]
                    pix_01 = img[yy + 1, xx]
                    pix_11 = img[yy + 1, xx + 1]

                    val_b = (1.0 - u) * (1.0 - v) * pix_00[0] + (
                        1.0 - u) * v * pix_10[0] + u * (
                            1.0 - v) * pix_01[0] + u * v * pix_11[0]
                    val_g = (1.0 - u) * (1.0 - v) * pix_00[1] + (
                        1.0 - u) * v * pix_10[1] + u * (
                            1.0 - v) * pix_01[1] + u * v * pix_11[1]
                    val_r = (1.0 - u) * (1.0 - v) * pix_00[2] + (
                        1.0 - u) * v * pix_10[2] + u * (
                            1.0 - v) * pix_01[2] + u * v * pix_11[2]
                    dst[y, x] = [val_b, val_g, val_r]

        centerx = int(lmk[83][0])
        centery = int(lmk[83][1])
        radius = int(self.calc_distance(lmk[83], lmk[79]) * 1.3)
        left = max(centerx - radius, 0)
        top = max(centery - radius, 0)
        right = min(centerx + radius, w)
        bottom = min(centery + radius, h)
        squareradius = radius * radius

        for x in range(left, right):
            offsetx = x - centerx
            for y in range(top, bottom):
                offsety = y - centery
                dis = offsetx * offsetx + offsety * offsety
                fscaleratio = (1.0
                               - np.sqrt(dis * 1.0 / squareradius)) * fratio

                if dis <= squareradius:
                    scalefactor = 1.0 - dis * 1.0 / squareradius
                    scalefactor = 1.0 - fscaleratio * scalefactor
                    scalefactor = np.clip(scalefactor, 0.0, 1.0)

                    x_new = (offsetx * scalefactor * 0.99) + centerx
                    y_new = (offsety * scalefactor) + centery

                    xx = math.floor(x_new)
                    yy = math.floor(y_new)

                    u = y_new - yy
                    v = x_new - xx

                    xx = np.clip(xx, 0, w - 2)
                    yy = np.clip(yy, 0, h - 2)

                    pix_00 = img[yy, xx]
                    pix_10 = img[yy, xx + 1]
                    pix_01 = img[yy + 1, xx]
                    pix_11 = img[yy + 1, xx + 1]

                    val_b = (1.0 - u) * (1.0 - v) * pix_00[0] + (
                        1.0 - u) * v * pix_10[0] + u * (
                            1.0 - v) * pix_01[0] + u * v * pix_11[0]
                    val_g = (1.0 - u) * (1.0 - v) * pix_00[1] + (
                        1.0 - u) * v * pix_10[1] + u * (
                            1.0 - v) * pix_01[1] + u * v * pix_11[1]
                    val_r = (1.0 - u) * (1.0 - v) * pix_00[2] + (
                        1.0 - u) * v * pix_10[2] + u * (
                            1.0 - v) * pix_01[2] + u * v * pix_11[2]
                    dst[y, x] = [val_b, val_g, val_r]

        dst = dst.astype(np.uint8)
        return dst

    def slim(self, img, center, dest, ratio):
        h, w, c = img.shape
        img = img.astype(np.float32)
        dst = img.copy()

        c_x = int(center[0])
        c_y = int(center[1])
        m_x = int(dest[0])
        m_y = int(dest[1])
        rad_max = max(np.abs(dest[0] - center[0]), np.abs(dest[1] - center[1]))
        msq = (m_x - c_x) * (m_x - c_x) + (m_y - c_y) * (m_y - c_y)
        ymin = np.clip(int(c_y - rad_max), 0, h)
        ymax = np.clip(int(c_y + rad_max), 0, h)
        xmin = np.clip(int(c_x - rad_max), 0, w)
        xmax = np.clip(int(c_x + rad_max), 0, w)

        val = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for i in range(ymin, ymax):
            y_offset = i - c_y
            for j in range(xmin, xmax):
                x_offset = j - c_x
                rad = np.sqrt(x_offset * x_offset + y_offset * y_offset)

                if rad <= rad_max:
                    dist = rad_max * rad_max - rad * rad
                    alpha = dist / (dist + msq)
                    alpha = alpha * alpha
                    x_new = j - ratio * alpha * (m_x - c_x)
                    y_new = i - ratio * alpha * (m_y - c_y)

                    x = math.floor(x_new)
                    y = math.floor(y_new)

                    u = y_new - y
                    v = x_new - x

                    x = np.clip(x, 0, w - 2)
                    y = np.clip(y, 0, h - 2)

                    val_00 = img[y, x]
                    val_01 = img[y, x + 1]
                    val_10 = img[y + 1, x]
                    val_11 = img[y + 1, x + 1]
                    val[0] = val_00[0]
                    val[1] = val_01[0]
                    val[2] = val_10[0]
                    val[3] = val_11[0]
                    val[4] = val_00[1]
                    val[5] = val_01[1]
                    val[6] = val_10[1]
                    val[7] = val_11[1]
                    val[8] = val_00[2]
                    val[9] = val_01[2]
                    val[10] = val_10[2]
                    val[11] = val_11[2]

                    b = (1.0 - u) * (1.0 - v) * val[0] + (1.0 - u) * v * val[
                        1] + u * (1.0 - v) * val[2] + u * v * val[3]
                    g = (1.0 - u) * (1.0 - v) * val[4] + (1.0 - u) * v * val[
                        5] + u * (1.0 - v) * val[6] + u * v * val[7]
                    r = (1.0 - u) * (1.0 - v) * val[8] + (1.0 - u) * v * val[
                        9] + u * (1.0 - v) * val[10] + u * v * val[11]

                    dst[i, j] = [b, g, r]

        dst = dst.astype(np.uint8)
        return dst

    def slim_face(self, img, lmk, strength):
        if strength < 0.01:
            return img

        degree = strength * 0.5

        h, w, c = img.shape

        center = lmk[10]
        dest = lmk[64]
        img = self.slim(img, center, dest, degree)

        center = lmk[22]
        dest = lmk[56]
        img = self.slim(img, center, dest, degree)

        return img

    def face_retouch(self, img, kpts):
        img = self.smooth(img, self.strength_smooth)

        img = self.filter(img, self.filter_map_white, self.strength_white)

        strength = 1.0
        img = self.filter(img, self.filter_map, strength)

        for kpt in kpts:
            img = self.enlarge_eye(img, kpt, self.strength_enlarge_eye)
            img = self.slim_face(img, kpt, self.strength_slim_face)

        return img
