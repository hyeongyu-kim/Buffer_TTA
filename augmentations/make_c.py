import os
from PIL import Image
import os.path
import time
import torch
import torchvision.datasets as dset
import torchvision.transforms as trn
import torch.utils.data as data
import numpy as np

from PIL import Image


# /////////////// Distortion Helpers ///////////////

import skimage as sk
from skimage.filters import gaussian
from io import BytesIO
# from wand.image import Image as WandImage               <--- REMOVED
# from wand.api import library as wandlibrary         <--- REMOVED
# import wand.color as WandColor                      <--- REMOVED
# import ctypes                                       <--- REMOVED
from PIL import Image as PILImage
import cv2
from scipy.ndimage import zoom as scizoom
from scipy.ndimage.interpolation import map_coordinates
import warnings

warnings.simplefilter("ignore", UserWarning)


def disk(radius, alias_blur=0.1, dtype=np.float32):
    if radius <= 8:
        L = np.arange(-8, 8 + 1)
        ksize = (3, 3)
    else:
        L = np.arange(-radius, radius + 1)
        ksize = (5, 5)
    X, Y = np.meshgrid(L, L)
    aliased_disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
    aliased_disk /= np.sum(aliased_disk)

    # supersample disk to antialias
    return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)


# --- (REMOVED Wand/ctypes helper code) ---


# +++ START: NEW HELPER FUNCTION (OpenCV Motion Blur Kernel) +++
def _create_motion_blur_kernel(length, sigma, angle):
    """
    wand의 motion_blur를 근사하는 OpenCV 모션 블러 커널을 생성합니다.
    'length' (wand의 radius) = 블러의 길이
    'sigma' (wand의 sigma) = 블러의 두께/퍼짐
    'angle' = 블러의 각도
    """
    
    # 1. 회전된 라인을 담을 수 있을 만큼 충분히 큰 커널 크기 설정
    ksize = int(length * 1.5 + sigma * 2 + 2) # 휴리스틱
    if ksize % 2 == 0:
        ksize += 1
    center = ksize // 2

    # 2. 수평 1D 라인 커널 생성
    line_k = np.zeros((ksize, ksize), dtype=np.float32)
    line_length = int(length)
    if line_length < 1:
        line_length = 1
    start = center - line_length // 2
    end = start + line_length
    line_k[center, start:end] = 1.
    
    # 3. 라인 커널 회전
    M = cv2.getRotationMatrix2D((center, center), angle, 1)
    rotated_k = cv2.warpAffine(line_k, M, (ksize, ksize))
    
    # 4. sigma > 0 이면 회전된 라인 커널을 가우시안 블러 처리 (두께 생성)
    if sigma > 0:
        blur_ksize = int(sigma * 4 + 1)
        if blur_ksize % 2 == 0:
            blur_ksize += 1
        rotated_k = cv2.GaussianBlur(rotated_k, (blur_ksize, blur_ksize), sigma)
    
    # 5. 커널 정규화
    rotated_k /= np.sum(rotated_k)
    return rotated_k
# +++ END: NEW HELPER FUNCTION +++


# modification of https://github.com/FLHerne/mapgen/blob/master/diamondsquare.py
def plasma_fractal(mapsize=32, wibbledecay=3):
    """
    Generate a heightmap using diamond-square algorithm.
    Return square 2d array, side length 'mapsize', of floats in range 0-255.
    'mapsize' must be a power of two.
    """
    assert (mapsize & (mapsize - 1) == 0)
    maparray = np.empty((mapsize, mapsize), dtype=np.float_)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100

    def wibbledmean(array):
        return array / 4 + wibble * np.random.uniform(-wibble, wibble, array.shape)

    def fillsquares():
        """For each square of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
        squareaccum += np.roll(squareaccum, shift=-1, axis=1)
        maparray[stepsize // 2:mapsize:stepsize,
        stepsize // 2:mapsize:stepsize] = wibbledmean(squareaccum)

    def filldiamonds():
        """For each diamond of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        mapsize = maparray.shape[0]
        drgrid = maparray[stepsize // 2:mapsize:stepsize, stepsize // 2:mapsize:stepsize]
        ulgrid = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
        ltsum = ldrsum + lulsum
        maparray[0:mapsize:stepsize, stepsize // 2:mapsize:stepsize] = wibbledmean(ltsum)
        tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
        ttsum = tdrsum + tulsum
        maparray[stepsize // 2:mapsize:stepsize, 0:mapsize:stepsize] = wibbledmean(ttsum)

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay

    maparray -= maparray.min()
    return maparray / maparray.max()


def clipped_zoom(img, zoom_factor):
    h = img.shape[0]
    # ceil crop height(= crop width)
    ch = int(np.ceil(h / zoom_factor))

    top = (h - ch) // 2
    img = scizoom(img[top:top + ch, top:top + ch], (zoom_factor, zoom_factor, 1), order=1)
    # trim off any extra pixels
    trim_top = (img.shape[0] - h) // 2

    return img[trim_top:trim_top + h, trim_top:trim_top + h]


# /////////////// End Distortion Helpers ///////////////


# /////////////// Distortions ///////////////

def gaussian_noise(x, severity=1):
    c = [0.04, 0.06, .08, .09, .10][severity - 1]

    x = np.array(x) / 255.
    return np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255


def shot_noise(x, severity=1):
    c = [500, 250, 100, 75, 50][severity - 1]

    x = np.array(x) / 255.
    return np.clip(np.random.poisson(x * c) / c, 0, 1) * 255


def impulse_noise(x, severity=1):
    c = [.01, .02, .03, .05, .07][severity - 1]

    x = sk.util.random_noise(np.array(x) / 255., mode='s&p', amount=c)
    return np.clip(x, 0, 1) * 255


def speckle_noise(x, severity=1):
    c = [.06, .1, .12, .16, .2][severity - 1]

    x = np.array(x) / 255.
    return np.clip(x + x * np.random.normal(size=x.shape, scale=c), 0, 1) * 255


def gaussian_blur(x, severity=1):
    c = [.4, .6, 0.7, .8, 1][severity - 1]

    x = gaussian(np.array(x) / 255., sigma=c, channel_axis=-1)
    return np.clip(x, 0, 1) * 255


def glass_blur(x, severity=1):
    # sigma, max_delta, iterations
    c = [(0.05,1,1), (0.25,1,1), (0.4,1,1), (0.25,1,2), (0.4,1,2)][severity - 1]

    x = np.uint8(gaussian(np.array(x) / 255., sigma=c[0], channel_axis=-1) * 255)

    # locally shuffle pixels
    for i in range(c[2]):
        for h in range(32 - c[1], c[1], -1):
            for w in range(32 - c[1], c[1], -1):
                dx, dy = np.random.randint(-c[1], c[1], size=(2,))
                h_prime, w_prime = h + dy, w + dx
                # swap
                x[h, w], x[h_prime, w_prime] = x[h_prime, w_prime], x[h, w]

    return np.clip(gaussian(x / 255., sigma=c[0], channel_axis=-1), 0, 1) * 255


def defocus_blur(x, severity=1):
    c = [(0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (1, 0.2), (1.5, 0.1)][severity - 1]

    x = np.array(x) / 255.
    kernel = disk(radius=c[0], alias_blur=c[1])

    channels = []
    for d in range(3):
        channels.append(cv2.filter2D(x[:, :, d], -1, kernel))
    channels = np.array(channels).transpose((1, 2, 0))  # 3x32x32 -> 32x32x3

    return np.clip(channels, 0, 1) * 255


# +++ START: MODIFIED motion_blur function +++
def motion_blur(x, severity=1):
    c = [(6,1), (6,1.5), (6,2), (8,2), (9,2.5)][severity - 1]

    # 1. PIL 이미지를 Numpy 배열로 변환
    x_np = np.array(x)
    
    # 2. 새로운 헬퍼 함수를 사용해 모션 블러 커널 생성
    angle = np.random.uniform(-45, 45)
    kernel = _create_motion_blur_kernel(c[0], c[1], angle) # c[0]=length, c[1]=sigma
    
    # 3. cv2.filter2D를 사용해 블러 적용
    # 원본 코드는 그레이스케일 입력을 RGB로 변환하는 로직을 포함했음
    if x_np.ndim == 2: # 그레이스케일 (H, W)
        blurred = cv2.filter2D(x_np, -1, kernel)
        # 3채널 RGB로 다시 스택
        blurred_rgb = np.stack([blurred] * 3, axis=-1)
    else: # RGB (H, W, C)
        blurred_rgb = cv2.filter2D(x_np, -1, kernel)
    
    # 4. 0-255 범위로 클리핑
    return np.clip(blurred_rgb, 0, 255)
# +++ END: MODIFIED motion_blur function +++


def zoom_blur(x, severity=1):
    c = [np.arange(1, 1.06, 0.01), np.arange(1, 1.11, 0.01), np.arange(1, 1.16, 0.01),
         np.arange(1, 1.21, 0.01), np.arange(1, 1.26, 0.01)][severity - 1]

    x = (np.array(x) / 255.).astype(np.float32)
    out = np.zeros_like(x)
    for zoom_factor in c:
        out += clipped_zoom(x, zoom_factor)

    x = (x + out) / (len(c) + 1)
    return np.clip(x, 0, 1) * 255


def fog(x, severity=1):
    c = [(.2,3), (.5,3), (0.75,2.5), (1,2), (1.5,1.75)][severity - 1]

    x = np.array(x) / 255.
    max_val = x.max()
    x += c[0] * plasma_fractal(wibbledecay=c[1])[:32, :32][..., np.newaxis]
    return np.clip(x * max_val / (max_val + c[0]), 0, 1) * 255


def frost(x, severity=1):
    c = [(1, 0.2), (1, 0.3), (0.9, 0.4), (0.85, 0.4), (0.75, 0.45)][severity - 1]
    idx = np.random.randint(5)
    # './frost1.png'와 같은 서리 이미지 파일이 로컬에 필요합니다.
    # 이 파일들이 없다면 이 함수는 에러를 발생시킵니다.
    # 원본 코드의 의존성이므로 여기서는 수정하지 않습니다.
    filename = ['./frost1.png', './frost2.png', './frost3.png', './frost4.jpg', './frost5.jpg', './frost6.jpg'][idx]
    frost = cv2.imread(filename)
    frost = cv2.resize(frost, (0, 0), fx=0.2, fy=0.2)
    # randomly crop and convert to rgb
    x_start, y_start = np.random.randint(0, frost.shape[0] - 32), np.random.randint(0, frost.shape[1] - 32)
    frost = frost[x_start:x_start + 32, y_start:y_start + 32][..., [2, 1, 0]]

    return np.clip(c[0] * np.array(x) + c[1] * frost, 0, 255)


# +++ START: MODIFIED snow function +++
def snow(x, severity=1):
    c = [(0.1,0.2,1,0.6,8,3,0.95),
         (0.1,0.2,1,0.5,10,4,0.9),
         (0.15,0.3,1.75,0.55,10,4,0.9),
         (0.25,0.3,2.25,0.6,12,6,0.85),
         (0.3,0.3,1.25,0.65,14,12,0.8)][severity - 1]

    x = np.array(x, dtype=np.float32) / 255.
    snow_layer = np.random.normal(size=x.shape[:2], loc=c[0], scale=c[1])  # [:2] for monochrome

    snow_layer = clipped_zoom(snow_layer[..., np.newaxis], c[2])
    snow_layer[snow_layer < c[3]] = 0

    # --- wand/MotionImage 대신 OpenCV 사용 ---
    # 1. [0, 1] 범위의 눈 레이어를 [0, 255] uint8 (그레이스케일)로 변환
    snow_layer_uint8 = (np.clip(snow_layer.squeeze(), 0, 1) * 255).astype(np.uint8)
    
    # 2. 모션 블러 커널 생성
    angle = np.random.uniform(-135, -45)
    kernel = _create_motion_blur_kernel(c[4], c[5], angle) # c[4]=length, c[5]=sigma
    
    # 3. cv2.filter2D로 블러 적용
    snow_layer_blurred = cv2.filter2D(snow_layer_uint8, -1, kernel)
    
    # 4. 다시 [0, 1] 범위의 float 타입으로 정규화
    snow_layer = np.clip(snow_layer_blurred, 0, 255).astype(np.float32) / 255.
    # --- (수정 완료) ---

    snow_layer = snow_layer[..., np.newaxis]

    x = c[6] * x + (1 - c[6]) * np.maximum(x, cv2.cvtColor(x, cv2.COLOR_RGB2GRAY).reshape(32, 32, 1) * 1.5 + 0.5)
    return np.clip(x + snow_layer + np.rot90(snow_layer, k=2), 0, 1) * 255
# +++ END: MODIFIED snow function +++


def spatter(x, severity=1):
    c = [(0.62,0.1,0.7,0.7,0.5,0),
         (0.65,0.1,0.8,0.7,0.5,0),
         (0.65,0.3,1,0.69,0.5,0),
         (0.65,0.1,0.7,0.69,0.6,1),
         (0.65,0.1,0.5,0.68,0.6,1)][severity - 1]
    x = np.array(x, dtype=np.float32) / 255.

    liquid_layer = np.random.normal(size=x.shape[:2], loc=c[0], scale=c[1])

    liquid_layer = gaussian(liquid_layer, sigma=c[2])
    liquid_layer[liquid_layer < c[3]] = 0
    if c[5] == 0:
        liquid_layer = (liquid_layer * 255).astype(np.uint8)
        dist = 255 - cv2.Canny(liquid_layer, 50, 150)
        dist = cv2.distanceTransform(dist, cv2.DIST_L2, 5)
        _, dist = cv2.threshold(dist, 20, 20, cv2.THRESH_TRUNC)
        dist = cv2.blur(dist, (3, 3)).astype(np.uint8)
        dist = cv2.equalizeHist(dist)
        #     ker = np.array([[-1,-2,-3],[-2,0,0],[-3,0,1]], dtype=np.float32)
        #     ker -= np.mean(ker)
        ker = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
        dist = cv2.filter2D(dist, cv2.CV_8U, ker)
        dist = cv2.blur(dist, (3, 3)).astype(np.float32)

        m = cv2.cvtColor(liquid_layer * dist, cv2.COLOR_GRAY2BGRA)
        m /= np.max(m, axis=(0, 1))
        m *= c[4]

        # water is pale turqouise
        color = np.concatenate((175 / 255. * np.ones_like(m[..., :1]),
                                238 / 255. * np.ones_like(m[..., :1]),
                                238 / 255. * np.ones_like(m[..., :1])), axis=2)

        color = cv2.cvtColor(color, cv2.COLOR_BGR2BGRA)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2BGRA)

        return cv2.cvtColor(np.clip(x + m * color, 0, 1), cv2.COLOR_BGRA2BGR) * 255
    else:
        m = np.where(liquid_layer > c[3], 1, 0)
        m = gaussian(m.astype(np.float32), sigma=c[4])
        m[m < 0.8] = 0
        #         m = np.abs(m) ** (1/c[4])

        # mud brown
        color = np.concatenate((63 / 255. * np.ones_like(x[..., :1]),
                                42 / 255. * np.ones_like(x[..., :1]),
                                20 / 255. * np.ones_like(x[..., :1])), axis=2)

        color *= m[..., np.newaxis]
        x *= (1 - m[..., np.newaxis])

        return np.clip(x + color, 0, 1) * 255


def contrast(x, severity=1):
    c = [.75, .5, .4, .3, 0.15][severity - 1]

    x = np.array(x) / 255.
    means = np.mean(x, axis=(0, 1), keepdims=True)
    return np.clip((x - means) * c + means, 0, 1) * 255


def brightness(x, severity=1):
    c = [.05, .1, .15, .2, .3][severity - 1]

    x = np.array(x) / 255.
    x = sk.color.rgb2hsv(x)
    x[:, :, 2] = np.clip(x[:, :, 2] + c, 0, 1)
    x = sk.color.hsv2rgb(x)

    return np.clip(x, 0, 1) * 255


def saturate(x, severity=1):
    c = [(0.3, 0), (0.1, 0), (1.5, 0), (2, 0.1), (2.5, 0.2)][severity - 1]

    x = np.array(x) / 255.
    x = sk.color.rgb2hsv(x)
    x[:, :, 1] = np.clip(x[:, :, 1] * c[0] + c[1], 0, 1)
    x = sk.color.hsv2rgb(x)

    return np.clip(x, 0, 1) * 255


def jpeg_compression(x, severity=1):
    c = [80, 65, 58, 50, 40][severity - 1]

    output = BytesIO()
    x.save(output, 'JPEG', quality=c)
    x = PILImage.open(output)

    return x


def pixelate(x, severity=1):
    c = [0.95, 0.9, 0.85, 0.75, 0.65][severity - 1]

    x = x.resize((int(32 * c), int(32 * c)), PILImage.BOX)
    x = x.resize((32, 32), PILImage.BOX)

    return x


# mod of https://gist.github.com/erniejunior/601cdf56d2b424757de5
def elastic_transform(image, severity=1):
    IMSIZE = 32
    c = [(IMSIZE*0, IMSIZE*0, IMSIZE*0.08),
         (IMSIZE*0.05, IMSIZE*0.2, IMSIZE*0.07),
         (IMSIZE*0.08, IMSIZE*0.06, IMSIZE*0.06),
         (IMSIZE*0.1, IMSIZE*0.04, IMSIZE*0.05),
         (IMSIZE*0.1, IMSIZE*0.03, IMSIZE*0.03)][severity - 1]

    image = np.array(image, dtype=np.float32) / 255.
    shape = image.shape
    shape_size = shape[:2]

    # random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size,
                       [center_square[0] + square_size, center_square[1] - square_size],
                       center_square - square_size])
    pts2 = pts1 + np.random.uniform(-c[2], c[2], size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = (gaussian(np.random.uniform(-1, 1, size=shape[:2]),
                   c[1], mode='reflect', truncate=3) * c[0]).astype(np.float32)
    dy = (gaussian(np.random.uniform(-1, 1, size=shape[:2]),
                   c[1], mode='reflect', truncate=3) * c[0]).astype(np.float32)
    dx, dy = dx[..., np.newaxis], dy[..., np.newaxis]

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))
    return np.clip(map_coordinates(image, indices, order=1, mode='reflect').reshape(shape), 0, 1) * 255


# /////////////// make total_corrupt class ///////////////

# --- (파일 상단에 import 문 추가) ---
import torch
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image
import warnings
# --- (기존 import 외에 위 4가지가 필요합니다) ---


# ( ... gaussian_noise, shot_noise 등 모든 손상 함수 정의 ... )


# /////////////// make total_corrupt class (수정된 버전) ///////////////

class CifarCcorruptions:
    """
    제공된 스크립트의 모든 손상 함수를 래핑(wrapping)하는 헬퍼 클래스입니다.
    
    (B, C, H, W) 텐서 배치를 입력받아 손상을 적용하고,
    (B, C, H, W) 텐서 배치로 반환합니다.
    """
    
    def __init__(self):
        """
        스크립트에 정의된 모든 손상 함수의 이름을 실제 함수와 매핑합니다.
        """
        self.corruption_dict = {
            'gaussian_noise': gaussian_noise,
            'shot_noise': shot_noise,
            'impulse_noise': impulse_noise,
            'speckle_noise': speckle_noise,
            'gaussian_blur': gaussian_blur,
            'glass_blur': glass_blur,
            'defocus_blur': defocus_blur,
            'motion_blur': motion_blur,   # <--- 수정된 함수가 사용됩니다
            'zoom_blur': zoom_blur,
            'fog': fog,
            'frost': frost,
            'snow': snow,                 # <--- 수정된 함수가 사용됩니다
            'spatter': spatter,
            'contrast': contrast,
            'brightness': brightness,
            'saturate': saturate,
            'jpeg_compression': jpeg_compression,
            'pixelate': pixelate,
            'elastic_transform': elastic_transform
        }
        
        # 참조를 위해 손상 이름 리스트도 저장합니다.
        self.corruption_names = list(self.corruption_dict.keys())

    def _apply_corruption_to_pil(self, x_pil: Image.Image, 
                                 corruption_name: str, 
                                 severity: int) -> Image.Image:
        """
        [Helper] PIL 이미지에 손상을 적용하고 PIL 이미지로 반환합니다.
        (Numpy 배열을 반환하는 함수들을 PIL로 통일)
        """
        corruption_func = self.corruption_dict[corruption_name]
        corrupted_x = corruption_func(x_pil, severity)
        
        # 반환 타입을 PIL.Image로 통일
        if isinstance(corrupted_x, np.ndarray):
            # float 타입인 경우 0-255로 클리핑하고 uint8로 변환
            if corrupted_x.dtype == np.float32 or corrupted_x.dtype == np.float64:
                corrupted_x = np.clip(corrupted_x, 0, 255)
            
            return Image.fromarray(corrupted_x.astype(np.uint8))
        
        elif isinstance(corrupted_x, Image.Image):
            # 이미 PIL 이미지인 경우
            return corrupted_x
        
        else:
            raise TypeError(f"손상 함수 {corruption_name}가 예상치 못한 타입을 반환: {type(corrupted_x)}")


    def __call__(self, x_batch_tensor: torch.Tensor, 
                 corruption_name: str, 
                 severity: int = 1) -> torch.Tensor: # 기본 severity를 5에서 1로 수정 (일반적)
        """
        주어진 (B, C, H, W) 텐서 배치에 특정 손상을 적용합니다.

        Args:
            x_batch_tensor (torch.Tensor): 입력 텐서 배치 (B, C, H, W).
                                           [0, 1] 범위로 정규화되어 있다고 가정합니다.
            corruption_name (str): 적용할 손상의 이름 (e.g., 'gaussian_noise').
            severity (int, optional): 손상의 심각도 (1-5). 기본값은 1.

        Returns:
            torch.Tensor: 손상이 적용된 (B, C, H, W) 텐서 배치.
        """
        
        if corruption_name not in self.corruption_dict:
            raise ValueError(f"알 수 없는 손상 이름입니다: {corruption_name}")
        
        if not (1 <= severity <= 5):
            warnings.warn(f"심각도 {severity}는 표준 범위[1, 5]를 벗어났습니다.")

        # 결과를 저장할 리스트
        corrupted_list = []
        
        # 원본 텐서의 디바이스 정보 저장
        original_device = x_batch_tensor.device
        
        # 배치(B)를 순회
        for i in range(x_batch_tensor.shape[0]):
            
            # 1. (C, H, W) 텐서 추출 (변환을 위해 CPU로 이동)
            x_tensor_single = x_batch_tensor[i].cpu()
            
            # 2. 텐서 -> PIL 이미지로 변환
            #    TF.to_pil_image는 [0, 1] 범위의 (C, H, W) 텐서를 입력받음
            pil_image = TF.to_pil_image(x_tensor_single)
            
            # 3. PIL 이미지에 손상 적용 (내부 헬퍼 함수 사용)
            corrupted_pil = self._apply_corruption_to_pil(
                pil_image, corruption_name, severity
            )
            
            # 4. 손상된 PIL 이미지 -> 텐서로 변환
            #    TF.to_tensor는 PIL 이미지를 [0, 1] 범위의 (C, H, W) 텐서로 변환
            corrupted_tensor_single = TF.to_tensor(corrupted_pil)
            
            # 5. 리스트에 추가
            corrupted_list.append(corrupted_tensor_single)

        # 6. 텐서 리스트를 (B, C, H, W) 배치 텐서로 스택
        #    .to(original_device)로 원본 텐서와 동일한 디바이스로 이동
        return torch.stack(corrupted_list, dim=0).to(original_device)