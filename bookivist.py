#!/usr/bin/env python3


"""Bookivist (alpha)
Copyright (C) 2017 Jacob VanDrunen

Usage:
  bookivist.py [options] <glob>

Options:
  -h, --help                    Show this message.
  -o <file>, --output <file>    Specify the output file [default: ./output.pdf]
  -q, --quiet                   Suppress stdout.
  -r <dpi>, --resolution <dpi>  Specify the output resolution [default: 300]
  -v, --version                 Show version.
"""

__version__ = (0,0,0)


from contextlib import contextmanager
import os
import sys
import shutil
import glob
import uuid
import img2pdf
import cv2
from PIL import Image
import numpy as np


def resize(image, ratio):
    return cv2.resize(image, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_CUBIC)


def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged


def auto_threshold(image, window=63):
    if window % 2 == 0:
        window += 1

    median = np.median(image)
    mean = np.mean(image)
    diff = abs(median - mean)

    thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, window, diff)
    return thresh


def row_threshold(image):
    result = np.zeros(image.shape[0])
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            result[i] += 1.0 if image[i][j] else 0.0
        result[i] = result[i] / float(image.shape[0])
    mean = np.mean(result)
    return np.array(list(map(lambda n: 1 if n > mean else 0, result)))


def col_threshold(image):
    result = np.zeros(image.shape[1])
    for i in range(image.shape[1]):
        for j in range(image.shape[0]):
            result[i] += 1.0 if image[j][i] else 0.0
        result[i] = result[i] / float(image.shape[0])
    mean = np.mean(result)
    return np.array(list(map(lambda n: 1 if n > mean else 0, result)))


def reconstruct(rb, cb):
    cols = len(rb)
    rows = len(cb)
    result = np.zeros((cols, rows), np.uint8)
    for y in range(cols):
        for x in range(rows):
            result[y][x] = (rb[y] & cb[x]) * 255
    return result


def blockify(image):
    edged = auto_canny(image)
    kernel = np.ones((5,5), np.uint8)
    dil = cv2.dilate(edged, kernel, iterations=1)
    rb = row_threshold(dil)
    cb = col_threshold(dil)
    result = reconstruct(rb, cb)
    return cv2.dilate(result, kernel, iterations=1)


def scan(file_name):
    img = cv2.imread(file_name)
    height, width, channels = img.shape

    ratio = 500.0 / img.shape[0]
    new_img = resize(img, ratio)
     
    edged = auto_canny(new_img)
    image, contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    page_contour = sorted(contours, key=lambda c: cv2.boundingRect(c)[2] * cv2.boundingRect(c)[3])[-1]
    x, y, w, h = [int(i / ratio) for i in cv2.boundingRect(page_contour)]

    if w * h < (width * height) * 0.67:
        page_crop = img
    else:
        page_crop = img[y:y+h,x:x+w]
        height, width, channels = page_crop.shape
    
    gray = cv2.cvtColor(page_crop, cv2.COLOR_BGR2GRAY)
    scanned = auto_threshold(gray, int(width / 32.0) if height > width else int(height / 32.0))
    return scanned


def normalize(image, resolution):
    # Fit the image to an 8.5x11 page
    height, width = image.shape

    # Portrait
    normal_w = int(8.5 * resolution)
    normal_h = int(11 * resolution)
    # Landscape
    if height < width:
        normal_w, normal_h = normal_h, normal_w

    # Scale width to max first, if height is too big, scale height
    ratio = normal_w / float(width)
    padding = 'y'  # Padding needed at top and bottom
    if (height * ratio) > normal_h:
        ratio = normal_h / float(height)
        padding = 'x'  # Padding needed at left and right
    
    # Resize image and calculate needed offsets
    scaled = resize(image, ratio)
    height, width = scaled.shape
    if padding == 'y':
        x_offset = 0
        y_offset = int((normal_h - height) / 2)
    else:
        x_offset = int((normal_w - width) / 2)
        y_offset = 0

    # Create the finished product with padding
    page = np.zeros((normal_h, normal_w), np.uint8)
    page.fill(255)
    page[y_offset:y_offset+height, x_offset:x_offset+width] = scaled
    return (normal_w, normal_h), page


def save_pdf(imgs, output_file, pdf_layout):
    with open(output_file, 'wb') as f:
        f.write(img2pdf.convert(*imgs, layout_fun=lambda w,h,r: pdf_layout))


def mod_glob(path):
    # Helper function to order the photos based on the time taken
    files = map(lambda f: (f, Image.open(f)._getexif()[36867]), glob.glob(path))
    return map(lambda t: t[0], sorted(files, key=lambda u: u[1]))


@contextmanager
def context_dir(dir_path):
    os.makedirs(dir_path)
    yield
    shutil.rmtree(dir_path)


def scan_all(glob_path, output_file, resolution):
    print('Scanning...')
    tmp_path = os.path.join('/', 'tmp', str(uuid.uuid1()))
    with context_dir(tmp_path):
        imgs = []
        for i, input_path in enumerate(mod_glob(glob_path), 1):
            output_path = os.path.join(tmp_path, '{}.png'.format(str(i).zfill(6)))
            dimensions, img = normalize(scan(input_path), resolution)
            cv2.imwrite(output_path, img)
            imgs.append(output_path)
            if i % 5 == 0:
                print('  {} pages done'.format(i))
        print('Finished scanning {} pages.'.format(i))
        print('Saving to PDF...')
        # Create PDF
        save_pdf(imgs, output_file, (dimensions[0], dimensions[1], dimensions[0], dimensions[1]))
        print('Done.')


if __name__ == '__main__':
    from docopt import docopt
    args = docopt(__doc__, version='.'.join(map(str, __version__)))
    if args['--quiet']:
        from io import StringIO
        sys.stdout = StringIO()
    scan_all(args['<glob>'], args['--output'], int(args['--resolution']))
