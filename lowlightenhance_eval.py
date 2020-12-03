import cv2
import numpy as np
import copy
import random
import math
from PIL import Image
from PIL import ImageStat
import os

def ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

def brightness1( im_file ):#转换图像到灰度，返回平均像素亮度
   im = Image.open(im_file).convert('L')
   stat = ImageStat.Stat(im)
   return stat.mean[0]

def brightness2( im_file ):#转换图像到灰度，返回RMS像素亮度
   im = Image.open(im_file).convert('L')
   stat = ImageStat.Stat(im)
   return stat.rms[0]

def brightness3( im_file ):#平均像素，然后转换为“可感知的亮度”
   im = Image.open(im_file)
   stat = ImageStat.Stat(im)
   r,g,b = stat.mean
   return math.sqrt(0.241*(r**2) + 0.691*(g**2) + 0.068*(b**2))

def brightness4( im_file ):#像素的均方根，然后转换为“感知亮度”
   im = Image.open(im_file)
   stat = ImageStat.Stat(im)
   r,g,b = stat.rms
   return math.sqrt(0.241*(r**2) + 0.691*(g**2) + 0.068*(b**2))

def brightness5( im_file ):#计算像素的“感知亮度”，然后返回平均值
   im = Image.open(im_file)
   stat = ImageStat.Stat(im)
   gs = (math.sqrt(0.241*(r**2) + 0.691*(g**2) + 0.068*(b**2))
         for r,g,b in im.getdata())
   return sum(gs)/stat.count[0]


def psnr1(img1,img2):
    #compute mse
    # mse = np.mean((img1-img2)**2)
    mse = np.mean((img1/1.0-img2/1.0)**2)
    #compute psnr
    if mse < 1e-10:
        return 100
    psnr1 = 20*math.log10(255/math.sqrt(mse))
    return psnr1

def psnr2(img1,img2):
    mse = np.mean((img1/255.0-img2/255.0)**2)
    if mse < 1e-10:
        return 100
    psnr2 = 20*math.log10(1/math.sqrt(mse))
    return psnr2

def MAE(img1,img2):
    mae = np.mean(np.abs(img1 - img2))
    return mae



if __name__ == '__main__':
    filePath = "E:/testpsnrimg/lowlight"

    file_list = os.listdir(filePath)
    # print("E:/DarkPair/low"+numstl+".png")
    imgnumber=0
    ss=0
    sslow=0
    sstmp=0
    brightness_low=0
    brightness_nor=0
    brightness_tmp=0
    psnrlow=0
    psnrtmp=0
    maelow=0
    maetmp=0
    for file_name in file_list:
        imgnumber+=1
        num1=file_name.find("low")
        num2=file_name.find(".png")
        strnum=file_name[num1+3:num2]
       #print(strnum)

        lowimg="E:/testpsnrimg/lowlight/low"+strnum+".png"
        normalimg="E:/testpsnrimg/normal/normal"+strnum+".png"
        zerodceimg="E:/testpsnrimg/tmp/normal"+strnum+".png"
        #print(brightness1(lowimg))
        #print(strnum+" : low "+str(brightness1(lowimg))+"\tnor "+str(brightness1(normalimg))+"\ttmp "+str(brightness1(zerodceimg)))
        img1 = cv2.imread(lowimg, 0)
        img2 = cv2.imread(normalimg, 0)
        img3 = cv2.imread(zerodceimg, 0)
        sslow+=calculate_ssim(img1,img2)
        sstmp+=calculate_ssim(img3,img2)

        brightness_low += brightness1(lowimg)
        brightness_nor += brightness1(normalimg)
        brightness_tmp += brightness1(zerodceimg)

        psnrlow+=psnr1(img1,img2)
        psnrtmp+=psnr1(img3,img2)

        maelow+=MAE(img1,img2)
        maetmp+=MAE(img3,img2)
    #print(brightness1("E:/testpsnrimg/lowlight/low00005.png"))


    brightness_low /= imgnumber
    brightness_nor /= imgnumber
    brightness_tmp /= imgnumber

    sslow /= imgnumber
    sstmp /= imgnumber


    psnrlow /= imgnumber
    psnrtmp /= imgnumber

    maelow /= imgnumber
    maetmp /= imgnumber

    print("psnr gt and low: "+str(psnrlow))
    print("psnr gt and tmp: " + str(psnrtmp))

    print("ssim gt and low: " + str(sslow))
    print("ssim gt and tmp: "+str(sstmp))

    print("mae gt and low: " + str(maelow))
    print("mae gt and tmp: " + str(maetmp))

    print("avg brightness low: " + str(brightness_low))
    print("avg brightness low: " + str(brightness_nor))
    print("avg brightness low: " + str(brightness_tmp))