import cv2 as cv
import math
import numpy as np
import os

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

''''
imag1 = cv.imread("E:/DarkPair/Zerodce/low00001.png")
print (imag1.shape)
imag2 = cv.imread("E:/DarkPair/Normal/normal00001.png")
imag3 = cv.imread("E:/DarkPair/Low/low00001.png")
print(imag2.shape)
# imag2 = imag2.reshape(352,352,3)
print(imag3.shape)
res1 = psnr1(imag1,imag2)
print("res1:",res1)
res2 = psnr2(imag1,imag2)
print("res2:",res2)

res1 = psnr1(imag3,imag2)
print("res3:",res1)
res2 = psnr2(imag3,imag2)
print("res4:",res2)
'''
file_handle = open('E:/psnrdata.txt', mode='w')
file_handle.writelines("img_number: \tgt and low\tgt and zerodce\n")
psnrgl=0
psnrgz=0
'''
for i in range(1,790):
    if i<10:
        numstl="0000"+str(i)
    if i>9 and i<100:
        numstl="000"+str(i)
    if i>99:
        numstl="00"+str(i)
'''
i=0
filePath = "E:/testpsnrimg/lowlight"

file_list = os.listdir(filePath)
# print("E:/DarkPair/low"+numstl+".png")
for file_name in file_list:
    num1=file_name.find("low")
    num2=file_name.find(".png")
    strnum=file_name[num1+3:num2]
    print(strnum)

    lowimg=cv.imread("E:/testpsnrimg/lowlight/low"+strnum+".png")
    normalimg=cv.imread("E:/testpsnrimg//Normal/normal"+strnum+".png")
    zerodceimg=cv.imread("E:/testpsnrimg/tmp/normal"+strnum+".png")
    i+=1
    res1 = psnr2(lowimg, normalimg)
    psnrgl +=res1
    #print("res1:", res1)
    res2 = psnr2(zerodceimg,normalimg)
    psnrgz +=res2
    #print("res2:",res2)
    file_handle.writelines(str(i)+"\t%.5f"%res1+"\t%.5f\n"%res2)

psnrgl/=i
psnrgz/=i
print("psnr1:",psnrgl)
print("psnr2:",psnrgz)

file_handle.close()