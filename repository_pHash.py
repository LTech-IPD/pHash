from cmath import cos, pi
from math import sqrt
import numpy as np
import cv2
PADDING=0
RESIZE=1
class pHashService():
    def __init__(self):
        pass
    def reshape(self,img,regionSize):
        width=img.shape[0]
        height=img.shape[1]
        (maxSize,minSize)=(width,height) if width>height else (height,width)
        imgSize=maxSize + regionSize - maxSize % regionSize if not maxSize % regionSize ==0 else maxSize
        maxPadding0=(imgSize-maxSize)//2
        maxPadding1=imgSize-maxSize-maxPadding0
        minPadding0=(imgSize-minSize)//2
        minPadding1=imgSize-minSize-minPadding0
        if width>height:
            img=cv2.copyMakeBorder(img,maxPadding0,maxPadding1,minPadding0,minPadding1,cv2.BORDER_CONSTANT,value=0)
        else:
            img=cv2.copyMakeBorder(img,minPadding0,minPadding1,maxPadding0,maxPadding1,cv2.BORDER_CONSTANT,value=0)
        return img
    def regionDCT(self,region,length,reserveLength):
        matrix=np.zeros([length,length],dtype=np.float32)
        matrix=np.mat(matrix)
        region=np.mat(region)
        for u in range(length):
            for v in range(length):
                if u==0:
                    c=sqrt(1/length)
                else:
                    c=sqrt(2/length)
                matrix[u,v]=(c*cos((v+0.5)*pi/length*u)).real
        results=matrix*region*matrix.T
        results=results[:reserveLength,:reserveLength]
        results=np.array(results)
        return results
    def __calc_DCT(self,img,regionSize,reserveLength):
        results=np.zeros([int(img.shape[0]*(reserveLength/regionSize)),int(img.shape[0]*(reserveLength/regionSize))],dtype=np.float32)
        for i in range(img.shape[0]//regionSize):
            for j in range(img.shape[0]//regionSize):
                results[reserveLength*i:reserveLength*i+reserveLength,
                        reserveLength*j:reserveLength*j+reserveLength]=self.regionDCT(img[regionSize*i:regionSize*i+regionSize,
                                                                                 regionSize*j:regionSize*j+regionSize],
                                                                                 length=regionSize,reserveLength=reserveLength)
        return results
    def pHash(self,img0:np.ndarray,img1:np.ndarray,channel:int,imgProcessStrategy:int,size0:int,size1:int):
        """
        `img0`、`img1`应为`YUV`格式。
        ====
        `imgProcessStrategy`:
        `phashService.PADDING`:`size0` 为分块DCT的块大小；`size1` 为每个块中需要保留的低频区域尺寸
        `phashService.RESIZE`:`size0` 为重设之后的图像尺寸；`size1` 为需要保留的低频区域尺寸
        """
        if not channel in range(0,3):
            raise Exception("通道必须为：[0,1,2]中的数。它们分别代表：亮度Y、色度Cb、色度Cr。")
        img0=img0[:,:,channel]
        img1=img1[:,:,channel]
        if imgProcessStrategy==PADDING:
            if size0<size1:
                raise Exception("分块计算DCT时，块的大小必须不小于需要保留的低频频域区域大小。")
            regionSize=size0
            reserveLength=size1
            img0=self.reshape(img0,regionSize)
            img1=self.reshape(img1,regionSize)
            if not img0.shape==img1.shape:
                raise Exception("原始图像填充后尺寸不一致，为确保计算相似度时矩阵的尺寸一致，请将imgProcessStrategy设置为RESIZE模式。")
            results0=self.__calc_DCT(img0,regionSize,reserveLength)
            results1=self.__calc_DCT(img1,regionSize,reserveLength)
            dct_mean0=np.mean(results0)
            dct_mean1=np.mean(results1)
            results0=np.int8(results0>dct_mean0)
            results1=np.int8(results1>dct_mean1)
            distance=results0-results1
            similarityMatrix=np.int8(distance!=0)
            similarity=1-np.count_nonzero(similarityMatrix)/similarityMatrix.size
            return similarity
        elif imgProcessStrategy==RESIZE:
            if size0<size1:
                raise Exception("频域区域大小必须不小于要保留的低频频域区域大小。")
            img0=cv2.resize(img0,(size0,size0))
            img1=cv2.resize(img1,(size0,size0))
            results0=self.__calc_DCT(img0,size0,size1)
            results1=self.__calc_DCT(img1,size0,size1)
            dct_mean0=np.mean(results0)
            dct_mean1=np.mean(results1)
            results0=np.int8(results0>dct_mean0)
            results1=np.int8(results1>dct_mean1)
            distance=results0-results1
            similarityMatrix=np.int8(distance!=0)
            print(np.count_nonzero(similarityMatrix))
            similarity=1-np.count_nonzero(similarityMatrix)/similarityMatrix.size
        else:
            raise Exception("imgProcessStrategy 应为 repository_pHash.PADDING 或 repository_pHash.RESIZE。")
        return similarity            
