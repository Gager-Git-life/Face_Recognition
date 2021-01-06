#!/bin/sh
###
 # @Descripttion: 
 # @version: 
 # @Author: Gager
 # @Date: 2020-11-26 15:50:00
 # @LastEditors: sueRimn
 # @LastEditTime: 2020-11-30 10:21:27
### 

if [ ! -d "./build/" ];then
  echo "[INFO]>>> 创建build文件夹"
  mkdir ./build
else
  echo "[INFO]>>> 清空build下内容"
  rm -rf build/*
fi

cd build
cmake ..
make -j8
mv MTCNN-MNN ..
cd ..

if [$1="pic"]
then
	./MTCNN-MNN 
else
	./MTCNN-MNN 
fi