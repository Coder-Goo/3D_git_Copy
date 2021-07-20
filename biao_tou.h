#pragma once
#ifndef BIAOTOU
#define BIAOTOU
#include<opencv2/opencv.hpp>
#include<iostream >
#include<fstream>
#include<vector>
#include<opencv2/imgproc/types_c.h>
#include<cmath>
#include<pcl/common/common_headers.h>
#include<pcl/io/pcd_io.h>
#include<pcl/point_types.h>
#include<pcl/point_cloud.h>
#include<string>



#define qipan_cornor_x 11 //棋盘格角点数
#define qipan_cornor_y 8

#define qipan_size_x 6//棋盘格的尺寸
#define qipan_size_y 6

#define FEN_GBIAN_LV 1280
#define T1 77.0//这里一定要写成小数形式，否则出问题
#define T2 73.0
#define T3 70.0
#define PI 3.1415926
#define DEFAULT_DISPARITY -10000

//#define INVALID_VALUE 

#endif 
