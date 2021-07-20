
//输入左右相机的标定图片提取的角点vector和相应的世界坐标点，得到一个标定参数文件
#pragma once
#ifndef JIAOZHENG
#define JIAOZHENG
#include"biao_tou.h"
using namespace cv;
using namespace std;


class Calibration
{
public:
	Calibration(string calib_file_name, vector<vector<Point2f>>imagePointVecL, vector<vector<Point2f>>imagePointVecR, vector<vector<Point3f>>realPointVec, Size imageSize);//构造函数，输入的文件流
	int calibration();//标定和矫正函数，输出重映射矩阵到yml文件中去
private:
	vector<vector<Point2f>>imagePointVecL;
	vector<vector<Point2f>>imagePointVecR;
	vector<vector<Point3f>>realPointVec;
	Size imagesize;
	string calib_file_name;
};
#endif
