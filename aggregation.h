#pragma once
#ifndef AGGREGATION_H
#define AGGREGATION_H
#include"biao_tou.h"


using namespace std;
using namespace cv;

class Aggregation
{
public:
	Aggregation(const Mat &leftImage, const Mat &rightImage, double phase_threshold_v, double phase_threshold_h, uint length_v, uint length_h);
	/*int Match();*/
	Mat Aggregation2D(Mat &costMap, bool horizontalFirst, uchar imageNo);
	//void GetLimits(vector<Mat> &upLimits, vector<Mat> &downLimits, vector<Mat> &leftLimits, vector<Mat> &rightLimits) const;
private:
	Mat images[2];

	double phase_threshold_v, phase_threshold_h;
	uint length_v, length_h;
	vector<Mat> upLimits;
	vector<Mat> downLimits;
	vector<Mat> leftLimits;
	vector<Mat> rightLimits;
	int H, W;


	int ComputeLimit(int height, int width, int directionH, int directionW, uchar imageNo);//计算臂长的函数
	Mat ComputeLimits(int directionH, int directionW, int imageNo);//计算臂长的函数,该函数调用上个函数ComputeLimit

	Mat Aggregation_1D(const Mat &costMap, int directionH, int directionW, Mat &windowSizes, uchar imageNo);

};

#endif





