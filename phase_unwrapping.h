//�����dst�Ǵ洢��ͨ����ɫͼ��
#ifndef PHASE_UNWRAPPING
#define PHASE_UNWRAPPING
#pragma once
#include"calibration.h"

using namespace std;
using namespace cv;

class Phase_Unwrapping
{
public:
	Phase_Unwrapping(vector<Mat> &dst , Mat mask);
	Mat phase_unwrapping();
private:
	vector<Mat> dst;
	Mat mask;
};
#endif