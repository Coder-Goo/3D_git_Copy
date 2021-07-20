#pragma once
#include"biao_tou.h"

using namespace std;
using namespace cv;
//ʹ��ʦ�ֵĴ���������λ
class  PhaseUnwrapping2 {
public:
	Mat PhaseUnwrapping3Frequency4step(const vector<Mat> &vecRect, Mat Mask);
private:
	vector<cv::Mat> img_vec;
	Mat mask;
	double frequency[3] = { 77.0,73.0,70.0 };

	Mat solutionPHase4step(std::vector<cv::Mat>& Image, Mat mask);
	Mat getPhaseUnwrappingNewMothed(Mat &Phase1, Mat Phase2, Mat &Phase3);
	Mat getPHaseFromTwoFluency(Mat &PHase1, Mat& PHase2);

};