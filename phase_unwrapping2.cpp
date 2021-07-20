#include"phase_unwrapping2.h"
//解相主函数
//输入校正的图片集和掩膜图片
//返回相位图片
Mat PhaseUnwrapping2::PhaseUnwrapping3Frequency4step(const vector<Mat> &vecRect, Mat Mask)
{
	
	vector<Mat> ImgFrequency[3];
	//将每个频率的图片分别放入一个vector中
	for (int i = 0; i < 12; i++)
	{
		if (i < 4)
		{
			Mat img_gray;
			cvtColor(vecRect[i], img_gray, CV_RGB2GRAY);
			ImgFrequency[0].push_back(img_gray);
		}
		else if (i >= 4 && i < 8)
		{
			Mat img_gray;
			cvtColor(vecRect[i], img_gray, CV_RGB2GRAY);
			ImgFrequency[1].push_back(img_gray);
		}
		else
		{
			Mat img_gray;
			cvtColor(vecRect[i], img_gray, CV_RGB2GRAY);
			ImgFrequency[2].push_back(img_gray);
		}
	}
	//获得三个频率的初始相位
	Mat phase1 = solutionPHase4step(ImgFrequency[0], Mask);
	Mat phase2 = solutionPHase4step(ImgFrequency[1], Mask);
	Mat phase3 = solutionPHase4step(ImgFrequency[2], Mask);
	Mat unWrappPhase1 = getPhaseUnwrappingNewMothed(phase1, phase2, phase3);
	return unWrappPhase1;
}



//解得每一个频率的绝对相位
Mat PhaseUnwrapping2::solutionPHase4step(std::vector<cv::Mat>& Image, Mat mask)
{
	Mat PHase = Mat::zeros(Image[0].size(), CV_64FC1);
	Mat PHaseToMask;
	for (int row = 0; row < PHase.rows; row++)
	{
		double* ptrPHase = PHase.ptr<double>(row);
		uchar*  ptrMask = mask.ptr<uchar>(row);
		uchar* ptrImage[4];
		ptrImage[0] = Image[0].ptr<uchar>(row);
		ptrImage[1] = Image[1].ptr<uchar>(row);
		ptrImage[2] = Image[2].ptr<uchar>(row);
		ptrImage[3] = Image[3].ptr<uchar>(row);
		for (int col = 0; col < PHase.cols; col++)
		{
			//if (ptrMask[col] < 0.01) continue;
			double value = static_cast<double>(atan2((ptrImage[3][col] - ptrImage[1][col]), (ptrImage[0][col] - ptrImage[2][col])));
			ptrPHase[col] = CV_PI + value;
		}
	}
	PHase.copyTo(PHaseToMask, mask);
	
	return PHaseToMask;
}


Mat PhaseUnwrapping2::getPHaseFromTwoFluency(Mat &PHase1, Mat &PHase2)
{
	Mat Phase12 = Mat::zeros(PHase1.size(), CV_64FC1);
	for (int row = 0; row < PHase1.rows; row++)
	{
		double* ptrPHase1 = PHase1.ptr<double>(row);
		double* ptrPHase2 = PHase2.ptr<double>(row);
		double* ptrPHase12 = Phase12.ptr<double>(row);
		for (int col = 0; col < PHase1.cols; col++)
		{
			if (ptrPHase1[col] >= ptrPHase2[col])
			{
				double value = ptrPHase1[col] - ptrPHase2[col];
				ptrPHase12[col] = value;
			}
			else
			{
				double value = 2 * CV_PI - ptrPHase2[col] + ptrPHase1[col];
				ptrPHase12[col] = value;
			}
		}
	}
	return Phase12;
}

Mat PhaseUnwrapping2::getPhaseUnwrappingNewMothed(Mat &Phase1, Mat Phase2, Mat &Phase3)
{
	double t1, t2, t3, t23, t12;//一个条纹占多少个像素
	int width = Phase1.cols;
	t1 = static_cast<double>(width) / frequency[0];
	t2 = static_cast<double>(width) / frequency[1];
	t3 = static_cast<double>(width) / frequency[2];
	t12 = t1 * t2 / (t2 - t1);
	t23 = t2 * t3 / (t3 - t2);
	Mat phase12 = getPHaseFromTwoFluency(Phase1, Phase2);
	Mat phase23 = getPHaseFromTwoFluency(Phase2, Phase3);
	Mat phase123 = getPHaseFromTwoFluency(phase12, phase23);
	Mat unwrappingPhase1 = Mat::zeros(Phase1.size(), CV_64FC1);
	Mat unwrappingPhase12 = Mat::zeros(Phase1.size(), CV_64FC1);
	Mat unwrappingPhase23 = Mat::zeros(Phase1.size(), CV_64FC1);
	for (int row = 0; row < Phase1.rows; row++)
	{
		double *ptrunwrappingPhase1 = unwrappingPhase1.ptr<double>(row);
		double *ptrPhase1 = Phase1.ptr<double>(row);
		double *ptrPhase2 = Phase2.ptr<double>(row);
		double *ptrPhase3 = Phase3.ptr<double>(row);
		double *ptrPhase12 = phase12.ptr<double>(row);
		double *ptrPhase23 = phase23.ptr<double>(row);
		double *ptrPhase123 = phase123.ptr<double>(row);


		double *ptrUnwrappingPhase12 = unwrappingPhase12.ptr<double>(row);
		double *ptrUnwrappingPhase23 = unwrappingPhase23.ptr<double>(row);
		for (int col = 0; col < Phase1.cols; col++)
		{
			ptrUnwrappingPhase12[col] = static_cast<double>
				(2 * CV_PI*round((t23*ptrPhase123[col] / (t23 - t12) - ptrPhase12[col]) / (2 * CV_PI)) + ptrPhase12[col]);
			ptrUnwrappingPhase23[col] = static_cast <double>
				(2 * CV_PI*round((t12*ptrPhase123[col] / (t23 - t12) - ptrPhase23[col]) / (2 * CV_PI)) + ptrPhase23[col]);
			double sum = 0;
			double T[3] = { t1,t2,t3 };
			double PHase[3] = { ptrPhase1[col],ptrPhase2[col],ptrPhase3[col] };
			for (int i = 0; i < 3; i++)
			{
				double Round =
					round((t12*ptrUnwrappingPhase12[col] + t23 * ptrUnwrappingPhase23[col]) / (4 * CV_PI*T[i]) - PHase[i] / (2 * CV_PI));
				sum += 2 * CV_PI*T[i] * Round + T[i] * PHase[i];
			}
			ptrunwrappingPhase1[col] = static_cast<double>(1 / t1 * 1 / 3 * sum);
		}
	}
	return unwrappingPhase1;
}