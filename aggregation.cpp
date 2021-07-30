#include"aggregation.h"
#include"biao_tou.h"

using namespace std;
using namespace cv;

Aggregation::Aggregation(const Mat &leftImage, const Mat &rightImage, double colorThreshold1, double colorThreshold2,double colorThreshold3,
	uint maxLength1, uint maxLength2)
{
	this->images[0] = leftImage;
	this->images[1] = rightImage;
	this->imgSize = leftImage.size();
	this->colorThreshold1 = colorThreshold1;
	this->colorThreshold2 = colorThreshold2;
	this->colorThreshold3 = colorThreshold3;
	this->maxLength1 = maxLength1;
	this->maxLength2 = maxLength2;
	this->upLimits.resize(2);
	this->downLimits.resize(2);
	this->leftLimits.resize(2);
	this->rightLimits.resize(2);

	for (uchar imageNo = 0; imageNo < 2; imageNo++)  //  imageNo�ֱ��ʾ���������ͼƬ
	{
		upLimits[imageNo]=( ComputeLimits(-1, 0, imageNo));
		downLimits[imageNo] = ( ComputeLimits(1, 0, imageNo));
		leftLimits[imageNo] = ( ComputeLimits(0, -1, imageNo));
		rightLimits[imageNo] = (ComputeLimits(0, 1, imageNo));
	}
	//Mat right= rightLimits[0];
	//Mat left= leftLimits[0];
	//Mat up= upLimits[0];
	//Mat down= downLimits[0];
	//waitKey(50);
}


Mat Aggregation::Aggregation2D( Mat &costMap, bool horizontalFirst, uchar imageNo)
{
	int directionH = 1, directionW = 0; 
	if (horizontalFirst) std::swap(directionH, directionW);
	Mat windowsSizes = Mat::ones(imgSize, CV_8UC1);
	for (uchar direction = 0; direction < 2; direction++)//�Ȱ�ˮƽ�ļ����������ٰ���ֱ�ļ��� �����õ����յĴ���ֵ
	{
		(Aggregation_1D(costMap, directionH, directionW, windowsSizes, imageNo)).copyTo(costMap);
		std::swap(directionH, directionW);
	}
	for (size_t h = 0; h < imgSize.height; h++)
	{
		float*cost_ptr = costMap.ptr<float>(h);
		float*win_ptr = windowsSizes.ptr<float>(h);
		for (size_t w = 0; w < imgSize.width; w++)
		{
			cost_ptr[w] /= win_ptr[w];//���ϸ�ֵ,��Ϊ���ż���ɨ���ʱ���ұߵĴ��ڴ�С��һ��,����Ҫ����
			//���ڵĴ�С
		}
	}
	return costMap.clone();  
}

Mat Aggregation::Aggregation_1D(const Mat &costMap, int directionH, int directionW, Mat &windowSizes, uchar imageNo)
{
	Mat temp1 = leftLimits[imageNo];
	Mat temp2 = rightLimits[imageNo];

	Mat tmpWindowSizes = Mat::zeros(imgSize, CV_8UC1);
	Mat aggregatedCosts(imgSize, CV_32F);
	int H = imgSize.height, W = imgSize.width;
	for (int h = 0; h < H; h++){
		float*phase_left_ptr = images[0].ptr<float>(h);
		const float* cost_map_ptr = costMap.ptr<float>(h);
		int *left_limit_ptr = leftLimits[imageNo].ptr<int>(h);
		int * right_limit_ptr = rightLimits[imageNo].ptr<int>(h);
		int *up_limit_ptr = upLimits[imageNo].ptr<int>(h);
		int *down_limit_ptr = downLimits[imageNo].ptr<int>(h);
		int * temp_win_size_ptr = tmpWindowSizes.ptr<int>(h);
		int * win_size_ptr = windowSizes.ptr<int>(h);
		float* aggre_cost_ptr = aggregatedCosts.ptr<float>(h);

		for (int w = 0; w < W; w++){
			if (phase_left_ptr[w] <0.01 || cost_map_ptr[w] < 0.001)//�������λͼλ�ڿհ�����������
			{
				continue;
			}
			else{				
					if (directionH == 0){  //����ˮƽ����
						float cost = 0.0;
						for (int d = -left_limit_ptr[w]; d <= right_limit_ptr[w]; d++) {
							//cout << w + d * directionW;
							cost += cost_map_ptr[w + d * directionW];
							temp_win_size_ptr[ w] += win_size_ptr[ w + d * directionW];
						}
						aggre_cost_ptr[w] = cost;
					}
					else{
						float cost = 0;
						for ( int d = -up_limit_ptr[w]; d <= down_limit_ptr[w]; d++) {
							cost += costMap.at<float>(h + d * directionH, w);
							temp_win_size_ptr[ w] += windowSizes.at<int>(h + d * directionH, w );
						}
						aggre_cost_ptr[w] = cost;
					}
			}
		}
	}
	tmpWindowSizes.copyTo(windowSizes);

	return aggregatedCosts.clone();
}



Mat Aggregation::ComputeLimits(int directionH, int directionW, int imageNo)//����۳�ֵ
{
	Mat limits = Mat::zeros(imgSize, CV_8UC1 );
	Mat temp = images[imageNo];
	int H = limits.rows;
	int W = limits.cols;
	for (int h = 0; h < H; h++)
	{
		float* images_ptr = images[imageNo].ptr<float>(h);
		float*limits_ptr= limits.ptr<float>(h);

		for (int w = 0; w < W; w++)
		{
			if (images_ptr[w] < 0.01)//�����λͼ�и����ص�Ϊ0����ֱ�ӷ��ر۳�Ϊ0
			{
				limits_ptr[w] = 0;
				continue;
			}
			else
			{
				limits_ptr[w] = ComputeLimit(h, w, directionH, directionW, imageNo);//����������ص��Ǳ۳�ֵ��һ
			}
		}
	}
	return limits;
}

int Aggregation::ComputeLimit(int height, int width, int directionH, int directionW, uchar imageNo)
{
	// reference pixel
	float  p = images[imageNo].at<float>(height, width);//height��width�����꣬���ô˺����ĸ��Ĳ�����h��w������forѭ��
	if (p < 0.01)//�����ͼ���и����ص�Ϊ0����ֱ�ӷ��ر۳�Ϊ0
	{
		return 0;
	}
	else
	{
		// coordinate of p1 the border patch pixel candidate
		int d = 1;
		int h1 = height + directionH;//directionHΪ-1
		int w1 = width + directionW;//directionWΪ0��ȡ������ֵʱ������������ϵķ���

		// pixel value of p1 predecessor
		float p2 = p;

		// test if p1 is still inside the picture
		bool inside = (0 <= h1) && (h1 < imgSize.height) && (0 <= w1) && (w1 < imgSize.width);

		if (inside)
		{
			bool colorCond = true, wLimitCond = true, fColorCond = true;

			while (colorCond && wLimitCond && fColorCond && inside)
			{
				float p1 = images[imageNo].at<float>(h1, w1);//p1�任����p2��Ϊ��ʱֵ�����Ƚ�

				// Do p1, p2 and p have similar color intensities?
				colorCond = fabs(p - p1) < colorThreshold1 && fabs(p1 - p2) < colorThreshold3;//�Ƿ������Ƶ�����ֵ
				// Is window limit not reached?
				wLimitCond = d < maxLength1;//�Ƿ�ﵽ����չv 

				// Better color similarities for farther neighbors?
				fColorCond = (d <= maxLength2) || (d > maxLength2 && fabs(p - p1) < colorThreshold2);//����ʹ��maxLenghth1����֤�������ٵ�������Եõ��㹻�������
				//����Ϊ�˱�֤��������������ʹ�ø÷��������ǻ�ʹ����maxLenghth2(maxLenghth2<maxLenghth1)����֤�������������򣬷�ֹ����������ʹ��
				//����ֵmaxLenghth1ʹ�ó�����ֵ�޶�����������

				p2 = p1;
				h1 += directionH;
				w1 += directionW;

				// test if p1 is still inside the picture
				inside = (0 <= h1) && (h1 < imgSize.height) && (0 <= w1) && (w1 < imgSize.width);

				d++;
			}

			d--;
		}

		return d - 1;
	}

}

