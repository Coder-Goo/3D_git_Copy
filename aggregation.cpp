#include"aggregation.h"
#include"biao_tou.h"

using namespace std;
using namespace cv;

Aggregation::Aggregation(const Mat &leftImage, const Mat &rightImage,  double phase_threshold_v, double phase_threshold_h, uint length_v, uint length_h)
{
	this->images[0] = leftImage;
	this->images[1] = rightImage;
	this ->H = leftImage.rows;
	this ->W = leftImage.cols;
	this->phase_threshold_v = phase_threshold_v;
	this->phase_threshold_h = phase_threshold_h;
	this->length_v = length_v;
	this->length_h = length_h;
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
	Mat windowsSizes = Mat::ones(Size(W, H), CV_8UC1);
	for (uchar direction = 0; direction < 2; direction++)//�Ȱ�ˮƽ�ļ����������ٰ���ֱ�ļ��� �����õ����յĴ���ֵ
	{
		(Aggregation_1D(costMap, directionH, directionW, windowsSizes, imageNo)).copyTo(costMap);
		std::swap(directionH, directionW);
	}
	for (size_t h = 0; h < H; h++)
	{
		float*cost_ptr = costMap.ptr<float>(h);
		uchar*win_ptr = windowsSizes.ptr<uchar>(h);
		for (size_t w = 0; w < W; w++)
		{
			cost_ptr[w] /= win_ptr[w];//���ϸ�ֵ,��Ϊ���ż���ɨ���ʱ���ұߵĴ��ڴ�С��һ��,����Ҫ����
			//���ڵĴ�С
		}
	}
	return costMap.clone();  
}

Mat Aggregation::Aggregation_1D(const Mat &costMap, int directionH, int directionW, Mat &windowSizes, uchar imageNo)
{

	Mat tmpWindowSizes = Mat::zeros(Size(W,H), CV_8UC1);
	Mat aggregatedCosts(Size(W, H), CV_32F);

	float*phase_left_ptr = nullptr;
	const float* cost_map_ptr = nullptr;
	uchar *left_limit_ptr = nullptr;
	uchar * right_limit_ptr = nullptr;
	uchar *up_limit_ptr = nullptr;
	uchar *down_limit_ptr = nullptr;
	uchar * temp_win_size_ptr = nullptr;
	uchar * win_size_ptr = nullptr;
	float* aggre_cost_ptr = nullptr;

	for (int h = 0; h < H; h++){
		phase_left_ptr = images[0].ptr<float>(h);
	    cost_map_ptr = costMap.ptr<float>(h);
		left_limit_ptr = leftLimits[imageNo].ptr<uchar>(h);
		right_limit_ptr = rightLimits[imageNo].ptr<uchar>(h);
		up_limit_ptr = upLimits[imageNo].ptr<uchar>(h);
		down_limit_ptr = downLimits[imageNo].ptr<uchar>(h);
		temp_win_size_ptr = tmpWindowSizes.ptr<uchar>(h);
		win_size_ptr = windowSizes.ptr<uchar>(h);
		aggre_cost_ptr = aggregatedCosts.ptr<float>(h);

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
							temp_win_size_ptr[ w] += windowSizes.at<uchar>(h + d * directionH, w );
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
	Mat limits = Mat::ones(Size(W,H), CV_8UC1 );
	//*******************************************************************
	uchar* limits_ptr = nullptr;
	float* images_ptr = nullptr;
	for (int h = 0; h < H; h++)
	{
		 images_ptr = images[imageNo].ptr<float>(h);
		 limits_ptr= limits.ptr<uchar>(h);
		for (int w = 0; w < W; w++){
			if (images_ptr[w] < 0.01)//�����λͼ�и����ص�Ϊ0����ֱ�ӷ��ر۳�Ϊ0
			{
				limits_ptr[w] = 0;
			}
			else{
				limits_ptr[w] += ComputeLimit(h, w, directionH, directionW, imageNo);
			}
		}
	}
	//*************************************************************************
	return limits;
}


//���ģ�Fixed window aggregation AD-census algorithm for phase-based stereo matching�� �ļ��㷽��
inline int Aggregation::ComputeLimit(int row, int col, int directionH, int directionW, uchar imageNo) {
	double  phase_threshold;
	uint length;
	if (directionH != 0) { //������ֱ����
		phase_threshold = phase_threshold_h;
		length = length_h;
	}
	else {
		phase_threshold = phase_threshold_v;
		length = length_v;
	}
	int r = row + directionH;
	int c = col + directionW;
	float p = images[imageNo].at<float>(row, col);
	bool inside = (0 <= r) && (r < H) && (0 <= c) && (c < W);
	int d = 0;
	if (inside) {
		bool phase_con = true, length_con = true;
		while (phase_con && length_con && inside) {
			d++;
			float p1 = images[imageNo].at<float>(r, c);
			phase_con = fabs(p - p1) < phase_threshold;
			length_con = d < length;
			r = row + directionH;
			c = col + directionW;
			inside = (0 <= r) && (r < H) && (0 <= c) && (c < W);
		}
	}
	return d;
}



////2011�����£� Fixed window aggregation AD-census algorithm for phase-based stereo matching  ������Ӧ���ڱ۳��ļ��㷽��
//int Aggregation::ComputeLimit(int height, int width, int directionH, int directionW, uchar imageNo){
//	// reference pixel
//	float  p = images[imageNo].at<float>(height, width);//height��width�����꣬���ô˺����ĸ��Ĳ�����h��w������forѭ��
//	if (p < 0.01){//�����ͼ���и����ص�Ϊ0����ֱ�ӷ��ر۳�Ϊ0
//		return 0;
//	}
//	else{
//		// coordinate of p1 the border patch pixel candidate
//		int d = 1;
//		int h1 = height + directionH;//directionHΪ-1
//		int w1 = width + directionW;//directionWΪ0��ȡ������ֵʱ������������ϵķ���
//		// pixel value of p1 predecessor
//		float p2 = p;
//		// test if p1 is still inside the picture
//		bool inside = (0 <= h1) && (h1 < imgSize.height) && (0 <= w1) && (w1 < imgSize.width);
//		if (inside){
//			bool colorCond = true, wLimitCond = true, fColorCond = true;
//			while (colorCond && wLimitCond && fColorCond && inside){
//				float p1 = images[imageNo].at<float>(h1, w1);//p1�任����p2��Ϊ��ʱֵ�����Ƚ�
//				// Do p1, p2 and p have similar color intensities?
//				colorCond = fabs(p - p1) < colorThreshold1 && fabs(p1 - p2) < colorThreshold3;//�Ƿ������Ƶ�����ֵ
//				// Is window limit not reached?
//				wLimitCond = d < maxLength1;//�Ƿ�ﵽ����չv 
//				// Better color similarities for farther neighbors?
//				fColorCond = (d <= maxLength2) || (d > maxLength2 && fabs(p - p1) < colorThreshold2);//����ʹ��maxLenghth1����֤�������ٵ�������Եõ��㹻�������
//				//����Ϊ�˱�֤��������������ʹ�ø÷��������ǻ�ʹ����maxLenghth2(maxLenghth2<maxLenghth1)����֤�������������򣬷�ֹ����������ʹ��
//				//����ֵmaxLenghth1ʹ�ó�����ֵ�޶�����������
//				p2 = p1;
//				h1 += directionH;
//				w1 += directionW;
//				// test if p1 is still inside the picture
//				inside = (0 <= h1) && (h1 < imgSize.height) && (0 <= w1) && (w1 < imgSize.width);
//
//				d++;
//			}
//			d--;
//		}
//		return d - 1;
//	}
//}