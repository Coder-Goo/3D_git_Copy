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

	for (uchar imageNo = 0; imageNo < 2; imageNo++)  //  imageNo分别表示左右相机的图片
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
	for (uchar direction = 0; direction < 2; direction++)//先把水平的集合起来，再把竖直的集合 起来得到最终的代价值
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
			cost_ptr[w] /= win_ptr[w];//复合赋值,因为沿着极线扫描的时候，右边的窗口大小不一致,所以要除以
			//窗口的大小
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
			if (phase_left_ptr[w] <0.01 || cost_map_ptr[w] < 0.001)//如果左相位图位于空白区域，则跳过
			{
				continue;
			}
			else{				
					if (directionH == 0){  //计算水平方向
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



Mat Aggregation::ComputeLimits(int directionH, int directionW, int imageNo)//计算臂长值
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
			if (images_ptr[w] < 0.01)//如果相位图中该像素点为0；则直接返回臂长为0
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


//论文：Fixed window aggregation AD-census algorithm for phase-based stereo matching。 的计算方法
inline int Aggregation::ComputeLimit(int row, int col, int directionH, int directionW, uchar imageNo) {
	double  phase_threshold;
	uint length;
	if (directionH != 0) { //计算竖直方向
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



////2011年文章： Fixed window aggregation AD-census algorithm for phase-based stereo matching  的自适应窗口臂长的计算方法
//int Aggregation::ComputeLimit(int height, int width, int directionH, int directionW, uchar imageNo){
//	// reference pixel
//	float  p = images[imageNo].at<float>(height, width);//height和width是坐标，调用此函数的给的参数是h和w的两个for循环
//	if (p < 0.01){//如果左图像中该像素点为0；则直接返回臂长为0
//		return 0;
//	}
//	else{
//		// coordinate of p1 the border patch pixel candidate
//		int d = 1;
//		int h1 = height + directionH;//directionH为-1
//		int w1 = width + directionW;//directionW为0，取这两个值时，计算的是向上的方向
//		// pixel value of p1 predecessor
//		float p2 = p;
//		// test if p1 is still inside the picture
//		bool inside = (0 <= h1) && (h1 < imgSize.height) && (0 <= w1) && (w1 < imgSize.width);
//		if (inside){
//			bool colorCond = true, wLimitCond = true, fColorCond = true;
//			while (colorCond && wLimitCond && fColorCond && inside){
//				float p1 = images[imageNo].at<float>(h1, w1);//p1变换，而p2作为临时值来做比较
//				// Do p1, p2 and p have similar color intensities?
//				colorCond = fabs(p - p1) < colorThreshold1 && fabs(p1 - p2) < colorThreshold3;//是否有相似的像素值
//				// Is window limit not reached?
//				wLimitCond = d < maxLength1;//是否达到最大臂展v 
//				// Better color similarities for farther neighbors?
//				fColorCond = (d <= maxLength2) || (d > maxLength2 && fabs(p - p1) < colorThreshold2);//我们使用maxLenghth1来保证在纹理少的区域可以得到足够大的区域，
//				//但是为了保证在纹理多的区域里使用该方法，我们还使用了maxLenghth2(maxLenghth2<maxLenghth1)来保证仍属于相似区域，防止纹理多的区域使用
//				//大阈值maxLenghth1使得超出阈值限定的相似区域
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