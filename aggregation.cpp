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

	for (uchar imageNo = 0; imageNo < 2; imageNo++)
	{
		upLimits[imageNo] = ComputeLimits(-1, 0, imageNo);
		downLimits[imageNo] = ComputeLimits(1, 0, imageNo);
		leftLimits[imageNo] = ComputeLimits(0, -1, imageNo);
		rightLimits[imageNo] = ComputeLimits(0, 1, imageNo);
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

	if (horizontalFirst)
		std::swap(directionH, directionW);

	Mat windowsSizes = Mat::ones(imgSize, CV_32S);

	for (uchar direction = 0; direction < 2; direction++)//先把水平的集合起来，再把竖直的集合 起来得到最终的代价值
	{
		(Aggregation_1D(costMap, directionH, directionW, windowsSizes, imageNo)).copyTo(costMap);
		std::swap(directionH, directionW);
	}

	for (size_t h = 0; h < imgSize.height; h++)
	{
		for (size_t w = 0; w < imgSize.width; w++)
		{
			costMap.at<float>(h, w) /= windowsSizes.at<int>(h, w);//复合赋值,因为沿着极线扫描的时候，右边的窗口大小不一致,所以要除以
			//窗口的大小
		}
	}
	return costMap.clone();
	
}

Mat Aggregation::Aggregation_1D(const Mat &costMap, int directionH, int directionW, Mat &windowSizes, uchar imageNo)
{
	Mat tmpWindowSizes = Mat::zeros(imgSize, CV_32S);
	Mat aggregatedCosts(imgSize, CV_32F);
	int dmin, dmax, d;
	int h, w;
	for (h = 0; h < imgSize.height; h++)
	{
		for (w = 0; w < imgSize.width; w++)
		{
			if (costMap.at<float>(h, w) < 0.001)//如果左相位图位于空白区域，则跳过
			{
				continue;
			}
			else
			{
				
					if (directionH == 0)
					{
						dmin = -leftLimits[imageNo].at<int>(h, w);
						dmax = rightLimits[imageNo].at<int>(h, w);
					}
					else
					{
						dmin = -upLimits[imageNo].at<int>(h, w);
						dmax = downLimits[imageNo].at<int>(h, w);
					}

					float cost = 0;
					for (d = dmin; d <= dmax; d++)
					{
						cost += costMap.at<float>(h + d * directionH, w + d * directionW);
						tmpWindowSizes.at<int>(h, w) += windowSizes.at<int>(h + d * directionH, w + d * directionW);
					}
					aggregatedCosts.at<float>(h, w) = cost;
				
			}
			
		}
	}

	tmpWindowSizes.copyTo(windowSizes);

	return aggregatedCosts.clone();
}



Mat Aggregation::ComputeLimits(int directionH, int directionW, int imageNo)//计算臂长值
{
	Mat limits(imgSize, CV_32S);
	int h, w;
	for (h = 0; h < imgSize.height; h++)
	{
		for (w = 0; w < imgSize.width; w++)
		{
			if (images[imageNo].at<float>(h, w) < 0.01)//如果左图像中该像素点为0；则直接返回臂长为0
			{
				limits.at<int>(h, w) = 0.0;
				continue;
			}
			else
			{
				limits.at<int>(h, w) = ComputeLimit(h, w, directionH, directionW, imageNo);//这个函数返回的是臂长值减一
			}
		}
	}
	return limits.clone();
}

int Aggregation::ComputeLimit(int height, int width, int directionH, int directionW, uchar imageNo)
{
	// reference pixel
	float  p = images[imageNo].at<float>(height, width);//height和width是坐标，调用此函数的给的参数是h和w的两个for循环
	if (p < 0.01)//如果左图像中该像素点为0；则直接返回臂长为0
	{
		return 0;
	}
	else
	{
		// coordinate of p1 the border patch pixel candidate
		int d = 1;
		int h1 = height + directionH;//directionH为-1
		int w1 = width + directionW;//directionW为0，取这两个值时，计算的是向上的方向

		// pixel value of p1 predecessor
		float p2 = p;

		// test if p1 is still inside the picture
		bool inside = (0 <= h1) && (h1 < imgSize.height) && (0 <= w1) && (w1 < imgSize.width);

		if (inside)
		{
			bool colorCond = true, wLimitCond = true, fColorCond = true;

			while (colorCond && wLimitCond && fColorCond && inside)
			{
				float p1 = images[imageNo].at<float>(h1, w1);//p1变换，而p2作为临时值来做比较

				// Do p1, p2 and p have similar color intensities?
				colorCond = fabs(p - p1) < colorThreshold1 && fabs(p1 - p2) < colorThreshold3;//是否有相似的像素值
				// Is window limit not reached?
				wLimitCond = d < maxLength1;//是否达到最大臂展v 

				// Better color similarities for farther neighbors?
				fColorCond = (d <= maxLength2) || (d > maxLength2 && fabs(p - p1) < colorThreshold2);//我们使用maxLenghth1来保证在纹理少的区域可以得到足够大的区域，
				//但是为了保证在纹理多的区域里使用该方法，我们还使用了maxLenghth2(maxLenghth2<maxLenghth1)来保证仍属于相似区域，防止纹理多的区域使用
				//大阈值maxLenghth1使得超出阈值限定的相似区域

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

