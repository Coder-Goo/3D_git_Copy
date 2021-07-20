#include"phase_unwrapping.h"
#include"biao_tou.h"

Phase_Unwrapping::Phase_Unwrapping(vector<Mat> &dst, Mat mask)
{
	this->dst = dst;
	this->mask = mask;

}
Mat Phase_Unwrapping::phase_unwrapping()
{
	vector<Mat>yml(12);
	vector<Mat>src;
	int ii = 0;
	for (vector<Mat>::iterator it = dst.begin(); it != dst.end(); it++)//输入的图片先变为灰度图
	{
		cvtColor(*it, yml[ii], COLOR_BGR2GRAY);
		//imshow("输入的图片", yml[ii]);
		src.push_back(yml[ii]);
		//waitKey(50);
		ii++;
	}
		vector<Mat> phase0;
		Mat phase1;
		phase1.create(src[1].size(), CV_32FC1);//create创造的Mat对象没有进行初始化
		phase0.push_back(phase1.clone());

		Mat phase2;
		phase2.create(src[1].size(), CV_32FC1);
		phase0.push_back(phase2.clone());//这里如果不使用clone（）函数，则浅复制导致只复制了指针，而没有复制对应的数字矩阵

		Mat phase3;
		phase3.create(src[1].size(), CV_32FC1);
		phase0.push_back(phase3.clone());
		int num = 0;
		for (int k = 0; k < 12; k += 4)//对每一个图片做atan2操作
		{
			for (int i = 0; i < src[1].rows; i++)
			{
				const uchar* p0 = src[k].ptr<uchar>(i);
				const uchar* p1 = src[k + 1].ptr<uchar>(i);
				const uchar* p2 = src[k + 2].ptr<uchar>(i);
				const uchar* p3 = src[k + 3].ptr<uchar>(i);

				float* phase00 = phase0[num].ptr<float>(i);//输出绝对相位图每行的首地址的指针
				for (int j = 0; j < src[1].cols; j++)
				{
					phase00[j] = static_cast<float>(atan2((p3[j] - p1[j]), (p0[j] - p2[j]))+3.1415926);
				}
			}
			num++;
		}
		/*Mat aaa = phase0[0];
		Mat aaaa = phase0[1];
		Mat aaaaa = phase0[2];
*/
		float P1 = static_cast<float>( FEN_GBIAN_LV / float(T1));
		float P2 = static_cast<float>( FEN_GBIAN_LV / float(T2));
		float P3 = static_cast<float>( FEN_GBIAN_LV / float(T3));

		float P12 = static_cast<float>(P1 * P2 / (P2 - P1));
		float P23 = static_cast<float>(P2 * P3 / (P3 - P2));
		float P123 = static_cast<float>(P23 * P12 / (P23 - P12));

		Mat Detaphase12;
		Detaphase12.create(src[1].size(), CV_32FC1);
		Mat Detaphase23;
		Detaphase23.create(src[1].size(), CV_32FC1);
		Mat Detaphase123;
		Detaphase123.create(src[1].size(), CV_32FC1);

		Mat n12;
		n12.create(src[1].size(), CV_32FC1);
		Mat n23;
		n23.create(src[1].size(), CV_32FC1);
		Mat n123;
		n123.create(src[1].size(), CV_32FC1);

/*		Mat N12;
		N12.create(src[1].size(), CV_8UC1);
		Mat N1;
		N1.create(src[1].size(), CV_8UC1)*/;

		Mat phai_1;
		phai_1.create(src[1].size(), CV_32FC1);
		Mat phai_12;
		phai_12.create(src[1].size(), CV_32FC1);
		Mat phai_23;
		phai_23.create(src[1].size(), CV_32FC1);

		Mat a_1;
		a_1.create(src[1].size(), CV_32FC1);
		Mat b_1;
		b_1.create(src[1].size(), CV_32FC1);
		Mat c_1;
		c_1.create(src[1].size(), CV_32FC1);

		Mat unwrap_phase = Mat::zeros(src[1].size(), CV_32FC1);

		for (int row = 0; row < src[1].rows; row++)
		{
			float*ptr1 = phase0[0].ptr<float>(row);
			float*ptr2 = phase0[1].ptr<float>(row);
			float*ptr3 = phase0[2].ptr<float>(row);
			float*ptr12 = Detaphase12.ptr<float>(row);
			float*ptr23 = Detaphase23.ptr<float>(row);
			float*ptr123 = Detaphase123.ptr<float>(row);
			float*n12ptr = n12.ptr<float>(row);
			float*n23ptr = n23.ptr<float>(row);
			float*n123ptr = n123.ptr<float>(row);

		/*	uchar*N12ptr = N12.ptr<uchar>(row);
			uchar*N1ptr = N1.ptr<uchar>(row);*/

			float*phai_1ptr = phai_1.ptr<float>(row);
			float*phai_12ptr = phai_12.ptr<float>(row);
			float*phai_23ptr = phai_23.ptr<float>(row);
			float*a_1ptr = a_1.ptr<float>(row);
			float*b_1ptr = b_1.ptr<float>(row);
			float*c_1ptr = c_1.ptr<float>(row);
			float*unwrap_phase_ptr = unwrap_phase.ptr<float>(row);
			
			Mat dist = mask;
			for (int col = 0; col < src[1].cols; col++)
			{
				/*******************************************优化位置******************************************/
				if (mask.at<uchar>(row,col)<0.01)
				{
					continue;
				}
				else
				{

					if (ptr1[col] >= ptr2[col])
					{
						ptr12[col] = ptr1[col] - ptr2[col];
					}
					else
					{
						ptr12[col] = 2 * PI + ptr1[col] - ptr2[col];
					}
					if (ptr2[col] >= ptr3[col])
					{
						ptr23[col] = ptr2[col] - ptr3[col];
					}
					else
					{
						ptr23[col] = 2 * PI + ptr2[col] - ptr3[col];
					}
					if (ptr12[col] >= ptr23[col])
					{
						ptr123[col] = ptr12[col] - ptr23[col];
					}
					else
					{
						ptr123[col] = 2 * PI + ptr12[col] - ptr23[col];
					}
					n123ptr[col] = ptr123[col] / (2 * PI);
					n12ptr[col] = P23 * n123ptr[col] / (P23 - P12);
					n23ptr[col] = P12 * n123ptr[col] / (P23 - P12);

				/*	N12ptr[col] = floor(n12ptr[col]);
					N1ptr[col] = floor(P2*(N12ptr[col] + ptr12[col] / (2 * PI)) / (P2 - P1));
					phai_1ptr[col] = 2 * PI*N1ptr[col] + ptr1[col];*/

					phai_12ptr[col] = static_cast < double>(2 * PI*round(P23*n123ptr[col] / (P23 - P12) - ptr12[col] / (2 * PI)) + ptr12[col]);
					phai_23ptr[col] = static_cast < double>(2 * PI*round(P12*n123ptr[col] / (P23 - P12) - ptr23[col] / (2 * PI)) + ptr23[col]);
					a_1ptr[col] = 2 * PI / 3.0 * P1*round((P12*phai_12ptr[col] + P23 * phai_23ptr[col]) / (4 * PI*P1) - ptr1[col] / (2 * PI)) + 1 / 3.0 * P1*ptr1[col];
					b_1ptr[col] = 2 * PI / 3.0 * P2*round((P12*phai_12ptr[col] + P23 * phai_23ptr[col]) / (4 * PI*P2) - ptr2[col] / (2 * PI)) + 1 / 3.0 * P2*ptr2[col];
					c_1ptr[col] = 2 * PI / 3.0 * P3*round((P12*phai_12ptr[col] + P23 * phai_23ptr[col]) / (4 * PI*P3) - ptr3[col] / (2 * PI)) + 1 / 3.0 * P3*ptr3[col];//一定要注意这里一定要除以3.0，否则出问题
					unwrap_phase_ptr[col] = static_cast < double>( 1 / P1 * (a_1ptr[col] + b_1ptr[col] + c_1ptr[col]));
				}
				
			}
		}
		//Mat aaa = a_1;
		return unwrap_phase.clone();
}