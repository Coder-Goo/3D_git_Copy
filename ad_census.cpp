#include"ad_census.h"


//计算代价的函数，输入的vector<vector<>>是DSI,传入的是引用类型
void CostCalculation(vector<vector<Mat>>&vec_cost_maps, const Mat &phase_left, const Mat &phase_right, const uchar &cost_window_size)//如果需要改变vector中的数据需要进行引用传参
{
	clock_t time_cost_computation_start, time_cost_computation_end;
	time_cost_computation_start = clock();
	const int height = phase_left.rows;
	const int width = phase_left.cols;

	//到时这里可以搞一个初始化函数来进行初始化
	vec_cost_maps.resize(2);//存放一个负视差的，一个正视差的；
	vec_cost_maps[0].resize(abs(min_disparity) + 1);//最小视差数加一个0视差
	vec_cost_maps[1].resize(abs(max_disparity) + 1);//最大视差数加一个0视差

	for (int i = 0; i < vec_cost_maps.size(); i++) {
		for (int j = 0; j < vec_cost_maps[i].size(); j++) {
			vec_cost_maps[i][j] = Mat::zeros(phase_left.size(), CV_32FC1);//指定每个视差图的大小
		}
	}

	for (int i = 0; i < 2; i++)//进行两次循环，一次正视差，一次负视差
	{
		int d_length = 0;
		if (i == 0) {//第一次先进行负视差匹配
			d_length = fabs(min_disparity);
		}
		else {
			d_length = fabs(max_disparity);
		}
		for (int d = 0; d < d_length; d++) {
			//下面三次for循环得到disparity space image（DSI）
			for (int row = cost_window_size / 2 + 1; row < height - cost_window_size / 2 - 1; row++) {
				for (int col = cost_window_size / 2 + 1; col < width - cost_window_size / 2 - 1; col++) {//两次循环，对相位图进行逐像素
					if (phase_left.at<float>(row, col) < 0.01) {//如果左相位图的像素值为0，跳过匹配
						continue;
					}
					else {
						int col_r;
						if (i == 0) {
							col_r = col + d;//负视差的时候，右图视差和左图视差之间的关系
						}
						else {
							col_r = col - d;//正视差的时候，右图视差和左图视差之间的关系
						}
						if (col_r >= (width - cost_window_size / 2) || col_r <= cost_window_size / 2) {
							continue;
						}
						else {
							if (phase_right.at<float>(row, col_r) < 0.01) {
								continue;
							}
							else {
								if (ad_census) {//使用ad_census进行代价计算	
									vec_cost_maps[i][d].at<float>(row, col) = compute_ad_census_cost(row, col, row, col_r, phase_left, phase_right, cost_window_size);
								}
								if (ad_rank) {//使用ad_rank进行代价计算
									vec_cost_maps[i][d].at<float>(row, col) = compute_ad_rank_cost(row, col, row, col_r, phase_left, phase_right, cost_window_size);
								}
							}
						}
					}
				}
			}
		}
	}
	time_cost_computation_end = clock();
	std::cout << "代价计算用时：" << double((time_cost_computation_end - time_cost_computation_start) / CLOCKS_PER_SEC) << endl;
}

//代价聚合
void CostAggregation(bool fixed_window, vector<vector<Mat>>&vec_cost_maps, vector<vector<Mat>>&vec_aggregation_maps, const Mat &phase_left, const Mat &phase_right, const uchar &aggregation_window_size)
{
	/*cout << "代价DSI的负视差大小是"<<vec_cost_maps[0].size()<<endl;
	cout << "代价DSI的正视差大小是" << vec_cost_maps[1].size() << endl;*/
	/******************************************///到时这里可以搞一个初始化函数来进行初始化
	vec_aggregation_maps.resize(2);//存放一个负视差的，一个正视差的；
	vec_aggregation_maps[0].resize(abs(min_disparity) + 1);//最小视差数加一个0视差
	vec_aggregation_maps[1].resize(abs(max_disparity) + 1);//最大视差数加一个0视差

	for (int i = 0; i < vec_cost_maps.size(); i++) {
		for (int j = 0; j < vec_cost_maps[i].size(); j++) {
			vec_aggregation_maps[i][j] = Mat::zeros(phase_left.size(), CV_32FC1);//指定每个视差图的大小
		}
	}

	const int height = phase_left.rows;
	const int width = phase_left.cols;

	if (fixed_window == true) {	//使用固定窗口

		clock_t time_Faggregation_start, time_Faggregation_end;
		time_Faggregation_start = clock();
		for (int i = 0; i < 2; i++) {//进行两次循环，一次正视差，一次负视差
			int d_length = 0;
			if (i == 0) {//第一次先进行负视差匹配
				d_length = fabs(min_disparity);
			}
			else {
				d_length = fabs(max_disparity);
			}
			for (int d = 0; d < d_length; d++) {
				for (int row = aggregation_window_size / 2 + 1; row < height - aggregation_window_size / 2 - 1; row++) {
					for (int col = aggregation_window_size / 2 + 1; col < width - aggregation_window_size / 2 - 1; col++) {//两次循环，对相位图进行逐像素
						if (vec_cost_maps[i][d].at<float>(row, col) < 0.01) {
							continue;
						}
						else {
							Mat kernel = vec_cost_maps[i][d](Rect(col - aggregation_window_size / 2, row - aggregation_window_size / 2, aggregation_window_size, aggregation_window_size));
							Scalar value = sum(kernel);
							float temp_value = value[0];
							vec_aggregation_maps[i][d].at<float>(row, col) = temp_value;
						}
					}
				}
			}
		}
		time_Faggregation_end = clock();
		std::cout << "固定窗口代价聚合用时：" << double((time_Faggregation_end - time_Faggregation_start) / CLOCKS_PER_SEC) << endl;
	}

	//使用自适应窗口
	else {
		clock_t time_ADaggregation_start, time_ADaggregation_end;
		time_ADaggregation_start = clock();
		Aggregation aggregation(phase_left, phase_right, color_threshold1, color_threshold2, color_threshold3, maxlength1, maxlength2);
		for (int i = 0; i < 2; i++) {//进行两次循环，一次正视差，一次负视差
			int d_length = 0;
			if (i == 0) {//第一次先进行负视差匹配
				d_length = fabs(min_disparity);
			}
			else {
				d_length = fabs(max_disparity);
			}
			for (int d = 0; d < d_length; d++) {
				vec_aggregation_maps[i][d] = aggregation.Aggregation2D(vec_cost_maps[i][d], true, i);
			}
		}
		time_ADaggregation_end = clock();
		std::cout << "自适应窗口聚算用时：" << double((time_ADaggregation_end - time_ADaggregation_start) / CLOCKS_PER_SEC) << endl;
	}
}

//计算右视图的DSI用来左右一致性检查
void CalculateRightDsi(vector<vector<Mat>>&vec_right_dsi, const vector<vector<Mat>>&vec_aggregation_maps, const Mat &phase_left, const Mat&phase_right)
{
	clock_t start, end;
	start = clock();
	//右视差图的DSI
	vec_right_dsi.resize(2);//存放一个负视差的，一个正视差的；
	vec_right_dsi[0].resize(abs(min_disparity) + 1);//最小视差数加一个0视差
	vec_right_dsi[1].resize(abs(max_disparity) + 1);//最大视差数加一个0视差
	for (int i = 0; i < vec_right_dsi.size(); i++) {
		for (int j = 0; j < vec_right_dsi[i].size(); j++) {
			vec_right_dsi[i][j] = Mat::zeros(phase_left.size(), CV_32FC1);//指定每个视差图的大小
		}
	}

	for (int i = 0; i < 2; i++) {//进行两次循环，一次正视差，一次负视差
		int d_length = 0;
		if (i == 0) {//第一次先进行负视差匹配
			d_length = fabs(min_disparity);
		}
		else {
			d_length = fabs(max_disparity);
		}
		for (int d = 0; d < d_length; d++) {
			for (int row = 0; row < phase_left.rows; row++) {
				for (int col = 0; col < phase_left.cols; col++) {//两次循环，对相位图进行逐像素
					if (phase_right.at<float>(row, col) < 0.0001) {
						continue;
					}
					if (i == 0) {//i=0是视差为负的情况
						if ((col - d > 0) && (col - d < phase_left.cols)) {
							if (phase_left.at<float>(row, col - d) < 0.000001) {//如果左边是无效区域，那么直接跳过
								continue;
							}
							vec_right_dsi[i][d].at<float>(row, col) = vec_aggregation_maps[i][d].at<float>(row, col - d);
						}
						else {
							continue;
						}
					}
					else {//i=1是视差为正的情况
						if ((col + d > 0) && (col + d < phase_left.cols)) {//保证仍处于图像区域内
							if (phase_left.at<float>(row, col + d) < 0.01) {//如果左边是无效区域，那么直接跳过
								continue;
							}
							vec_right_dsi[i][d].at<float>(row, col) = vec_aggregation_maps[i][d].at<float>(row, col + d);
						}
						else {
							continue;
						}
					}
				}
			}
		}
	}
	end = clock();
	std::cout << "计算右视图DSI用时：" << double((end - start) / CLOCKS_PER_SEC) << endl;
}


//由聚合的DSI计算视差
Mat DisparityCalculation(const vector<vector<Mat>>&vec_maps, const Mat &phase_left, const Mat &phase_right)
{
	Mat disparity;
	clock_t time_disparity_computation_start, time_disparity_computation_end;
	time_disparity_computation_start = clock();
	disparity = Mat::zeros(phase_left.size(), CV_32FC1);

	Mat cost_map = Mat::zeros(phase_left.size(), CV_32FC1);

	const int height = phase_left.rows;
	const int width = phase_left.cols;
	//视差计算
	int d_length = 0;
	int best_disparity = INVALID_DISPARITY;

	for (int row = 0; row < height; row++) {
		for (int col = 0; col < width; col++) {
			float cost_value = 1000000;
			if (phase_left.at<float>(row, col) < 0.01) {
				continue;
			}
			else {
				for (int i = 0; i < 2; i++) {
					if (i == 0) {
						d_length = abs(min_disparity);
					}
					else {
						d_length = abs(max_disparity);
					}
					for (int d = 0; d < d_length; d++) {
						if ((vec_maps[i][d].at<float>(row, col) < 0.00001)) {
							continue;
						}
						else {
							if (vec_maps[i][d].at<float>(row, col) < cost_value) {
								cost_value = vec_maps[i][d].at<float>(row, col);
								if (i == 0) {
									best_disparity = -d;
								}
								else {
									best_disparity = d;
								}
							}
						}
					}
				}
			}
			////////////////////////////////////////////////////////////////////////
			if (best_disparity == INVALID_DISPARITY) {//如果视差仍属于初始值，那么应该是位于无效区域
				continue;
			}
			disparity.at<float>(row, col) = best_disparity;//没哟进行视差精细化

/******************************************************视差精细化**********************************************/
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

			//else//进行亚像素视差精细化
			//{
			//	if (best_disparity == 0)//如果最佳视差为0，那么它位于正负视差交界处，需特殊处理
			//	{
			//		if (vec_maps[0].size() < 2)//如果没有负视差，那么直接赋值0
			//		{
			//			disparity.at<float>(row, col) =0.0;
			//			continue;
			//		}
			//		float c1 = vec_maps[0][1].at<float>(row, col);
			//		float c2 = vec_maps[0][0].at<float>(row, col);
			//		float c3 = vec_maps[1][1].at<float>(row, col);
			//		disparity.at<float>(row, col) = -(c3 - c1) / (c3 + c1 - 2 * c2) / 2;
			//	}
			//	else
			//	{
			//		if ((best_disparity > 0) && ((best_disparity + 1) < max_disparity))//正视差
			//		{
			//			float c1 = vec_maps[1][best_disparity - 1].at<float>(row, col);
			//			float c2 = vec_maps[1][best_disparity].at<float>(row, col);
			//			float c3 = vec_maps[1][best_disparity + 1].at<float>(row, col);
			//			disparity.at<float>(row, col) = best_disparity - (c3 - c1) / (c3 + c1 - 2 * c2) / 2;
			//		}
			//		else if ((best_disparity < 0) && (abs(best_disparity - 1) <abs( min_disparity)))//负视差
			//		{
			//			float c1 = vec_maps[0][abs(best_disparity -1)].at<float>(row, col);
			//			float c2 = vec_maps[0][abs(best_disparity)].at<float>(row, col);
			//			float c3 = vec_maps[0][abs(best_disparity + 1)].at<float>(row, col);
			//			disparity.at<float>(row, col) = best_disparity - (c3 - c1) / (c3 + c1 - 2 * c2) / 2;
			//		}
			//		else
			//		{
			//			disparity.at<float>(row, col) = best_disparity;
			//			continue;
			//		}

			//	}
			//}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/****************************************************视差精细化结束**************************************************************/
		}
	}
	time_disparity_computation_end = clock();
	std::cout << "从DSI得到视差图用时：" << double((time_disparity_computation_end - time_disparity_computation_start) / CLOCKS_PER_SEC) << endl;
	return disparity.clone();
}


//得到Rank_transform的等级
inline int Rank_value(float diff)
{
	int Crank = 0;
	if (diff <= u){
		if (diff > -u) {    //-u<diff<=u
			return 0;
		}else{
			if (diff <= -v)  {
				return -2;
			}
			else {                //-v<diff<=u
				return -1;
			}
		}
	}
	else {
		if (diff >= v){     //diff>=v
			return 2;
		}
		else {               // u<diff<v
			return 1;
		}
	}
}

/***********************************************AD-Rank-transform*************************************************************************/
inline float compute_ad_rank_cost(int left_row, int left_col, int right_row, int right_col, Mat phase_left, Mat phase_right, int win_size)
{
	//bool left_out = (left_row - win_size / 2 <= 0) || (left_row + win_size / 2 >= phase_left.rows) || (left_col - win_size / 2 <= 0) || (left_col + win_size / 2 >= phase_left.cols);
	//bool right_out = (right_row - win_size / 2 <= 0) || (right_row + win_size / 2 >= phase_right.rows) || (right_col - win_size / 2 <= 0) || (right_col + win_size / 2 >= phase_right.cols);
	//if (left_out || right_out)
	//{
	//	return 0;
	//}
	int rank_number = 0;
	float left_center = phase_left.at<float>(left_row, left_col);
	float right_center = phase_right.at<float>(right_row, right_col);

	float sad_diff = fabs(left_center - right_center);//AD值

	for (int row = -win_size / 2; row < win_size / 2; row++)
	{
		for (int col = -win_size / 2; col < win_size / 2; col++)
		{
			float temp_value_left = phase_left.at<float>(left_row + row, left_col + col);

			float temp_value_right = phase_right.at<float>(right_row + row, right_col + col);

			float diff_left = temp_value_left - left_center;
			float diff_right = temp_value_right - right_center;

			if (Rank_value(diff_left) != Rank_value(diff_right))
			{
				rank_number++;
			}
		}
	}
	/*float temp_cost = sad_diff / LMADA_AD;
	float temp_cost1 = temp_cost + (rank_number / LMADA_RANK);
	return temp_cost1;*/
	float temp_cost = 1 - exp(-sad_diff / LMADA_AD);
	temp_cost += 1 - exp(-rank_number / LMADA_RANK);
	return temp_cost;
}

inline float compute_ad_census_cost(int left_row, int left_col, int right_row, int right_col, Mat phase_left, Mat phase_right, int win_size)
{
	//bool left_out = (left_row - win_size / 2 <= 0) || (left_row + win_size / 2 >= phase_left.rows) || (left_col - win_size / 2 <= 0) || (left_col + win_size / 2 >= phase_left.cols);
	//bool right_out = (right_row - win_size / 2 <= 0) || (right_row + win_size / 2 >= phase_right.rows) || (right_col - win_size / 2 <= 0) || (right_col + win_size / 2 >= phase_right.cols);
	//if (left_out || right_out)
	//{
	//	return 0;
	//}
	int census_number = 0;
	float left_center = phase_left.at<float>(left_row, left_col);
	float right_center = phase_right.at<float>(right_row, right_col);

	float sad_diff = fabs(left_center - right_center);
	for (int row = -win_size / 2; row < win_size / 2; row++)
	{
		for (int col = -win_size / 2; col < win_size / 2; col++)
		{
			float temp_value_left = phase_left.at<float>(left_row + row, left_col + col);

			float temp_value_right = phase_right.at<float>(right_row + row, right_col + col);
			bool diff = (temp_value_left - left_center)*(temp_value_right - right_center) < 0;
			census_number += (diff) ? 1 : 0;
		}
	}
	float temp_cost = 1 - exp(-sad_diff / LMADA_AD);
	temp_cost += 1 - exp(-census_number / LMADA_CENSUS);
	return temp_cost;

}
