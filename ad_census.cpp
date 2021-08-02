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
			vec_cost_maps[i][j] = Mat(phase_left.rows,phase_left.cols, CV_32FC1, DEFAULT_DISPARITY);
		}
	}
	int half_win_size = cost_window_size / 2;
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
			for (int row = half_win_size + 1; row < height - half_win_size - 1; row++) {
				const float* left_ptr = phase_left.ptr<float>(row);
				const float* right_ptr = phase_right.ptr<float>(row);
				float* cost_map_ptr = vec_cost_maps[i][d].ptr<float>(row);
				for (int col = half_win_size + 1; col < width - half_win_size - 1; col++) {
					
					if (left_ptr[col] < 0.01) { //如果左相位图的像素值为0，跳过匹配
						continue;
					}
					else {
						int col_r = (i == 0 ? col + d : col - d);
						if (col_r <= half_win_size || col_r >= width - half_win_size)  break;
						else {

							if (right_ptr[col_r]< 0.01) {
								continue;
							}
							else {
									cost_map_ptr[col] = compute_ad_census_cost(row, col, row, col_r, phase_left, phase_right, cost_window_size);
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
			vec_aggregation_maps[i][j] = Mat(phase_left.rows, phase_left.cols, CV_32FC1, DEFAULT_DISPARITY);//指定每个视差图的大小
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
		Aggregation aggregation(phase_left, phase_right,phase_threshold_v, phase_threshold_h, length_v, length_h );
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
	clock_t time_disparity_computation_start, time_disparity_computation_end;
	time_disparity_computation_start = clock();

	Mat disparity = Mat(phase_left.size(), CV_32FC1, DEFAULT_DISPARITY);
	Mat temp_cost_map = Mat(phase_left.rows, phase_left.cols, CV_32FC1, 10000);
	const int height = phase_left.rows;
	const int width = phase_left.cols;
	//视差计算
	int d_length = 0;
	int best_disparity = INVALID_DISPARITY;


	const float* cost_ptr = nullptr;
	const float* left_phase_ptr = nullptr;
	float* temp_cost_ptr =nullptr;
	float* disparity_ptr = nullptr;

	for (int i = 0; i < 2; i++) {
		if (i == 0) d_length = abs(min_disparity);
		else d_length = abs(max_disparity);
		for (int d = 0; d <= d_length; d++) {
			for (int row = 0; row < height; row++) {

				cost_ptr = vec_maps[i][d].ptr<float>(row);
				left_phase_ptr = phase_left.ptr<float>(row);
				temp_cost_ptr = temp_cost_map.ptr<float>(row);
				disparity_ptr = disparity.ptr<float>(row);

				for (int col = 0; col < width; col++) {
					if (left_phase_ptr[col] < 0.001) continue;
					else {
						if (cost_ptr[col] < 0.01) continue;
						else {
							if (cost_ptr[col] < temp_cost_ptr[col]) {
								temp_cost_ptr[col] = cost_ptr[col];
								if (i == 0) disparity_ptr[col] = -d;
								else disparity_ptr[col] = d;
							}
						}
					}
				}
			}
		}
	}
	
	time_disparity_computation_end = clock();
	std::cout << "从DSI得到视差图用时：" << double((time_disparity_computation_end - time_disparity_computation_start) / CLOCKS_PER_SEC) << endl;
	return disparity.clone();
}
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



inline float compute_ad_census_cost(int &left_row, int &left_col, int& right_row, int& right_col, const Mat &phase_left,const  Mat &phase_right, const uchar &win_size)
{
	int census_number = 0;
	const float* left_center_ptr = phase_left.ptr<float>(left_row);
	const float* right_center_ptr = phase_right.ptr<float>(right_row);
	float sad_diff = fabs(left_center_ptr[left_col] - right_center_ptr[right_col]);
	for (int row = -win_size / 2; row < win_size / 2; row++)
	{
		const float* left_ptr = phase_left.ptr<float>(left_row + row);
		const float* right_ptr = phase_right.ptr<float>(right_row + row);

		for (int col = -win_size / 2; col < win_size / 2; col++)
		{
			bool diff = (left_ptr[left_col+col] - left_center_ptr[left_col])*(right_ptr[right_col+col] - right_center_ptr[right_col]) < 0;
			census_number += (diff) ? 1 : 0;
		}
	}
	float temp_cost = 1 - exp(-sad_diff / LMADA_AD);
	temp_cost += 1 - exp(-census_number / LMADA_CENSUS);
	return temp_cost;

}
