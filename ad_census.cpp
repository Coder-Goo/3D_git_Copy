#include"ad_census.h"


//������۵ĺ����������vector<vector<>>��DSI,���������������
void CostCalculation(vector<vector<Mat>>&vec_cost_maps, const Mat &phase_left, const Mat &phase_right, const uchar &cost_window_size)//�����Ҫ�ı�vector�е�������Ҫ�������ô���
{
	clock_t time_cost_computation_start, time_cost_computation_end;
	time_cost_computation_start = clock();
	const int height = phase_left.rows;
	const int width = phase_left.cols;
	//��ʱ������Ը�һ����ʼ�����������г�ʼ��
	vec_cost_maps.resize(2);//���һ�����Ӳ�ģ�һ�����Ӳ�ģ�
	vec_cost_maps[0].resize(abs(min_disparity) + 1);//��С�Ӳ�����һ��0�Ӳ�
	vec_cost_maps[1].resize(abs(max_disparity) + 1);//����Ӳ�����һ��0�Ӳ�

	for (int i = 0; i < vec_cost_maps.size(); i++) {
		for (int j = 0; j < vec_cost_maps[i].size(); j++) {
			vec_cost_maps[i][j] = Mat(phase_left.rows,phase_left.cols, CV_32FC1, DEFAULT_DISPARITY);
		}
	}
	int half_win_size = cost_window_size / 2;
	for (int i = 0; i < 2; i++)//��������ѭ����һ�����Ӳһ�θ��Ӳ�
	{
		int d_length = 0;
		if (i == 0) {//��һ���Ƚ��и��Ӳ�ƥ��
			d_length = fabs(min_disparity);
		}
		else {
			d_length = fabs(max_disparity);
		}
		for (int d = 0; d < d_length; d++) {
			//��������forѭ���õ�disparity space image��DSI��
			for (int row = half_win_size + 1; row < height - half_win_size - 1; row++) {
				const float* left_ptr = phase_left.ptr<float>(row);
				const float* right_ptr = phase_right.ptr<float>(row);
				float* cost_map_ptr = vec_cost_maps[i][d].ptr<float>(row);
				for (int col = half_win_size + 1; col < width - half_win_size - 1; col++) {
					
					if (left_ptr[col] < 0.01) { //�������λͼ������ֵΪ0������ƥ��
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
	std::cout << "���ۼ�����ʱ��" << double((time_cost_computation_end - time_cost_computation_start) / CLOCKS_PER_SEC) << endl;
}

//���۾ۺ�
void CostAggregation(bool fixed_window, vector<vector<Mat>>&vec_cost_maps, vector<vector<Mat>>&vec_aggregation_maps, const Mat &phase_left, const Mat &phase_right, const uchar &aggregation_window_size)
{
	/*cout << "����DSI�ĸ��Ӳ��С��"<<vec_cost_maps[0].size()<<endl;
	cout << "����DSI�����Ӳ��С��" << vec_cost_maps[1].size() << endl;*/
	/******************************************///��ʱ������Ը�һ����ʼ�����������г�ʼ��
	vec_aggregation_maps.resize(2);//���һ�����Ӳ�ģ�һ�����Ӳ�ģ�
	vec_aggregation_maps[0].resize(abs(min_disparity) + 1);//��С�Ӳ�����һ��0�Ӳ�
	vec_aggregation_maps[1].resize(abs(max_disparity) + 1);//����Ӳ�����һ��0�Ӳ�

	for (int i = 0; i < vec_cost_maps.size(); i++) {
		for (int j = 0; j < vec_cost_maps[i].size(); j++) {
			vec_aggregation_maps[i][j] = Mat(phase_left.rows, phase_left.cols, CV_32FC1, DEFAULT_DISPARITY);//ָ��ÿ���Ӳ�ͼ�Ĵ�С
		}
	}

	const int height = phase_left.rows;
	const int width = phase_left.cols;

	if (fixed_window == true) {	//ʹ�ù̶�����

		clock_t time_Faggregation_start, time_Faggregation_end;
		time_Faggregation_start = clock();
		for (int i = 0; i < 2; i++) {//��������ѭ����һ�����Ӳһ�θ��Ӳ�
			int d_length = 0;
			if (i == 0) {//��һ���Ƚ��и��Ӳ�ƥ��
				d_length = fabs(min_disparity);
			}
			else {
				d_length = fabs(max_disparity);
			}
			for (int d = 0; d < d_length; d++) {
				for (int row = aggregation_window_size / 2 + 1; row < height - aggregation_window_size / 2 - 1; row++) {
					for (int col = aggregation_window_size / 2 + 1; col < width - aggregation_window_size / 2 - 1; col++) {//����ѭ��������λͼ����������
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
		std::cout << "�̶����ڴ��۾ۺ���ʱ��" << double((time_Faggregation_end - time_Faggregation_start) / CLOCKS_PER_SEC) << endl;
	}

	//ʹ������Ӧ����
	else {
		clock_t time_ADaggregation_start, time_ADaggregation_end;
		time_ADaggregation_start = clock();
		Aggregation aggregation(phase_left, phase_right,phase_threshold_v, phase_threshold_h, length_v, length_h );
		for (int i = 0; i < 2; i++) {//��������ѭ����һ�����Ӳһ�θ��Ӳ�
			int d_length = 0;
			if (i == 0) {//��һ���Ƚ��и��Ӳ�ƥ��
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
		std::cout << "����Ӧ���ھ�����ʱ��" << double((time_ADaggregation_end - time_ADaggregation_start) / CLOCKS_PER_SEC) << endl;
	}
}

//��������ͼ��DSI��������һ���Լ��
void CalculateRightDsi(vector<vector<Mat>>&vec_right_dsi, const vector<vector<Mat>>&vec_aggregation_maps, const Mat &phase_left, const Mat&phase_right)
{
	clock_t start, end;
	start = clock();
	//���Ӳ�ͼ��DSI
	vec_right_dsi.resize(2);//���һ�����Ӳ�ģ�һ�����Ӳ�ģ�
	vec_right_dsi[0].resize(abs(min_disparity) + 1);//��С�Ӳ�����һ��0�Ӳ�
	vec_right_dsi[1].resize(abs(max_disparity) + 1);//����Ӳ�����һ��0�Ӳ�
	for (int i = 0; i < vec_right_dsi.size(); i++) {
		for (int j = 0; j < vec_right_dsi[i].size(); j++) {
			vec_right_dsi[i][j] = Mat::zeros(phase_left.size(), CV_32FC1);//ָ��ÿ���Ӳ�ͼ�Ĵ�С
		}
	}

	for (int i = 0; i < 2; i++) {//��������ѭ����һ�����Ӳһ�θ��Ӳ�
		int d_length = 0;
		if (i == 0) {//��һ���Ƚ��и��Ӳ�ƥ��
			d_length = fabs(min_disparity);
		}
		else {
			d_length = fabs(max_disparity);
		}
		for (int d = 0; d < d_length; d++) {
			for (int row = 0; row < phase_left.rows; row++) {
				for (int col = 0; col < phase_left.cols; col++) {//����ѭ��������λͼ����������
					if (phase_right.at<float>(row, col) < 0.0001) {
						continue;
					}
					if (i == 0) {//i=0���Ӳ�Ϊ�������
						if ((col - d > 0) && (col - d < phase_left.cols)) {
							if (phase_left.at<float>(row, col - d) < 0.000001) {//����������Ч������ôֱ������
								continue;
							}
							vec_right_dsi[i][d].at<float>(row, col) = vec_aggregation_maps[i][d].at<float>(row, col - d);
						}
						else {
							continue;
						}
					}
					else {//i=1���Ӳ�Ϊ�������
						if ((col + d > 0) && (col + d < phase_left.cols)) {//��֤�Դ���ͼ��������
							if (phase_left.at<float>(row, col + d) < 0.01) {//����������Ч������ôֱ������
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
	std::cout << "��������ͼDSI��ʱ��" << double((end - start) / CLOCKS_PER_SEC) << endl;
}


//�ɾۺϵ�DSI�����Ӳ�
Mat DisparityCalculation(const vector<vector<Mat>>&vec_maps, const Mat &phase_left, const Mat &phase_right)
{
	clock_t time_disparity_computation_start, time_disparity_computation_end;
	time_disparity_computation_start = clock();

	Mat disparity = Mat(phase_left.size(), CV_32FC1, DEFAULT_DISPARITY);
	Mat temp_cost_map = Mat(phase_left.rows, phase_left.cols, CV_32FC1, 10000);
	const int height = phase_left.rows;
	const int width = phase_left.cols;
	//�Ӳ����
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
	std::cout << "��DSI�õ��Ӳ�ͼ��ʱ��" << double((time_disparity_computation_end - time_disparity_computation_start) / CLOCKS_PER_SEC) << endl;
	return disparity.clone();
}
/******************************************************�Ӳϸ��**********************************************/
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

			//else//�����������Ӳϸ��
			//{
			//	if (best_disparity == 0)//�������Ӳ�Ϊ0����ô��λ�������Ӳ�紦�������⴦��
			//	{
			//		if (vec_maps[0].size() < 2)//���û�и��Ӳ��ôֱ�Ӹ�ֵ0
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
			//		if ((best_disparity > 0) && ((best_disparity + 1) < max_disparity))//���Ӳ�
			//		{
			//			float c1 = vec_maps[1][best_disparity - 1].at<float>(row, col);
			//			float c2 = vec_maps[1][best_disparity].at<float>(row, col);
			//			float c3 = vec_maps[1][best_disparity + 1].at<float>(row, col);
			//			disparity.at<float>(row, col) = best_disparity - (c3 - c1) / (c3 + c1 - 2 * c2) / 2;
			//		}
			//		else if ((best_disparity < 0) && (abs(best_disparity - 1) <abs( min_disparity)))//���Ӳ�
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
/****************************************************�Ӳϸ������**************************************************************/



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
