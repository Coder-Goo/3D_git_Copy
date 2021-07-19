
#include"calibration.h"
#include"phase_unwrapping.h"
#include"aggregation.h"
#include"phase_unwrapping2.h"

using namespace cv;
using namespace std;

Size read_file(ifstream & ifile, vector<vector<Point2f>>&imagePointVec, vector<vector<Point3f>>&realPointVec, Size&nei_jiaodian_number1, Size&square_size1);

//void computeSAD(Mat &left, Mat&right, vector<Point2f>&vecL, vector<Point2f>&vecR, Size winSize);
//void Q2pcd(Mat Q, string pcdFileName);
////pcl::PointCloud<pcl::PointXYZ>::Ptr phaseSad2Pcd(Mat phase_left, Mat phase_right, Mat Q);
//void phaseSad2Pcd(Mat phase_left, Mat phase_right, Mat Q); 
void phaseSad2PcdSadonly(Mat phase_left, Mat phase_right, Mat Q);
////第三版计算视差：SAD+CENSUS
// void phaseSadCensus2Pcd(Mat phase_left, Mat phase_right, int min_disparity, int max_disparity, Mat Q);
//
inline float compute_ad_census_cost(int left_row, int left_col, int right_row, int right_col, Mat phase_left, Mat phase_right, int win_size);
inline float compute_ad_rank_cost(int left_row, int left_col, int right_row, int right_col, Mat phase_left, Mat phase_right, int win_size);
float GetPreciseDisparity(int best_disparity, int row, int col);
float GetPreciseDisparityFromAD(Point2f x1, Point2f x2, Point2f x3);

//标定
int CalibrationImage();//标定
//计算相位图
int CalculatePhaseImage(Mat &phase_left, Mat &phase_right);
//计算代价
void CostCalculation(vector<vector<Mat>>&vec_cost_maps, const Mat &phase_left, const Mat &phase_right, const uchar &cost_window_size);
//计算左视图的DSI
void CalculateRightDsi(vector<vector<Mat>>&vec_right_dsi, const vector<vector<Mat>>&vec_aggregation_maps,const Mat& phase_left,const Mat &phase_right);
 //代价聚合
void CostAggregation(bool fixed_window,  vector<vector<Mat>>&vec_cost_maps, vector<vector<Mat>>&vec_aggregation_maps, const Mat &phase_left, const Mat &phase_right, const uchar &aggregation_window_size);
//视差计算
Mat DisparityCalculation(const vector<vector<Mat>>&vec_aggregation_maps, const Mat &phase_left, const Mat &phase_right);
//AD计算视差
Mat ADCalculateDisparity(const Mat &phase_left, const Mat &phase_right);
//由视差计算点云
void CalculatePointCloud(const Mat &disparity, const Mat &phase_left, const Mat &Q, String &file_name);



const int INVALID_DISPARITY = 6550;
 bool dsi =false;//如果dsi为true，那么就直接使用ad来计算，否则用dsi
 bool sad = false;
 bool ad_census = true;
 bool ad_rank = false;
 bool fixed_window = true;
 String pcd_file_name = "AD点云";

 const int cost_window_size = 7;
 const int aggeragation_window_size = 7;
 float LMADA_AD = 1.0;
 float  LMADA_CENSUS = 70.0;
 float LMADA_RANK = 10.0;

 //rank阈值
 float u = 0.2;
 float v = 0.35;//设置v=0.6,u=0.3

 //视差范围
 const int min_disparity = -130;
 const int max_disparity = 140;

 //自适应窗口阈值
 const double color_threshold1 = 0.6;
 const double color_threshold2 = 0.8;
 const double color_threshold3 = 0.2;
 const int maxlength1 = 9;
 const int maxlength2 = 5;//maxlength2<maxlength1且color_threshold1>color_threshold



string calib_file_name = "./Calibdata.xml";
 //string calib_file_name = "输出的内外参数.yml";


int main()
{
	Mat Q;//投影矩阵Q
	//CalibrationImage();//标定
	FileStorage fin(calib_file_name, FileStorage::READ);
	fin["Q"] >> Q;
	cout << Q << endl;

  	Mat phase_left,phase_right;//左右相位图
	vector<vector<Mat>>vec_cost_maps,vec_aggregation_maps,vec_aggregation_right_maps;
	Mat left_disparity,right_disparity;
	
	std::cout << CalculatePhaseImage(phase_left, phase_right) << endl;//解相位

	if(dsi)//使用DSI 计算视差图
	{
		CostCalculation(vec_cost_maps, phase_left, phase_right, cost_window_size);
		CostAggregation(fixed_window, vec_cost_maps, vec_aggregation_maps, phase_left, phase_right, aggeragation_window_size);
		vector<vector<Mat>>right_dsi;
		CalculateRightDsi(right_dsi, vec_aggregation_maps,phase_left,phase_right);
		left_disparity=DisparityCalculation( vec_aggregation_maps, phase_left, phase_right);
		//right_disparity= DisparityCalculation(right_dsi, phase_left, phase_right);
		int a = 0;
		
		CalculatePointCloud(left_disparity, phase_left, Q, pcd_file_name);
	}
	else//不使用DSI计算视差图
	{
		if (sad) {
			phaseSad2PcdSadonly(phase_left, phase_right, Q);
		}
		else {
			Mat disparity = ADCalculateDisparity(phase_left, phase_right);
			CalculatePointCloud(disparity, phase_left, Q, pcd_file_name);
		}
	}
	/*******************************************************************************************
	* 第一版输出点云的方法，使用视差和SAD来计算
	* 只使用SAD
	*
	*****************************************************/
	//phaseSad2PcdSadonly(L_phase, R_phase,Q);

	/***********************************************************************************
	*   第二版输出点云的方法，使用视差输出点云，视差经过优化，精度更高
	*   SAD+唯一性检测+LRC,没有亚像素精细化，因为经过左右一致性检测之后，视差图很稀疏，还没想好如何填充视差
	*
	**************************************************************************************/
	//phaseSad2Pcd(L_phase, R_phase, Q);
	//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

	/**************************************************************************************
	* 第三版输出点云的方法，
	* SAD+CENSUS
	******************************************************************************************/
	//phaseSadCensus2Pcd(L_phase, R_phase, -200, 200, Q);
	//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	

	fin.release();
	return 0;
}

//标定
int CalibrationImage()
{
	clock_t startTime, endTime;
	startTime = clock();
	/************************************************先读入标定图片***********************************************/
	ifstream left_file("./calibration_recify/标定图片索引/左相机的标定图片.txt");//输入的左相机的标定图片
	ifstream right_file("./calibration_recify/标定图片索引/右相机的标定图片.txt");//输入的右相机的标定图片
	vector<vector<Point2f>>image_point_vecL;//用来存储左相机检测的点
	vector<vector<Point2f>>image_point_vecR;//用来存储右相机检测的点
	Size nei_jiaodian_number = Size(qipan_cornor_x, qipan_cornor_y);//内角点数
	Size square_size = Size(qipan_size_x, qipan_size_y);//棋盘格大小
	vector<vector<Point3f>>object_point_vec;//用来存储棋盘的点
	if (!left_file)//判断是否正确读入标定图片
	{
	cout << "左相机图片输入流错误";
	return 1;
	}
	if (!right_file)
	{
	cout << "右相机图片输入流文档错误";
	return  1;
	}

	//使用cornerSubPix和drawCorner（这两个函数在read_file()中定义）来画出角点。得到三个vector向量用于标定

	 Size imageSize = read_file(left_file, image_point_vecL, object_point_vec, nei_jiaodian_number, square_size);
	 object_point_vec.clear();
	 imageSize = read_file(right_file, image_point_vecR, object_point_vec, nei_jiaodian_number, square_size);
	  endTime = clock();
	  cout << image_point_vecL.size() << endl;
	  cout << "read_file函数得到标定点用时：" << double((endTime - startTime) / CLOCKS_PER_SEC) << "s" << endl;
	 //cout << image_point_vecR.size()<< endl;
	//**********************************************下面进行标定**************************************
	 clock_t startTime1, endTime1;
	 startTime1 = clock();
	 Calibration calibration(calib_file_name,image_point_vecL,image_point_vecR,object_point_vec,imageSize);
	 calibration.calibration();//完成标定和矫正,输出标定参数到yml文件中。
     endTime1 = clock();
     cout << "标定用时：" << double((endTime1 - startTime1) / CLOCKS_PER_SEC) << "s" << endl;
	 return 0;
}
//计算相位图
int CalculatePhaseImage(Mat &phase_left ,Mat &phase_right)
{
	/********************************************下面进行相位图矫正***********************************/
//**********************************************先读入相位图片************************************
	ifstream left_phase("./phase_unwrapping/图片索引/左相机条纹图片.txt");//输入的相位图片
	ifstream right_phase("./phase_unwrapping/图片索引/右相机条纹图片.txt");//输入的相位图片
	if (!left_phase)
	{
		cout << "输入左图像错误" << endl;
		return 1;
	}
	if (!right_phase)
	{
		cout << "输入右图像错误" << endl;
		return 1;
	}
	/************************************************制作mask*********************************/
	Mat src1 = imread("./phase_unwrapping/IMAGES/L0.bmp");
	Mat src2 = imread("./phase_unwrapping/IMAGES/R0.bmp");
	Mat src11, src22;
	cvtColor(src1, src11, COLOR_BGR2GRAY);
	cvtColor(src2, src22, COLOR_BGR2GRAY);

	Mat src111, src222;
	threshold(src11, src111, 50, 255, THRESH_BINARY|THRESH_OTSU);
	threshold(src22, src222, 50, 255, THRESH_BINARY | THRESH_OTSU);


	/*********************************************************以前是用了开运算和闭运算，这样会使细节轮廓扩大或减小，8、*//////////////
	//对Mask进行开运算，消除噪点
	Mat mask_L;
	Mat mask_R;
	Mat element = getStructuringElement(MORPH_RECT, Size(5, 5));
	morphologyEx(src111, mask_L, MORPH_OPEN, element);
	morphologyEx(src222, mask_R, MORPH_OPEN, element);
	imwrite("./phase_unwrapping/mask_l.bmp", mask_L);
	imwrite("./phase_unwrapping/mask_R.bmp", mask_R);
	//imshow("mask_L", mask_L);//试验过这两个mask没有问题
	//imshow("mask_R", mask_R);
	//waitKey(50);
	//getchar();

	//制作mask 
	/*******************************制作mask结束*******************************************************************/

	/******************************读入相位图片****************************************************************/
	vector<Mat>left_img;
	vector<Mat>right_img;
	while (!left_phase.eof())
	{
		string str;
		getline(left_phase, str);
		Mat src = imread(str);
		Mat srcTepL;
		/*imshow("输入的图片", src);
		waitKey();*/
		src.copyTo(srcTepL, mask_L);
		/*imshow("mask之后的图片", srcTepL2);
		waitKey();*/
		left_img.push_back(srcTepL.clone());
		string str_right;
		getline(right_phase, str_right);
		Mat dst = imread(str_right);
		Mat srcTepR;
		dst.copyTo(srcTepR, mask_R);
		right_img.push_back(srcTepR.clone());
	}

	if (right_img.size() != 12)
	{
		cout << "vector中读入的图片数量错误" << endl;
		return 1;
	}
	if (left_img.empty())
	{
		cout << "左相机相位图片vector为空" << endl;
		return 1;
	}
	/********************************************下面开始矫正***************************************************/
	clock_t startTime2, endTime2;

	Mat map11, map12, map21, map22;
	FileStorage fin(calib_file_name, FileStorage::READ);
	fin["map11"] >> map11;
	fin["map12"] >> map12;
	fin["map21"] >> map21;
	fin["map22"] >> map22;
	startTime2 = clock();
	vector<Mat>leftImg;
	vector<Mat>rightImg;
	leftImg.resize(12);
	rightImg.resize(12);
	for (int i = 0; i < left_img.size(); i++)
	{
		remap(left_img[i], leftImg[i], map11, map12, INTER_LINEAR);
		remap(right_img[i], rightImg[i], map21, map22, INTER_LINEAR);
	}
	Mat mask_L1, mask_R1;
	remap(mask_L, mask_L1, map11, map12, INTER_LINEAR);
	remap(mask_R, mask_R1, map21, map22, INTER_LINEAR);

	endTime2 = clock();
	cout << "rectify矫正用时：" << double((endTime2 - startTime2) / CLOCKS_PER_SEC) << "s" << endl;

	//***********************画极线判断是否成功矫正**********************************************************//

	
	/*Mat src_tep1 = leftImg[1];
	Mat src_tep2 = rightImg[1];
	imshow("src1", src_tep1);
	imshow("src2", src_tep2);
	Size halfSize = Size(src_tep1.cols / 2, src_tep1.rows / 2);
	Mat src_half, src_half2;
	resize(src_tep1, src_half, halfSize);
	resize(src_tep2, src_half2, halfSize);

	Mat dst(Size(src_half.cols+ src_half2.cols, src_half.rows), CV_32FC3);
	src_half.colRange(0, src_half.cols).copyTo(dst.colRange(0, src_half.cols));
	src_half2.colRange(0, src_half2.cols).copyTo(dst.colRange(src_half.cols, dst.cols));

	for (int i = 20; i < dst.rows; i += 16)
	{
		line(dst,Point(1, i), Point(dst.cols,i), Scalar(0, 255, 00), 1, 8, 0);
	}
	Mat dst1111;
	dst.convertTo(dst1111, CV_8U);
	imshow("dst", dst1111);
	waitKey(500);
	cout << "成功输出极线" << endl;*/

	/************************************画极线结束**************************************************/


	/*******************************************下面开始解相位*******************************************/
	clock_t startTime3, endTime3;
 	startTime3 = clock();

	//自己写的解相方法
	/*Phase_Unwrapping L_phase_wrap(leftImg, mask_L1);
	Mat L_phase = L_phase_wrap.phase_unwrapping();
	Phase_Unwrapping R_phase_wrap(rightImg, mask_R1);
	Mat R_phase = R_phase_wrap.phase_unwrapping();*/

	//师兄的解相方法
	PhaseUnwrapping2 L_phase_wrap;
    Mat L_phase = L_phase_wrap.PhaseUnwrapping3Frequency4step(leftImg, mask_L1);
	PhaseUnwrapping2 R_phase_wrap;
	Mat R_phase = R_phase_wrap.PhaseUnwrapping3Frequency4step(rightImg, mask_R1);

	endTime3 = clock();
	cout << "解相位用时：" << double((endTime3 - startTime3) / CLOCKS_PER_SEC) << "s" << endl;
	/***********************************用来显示看是否正确,得先把图片变为uint类型*************************************/

	FileStorage fout("C:\\Program Files\\MATLAB\\R2016b\\bin\\StereoMatching/phase_left.xml", FileStorage::WRITE);
	fout <<"L_phase"<< L_phase;
	fout.release();
	FileStorage fout1("C:\\Program Files\\MATLAB\\R2016b\\bin\\StereoMatching/phase_right.xml", FileStorage::WRITE);
	fout1 <<"R_phase"<< R_phase;
	fout1.release();

	//Mat L_phase1, R_phase1;
	//L_phase.convertTo(L_phase1, CV_8U);

	//R_phase.convertTo(R_phase1, CV_8U);

	//imshow("L_phase", L_phase1);
	//waitKey(500);
	//imshow("R_phase", R_phase1);
	//waitKey(500);

	//int a = 0;
/*************************************************************写出相位图***************************************/
/**********************************用sobel算子看看相位图的梯度************************************************/
//for (int row = 0; row < L_phase.rows; row++)
//{
//	for (int col = 0; col < L_phase.cols; col++)
//	{
//		if (L_phase.at<float>(row, col) < 0.01)
//		{
//			L_phase.at<float>(row, col) == 0;
//		}
//	}
//}
//for (int row = 0; row < L_phase.rows; row++)
//{
//	for (int col = 0; col < L_phase.cols; col++)
//	{
//		if (R_phase.at<float>(row, col) < 0.01)
//		{
//			R_phase.at<float>(row, col) == 0;
//		}
//	}
//}
//Mat grad_x, grad_y;
//Sobel(L_phase, grad_x, -1, 1, 0, 3, 1, 1, BORDER_DEFAULT);
//Sobel(R_phase, grad_y, -1, 1, 0, 3, 1, 1, BORDER_DEFAULT);

//int a = 0;
/*************************************************************************************************************/

/*imwrite("D:\\VS文件\\匹配main程序\\匹配main程序\\L_phase.bmp", L_phase1);
imwrite("D:\\VS文件\\匹配main程序\\匹配main程序\\R_phase.bmp", R_phase1);*/
	if (L_phase.depth() == CV_64FC1) {
		L_phase.convertTo(phase_left, CV_32FC1);
		R_phase.convertTo(phase_right, CV_32FC1);
		return 0;
	}
	phase_left = L_phase.clone();
	phase_right = R_phase.clone();
	return 0;
}

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

	for (int i = 0; i < vec_cost_maps.size(); i++){
		for (int j = 0; j < vec_cost_maps[i].size(); j++){
			vec_cost_maps[i][j] = Mat::zeros(phase_left.size(), CV_32FC1);//指定每个视差图的大小
		}
	}

	for (int i = 0; i < 2; i++)//进行两次循环，一次正视差，一次负视差
	{
		int d_length = 0;
		if (i == 0){//第一次先进行负视差匹配
			d_length = fabs(min_disparity);
		}
		else{
			d_length = fabs(max_disparity);
		}
		for (int d = 0; d < d_length; d++){
			//下面三次for循环得到disparity space image（DSI）
			for (int row = cost_window_size / 2 + 1; row < height - cost_window_size / 2 - 1; row++){
				for (int col = cost_window_size / 2 + 1; col < width - cost_window_size / 2 - 1; col++){//两次循环，对相位图进行逐像素
					if (phase_left.at<float>(row, col) < 0.01){//如果左相位图的像素值为0，跳过匹配
						continue;
					}
					else{
						int col_r;
						if (i == 0){
							col_r = col + d;//负视差的时候，右图视差和左图视差之间的关系
						}
						else{
							col_r = col - d;//正视差的时候，右图视差和左图视差之间的关系
						}
						if (col_r >= (width - cost_window_size / 2) || col_r <= cost_window_size / 2){
							continue;
						}
						else{
							if (phase_right.at<float>(row, col_r) < 0.01){
								continue;
							}
							else{
								if (ad_census){//使用ad_census进行代价计算	
									vec_cost_maps[i][d].at<float>(row, col) = compute_ad_census_cost(row, col, row, col_r, phase_left, phase_right, cost_window_size);
								}
								if (ad_rank){//使用ad_rank进行代价计算
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
void CostAggregation(bool fixed_window,  vector<vector<Mat>>&vec_cost_maps,vector<vector<Mat>>&vec_aggregation_maps,const Mat &phase_left,const Mat &phase_right,const uchar &aggregation_window_size)
{
	/*cout << "代价DSI的负视差大小是"<<vec_cost_maps[0].size()<<endl;
	cout << "代价DSI的正视差大小是" << vec_cost_maps[1].size() << endl;*/
	/******************************************///到时这里可以搞一个初始化函数来进行初始化
	vec_aggregation_maps.resize(2);//存放一个负视差的，一个正视差的；
	vec_aggregation_maps[0].resize(abs(min_disparity) + 1);//最小视差数加一个0视差
	vec_aggregation_maps[1].resize(abs(max_disparity) + 1);//最大视差数加一个0视差

	for (int i = 0; i < vec_cost_maps.size(); i++){
		for (int j = 0; j < vec_cost_maps[i].size(); j++){
			vec_aggregation_maps[i][j] = Mat::zeros(phase_left.size(), CV_32FC1);//指定每个视差图的大小
		}
	}

	const int height = phase_left.rows;
	const int width = phase_left.cols;

	if (fixed_window == true){	//使用固定窗口

		clock_t time_Faggregation_start, time_Faggregation_end;
		time_Faggregation_start = clock();
		for (int i = 0; i < 2; i++){//进行两次循环，一次正视差，一次负视差
			int d_length = 0;
			if (i == 0){//第一次先进行负视差匹配
				d_length = fabs(min_disparity);
			}
			else{
				d_length = fabs(max_disparity);
			}
			for (int d = 0; d < d_length; d++){
				for (int row = aggregation_window_size / 2 + 1; row < height - aggregation_window_size / 2 - 1; row++){
					for (int col = aggregation_window_size / 2 + 1; col < width - aggregation_window_size / 2 - 1; col++){//两次循环，对相位图进行逐像素
						if (vec_cost_maps[i][d].at<float>(row, col) < 0.01){
							continue;
						}
							else{
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
	else{
		clock_t time_ADaggregation_start, time_ADaggregation_end;
		time_ADaggregation_start = clock();
		Aggregation aggregation(phase_left, phase_right, color_threshold1, color_threshold2, color_threshold3, maxlength1, maxlength2);
		for (int i = 0; i < 2; i++){//进行两次循环，一次正视差，一次负视差
			int d_length = 0;
			if (i == 0){//第一次先进行负视差匹配
				d_length = fabs(min_disparity);
			}
			else{
				d_length = fabs(max_disparity);
			}
			for (int d = 0; d < d_length; d++){
				vec_aggregation_maps[i][d]=aggregation.Aggregation2D(vec_cost_maps[i][d],true, i);
			}
		}
		time_ADaggregation_end = clock();
		std::cout << "自适应窗口聚算用时：" << double((time_ADaggregation_end - time_ADaggregation_start) / CLOCKS_PER_SEC) << endl;
	}
}

//计算右视图的DSI用来左右一致性检查
void CalculateRightDsi(vector<vector<Mat>>&vec_right_dsi,const vector<vector<Mat>>&vec_aggregation_maps,const Mat &phase_left,const Mat&phase_right)
{
	clock_t start, end;
	start = clock();
	//右视差图的DSI
	vec_right_dsi.resize(2);//存放一个负视差的，一个正视差的；
	vec_right_dsi[0].resize(abs(min_disparity) + 1);//最小视差数加一个0视差
	vec_right_dsi[1].resize(abs(max_disparity) + 1);//最大视差数加一个0视差
	for (int i = 0; i < vec_right_dsi.size(); i++){
		for (int j = 0; j < vec_right_dsi[i].size(); j++){
			vec_right_dsi[i][j] = Mat::zeros(phase_left.size(), CV_32FC1);//指定每个视差图的大小
		}
	}

	for (int i = 0; i < 2; i++){//进行两次循环，一次正视差，一次负视差
		int d_length = 0;
		if (i == 0){//第一次先进行负视差匹配
			d_length = fabs(min_disparity);
		}
		else{
			d_length = fabs(max_disparity);
		}
		for (int d = 0; d < d_length; d++){
			for (int row = 0 ; row < phase_left.rows; row++){
				for (int col = 0; col < phase_left.cols; col++){//两次循环，对相位图进行逐像素
					if (phase_right.at<float>(row, col) < 0.0001){
						continue;
					}
					if (i == 0){//i=0是视差为负的情况
						if ((col - d > 0) && (col - d < phase_left.cols)){
							if (phase_left.at<float>(row, col - d)<0.000001){//如果左边是无效区域，那么直接跳过
								continue;
							}
							vec_right_dsi[i][d].at<float>(row, col) = vec_aggregation_maps[i][d].at<float>(row, col - d);
						}
						else{
							continue;
						}
					}
					else {//i=1是视差为正的情况
						if ((col + d > 0) && (col + d < phase_left.cols)){//保证仍处于图像区域内
							if (phase_left.at<float>(row, col + d) < 0.01){//如果左边是无效区域，那么直接跳过
								continue;
							}
							vec_right_dsi[i][d].at<float>(row, col) = vec_aggregation_maps[i][d].at<float>(row, col + d);
						}
						else{
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
Mat DisparityCalculation(const vector<vector<Mat>>&vec_maps,const Mat &phase_left,const Mat &phase_right)
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

	for (int row = 0; row < height; row++){
		for (int col = 0; col < width; col++){
			float cost_value = 1000000;
			if (phase_left.at<float>(row, col) < 0.01){
				continue;
			}
			else{
				for (int i = 0; i < 2; i++){
					if (i == 0){
						d_length = abs(min_disparity);
					}
					else{
						d_length = abs(max_disparity);
					}
					for (int d = 0; d < d_length; d++){
						if ((vec_maps[i][d].at<float>(row, col) < 0.00001)){
							continue;
						}
						else{
							if (vec_maps[i][d].at<float>(row, col) < cost_value){
								cost_value = vec_maps[i][d].at<float>(row, col);
								if (i == 0){
									best_disparity = -d;
								}
								else{
									best_disparity = d;
								}
							}
						}
					}
				}
			}
			////////////////////////////////////////////////////////////////////////
			if (best_disparity == INVALID_DISPARITY){//如果视差仍属于初始值，那么应该是位于无效区域
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


Mat ADCalculateDisparity( const Mat &phase_left,const  Mat &phase_right)
{
	clock_t time_start, time_end;
	time_start = clock();
	Mat disparity (phase_left.rows, phase_left.cols, CV_32FC1, DEFAULT_DISPARITY);
  	int height = phase_left.rows;
	int width = phase_left.cols;
	int X1, X2;
	double cost = 10000;
	double temp = 10000;
	//使用ptr来访问像素
	for (int i = 0; i < height; i++) {
		const float* left_ptr = phase_left.ptr<float>(i);
		const float* right_ptr = phase_right.ptr<float>(i);
		float *disp_ptr = disparity.ptr<float>(i);
		for (int j = 0; j < width; j++) {
			if (left_ptr[j] < 0.001) continue;
			X1 = j;
			X2 = 0;
			cost = 10000;
			for (int k = 0; k < width; k++) {
				if (right_ptr[k] > 0.001) {
					temp = fabs(left_ptr[j] - right_ptr[k]);
					if (temp < cost) {
						cost = temp;
						X2 = k;
					}
				}
			}
			if (X2 != 0 && cost < 0.3) {
				disp_ptr[j] = X1 - X2;
			}
		}
	}
	time_end = clock();
	std::cout << "从AD计算得到视差图用时：" << double((time_end - time_start) / CLOCKS_PER_SEC) << endl;
	return disparity;
}

//亚像素精细化
float GetPreciseDisparityFromAD(Point2f x1, Point2f x2, Point2f x3)
{
	float c1 = x1.y;
	float c2 = x2.y;
	float c3 = x3.y;

	float d1 = x1.x;
	float d2 = x2.x;
	float d3 = x3.x;

	float res = d2 - (c3 - c1) / (c3 + c1 - 2 * c2) / 2;
	return res;
}



//从视差图计算点云
void CalculatePointCloud(const Mat &disparity,const Mat & phase_left,const Mat &Q,String &file_name)
{
	pcl::PointCloud<pcl::PointXYZ>points_cloud;
	clock_t time_2piont_cloud_start, time_2point_cloud_end;
	time_2piont_cloud_start = clock();

	const int height = disparity.rows;
	const int width = disparity.cols;

	//下面是由视差图计算点云
	for (int col = 0; col < width; col++)
	{
		for (int row = 0; row < height; row++)
		{
			if (phase_left.at<float>(row, col) < 0.01 || disparity.at<float>(row, col) == DEFAULT_DISPARITY)
			{
				continue;
			}
			else
			{
				//计算点云XYZ
				pcl::PointXYZ tep_points;
				Vec4d xyd1 = Vec4d(col, row, disparity.at<float>(row, col), 1);
				Matx44d _Q;
				Q.convertTo(_Q, CV_64F);
				Vec4d XYZW = _Q * xyd1;
				tep_points.x = XYZW[0] / XYZW[3];
				tep_points.y = XYZW[1] / XYZW[3];
				tep_points.z = XYZW[2] / XYZW[3];
				//points_cloud->points.push_back(tep_points);
				points_cloud.points.push_back(tep_points);
			}
		}
	}
	time_2point_cloud_end = clock();
	cout << "从视差图得到点云图用时：" << double((time_2point_cloud_end - time_2piont_cloud_start) / CLOCKS_PER_SEC);
	points_cloud.width = points_cloud.points.size();
	points_cloud.height = 1;
	pcl::io::savePCDFile(file_name +".pcd", points_cloud);
}


//亚像素视差精细化
//float GetPreciseDisparity(int best_disparity,int row,int col )
//{
//		float precise_disparity;
//		int i;
//		if (best_disparity == 0)//如果最佳视差为0，那么它位于正负视差交界处，需特殊处理
//		{
//			float c1 = vec_aggregation_maps[0][1].at<float>(row, col);
//			float c2 = vec_aggregation_maps[0][0].at<float>(row, col);
//			float c3 = vec_aggregation_maps[1][1].at<float>(row, col);
//			precise_disparity = -(c3 - c1) / (c3 + c1 - 2 * c2) / 2;
//		}
//		else
//		{
//			bool temp = (best_disparity > 0);
//			switch (temp)
//			{
//			case true:
//				i = 1;
//			case false:
//				i = 0;
//			default:
//				cout << "视差精细化错误" << endl;
//			}
//			float c1 = vec_aggregation_maps[i][best_disparity - 1].at<float>(row, col);
//			float c2 = vec_aggregation_maps[i][best_disparity].at<float>(row, col);
//			float c3 = vec_aggregation_maps[i][best_disparity + 1].at<float>(row, col);
//			precise_disparity = best_disparity - (c3 - c1) / (c3 + c1 - 2 * c2) / 2;
//		}
//
//		return precise_disparity;
//}

///**********************************************************************************************************
//* 第三版计算视差的函数 ，代价计算使用SAD+CENSUS
//* 对左右相位图先代价计算—>取窗口聚合—>取左右差值最小得粗视差—>唯一性检测—>左右一致性检查得到LRCZ_map_disparity_right—>输出点云
//*function: 从相位图中计算视差，然后由视差得到点云数据
//* input：   phase1——左相机绝对相位图
//*           phase2——右相机绝对相位图
//* output：  输出点云XYZ数据坐标
//* history:  2020,7.26
//***********************************************************************************************************/
////pcl::PointCloud<pcl::PointXYZ>::Ptr phaseSad2Pcd(Mat phase_left, Mat phase_right, Mat Q)
void phaseSadCensus2Pcd(Mat phase_left, Mat phase_right, int min_disparity, int max_disparity, Mat Q)
{
	/*Mat cost_map_left = Mat::zeros(phase_left.size(), CV_32FC1);
	Mat cost_map_right = Mat::zeros(phase_left.size(), CV_32FC1);*/
	const int width = phase_left.cols;
	const int height = phase_left.rows;
	int sad_win_size = 11;

	Mat map_disparity_left = Mat::zeros(phase_left.size(), CV_32FC1);//视差图
	//Mat map_disparity_right = Mat::zeros(phase_left.size(), CV_32FC1);//视差图
	//Mat LRC_map_disparity_right = Mat::zeros(phase_left.size(), CV_32FC1);//;左右一致性检查后的视差图

	//pcl::PointCloud<pcl::PointXYZ>::Ptr points_cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>points_cloud;

	vector<vector<Mat>>vec_cost_disparity_maps;//DSI
	vec_cost_disparity_maps.resize(2);//存放一个负视差的，一个正视差的；
	vec_cost_disparity_maps[0].resize(abs(min_disparity) + 1);//最小视差数加一个0视差
	vec_cost_disparity_maps[1].resize(abs(max_disparity) + 1);//最大视差数加一个0视差
	
	for(int i = 0; i < vec_cost_disparity_maps.size(); i++)
	{
		for (int j = 0; j < vec_cost_disparity_maps[i].size(); j++)
		{
			vec_cost_disparity_maps[i][j]=Mat::zeros(phase_left.size(), CV_32FC1);//指定每个视差图的大小
		}
	}

	for (int row = 0; row < height; row++)
	{
		for (int col = 0; col < width; col++)
		{
			if (phase_left.at<float>(row, col) < 0.01)
			{
				phase_left.at<float>(row, col) = 0;
			}
		}
	}

	for (int row = 0; row < height; row++)
	{
		for (int col = 0; col < width; col++)
		{
			if (phase_right.at<float>(row, col) < 0.01)
			{
				phase_right.at<float>(row, col) = 0;
			}
		}
	}

	/***********************************************代价计算*********************************************************************/
	clock_t time_cost_computation_start, time_cost_computation_end;
	time_cost_computation_start = clock();
	//先进行代价计算：
	for (int i = 0; i < 2; i++)//进行两次循环，一次正视差，一次负视差
	{
		int d_length = 0;
		if (i == 0)//第一次先进行负视差匹配
		{
			d_length = fabs(min_disparity);
		}
		else
		{
			d_length = fabs(max_disparity);
		}
		for (int d = 0; d < d_length; d++)
		{
			//下面三次for循环得到disparity space image（DSI）
			for (int row = sad_win_size / 2+1; row < height - sad_win_size / 2-1; row++)
			{
				for (int col = sad_win_size / 2+1; col < width - sad_win_size / 2-1; col++)//两次循环，对相位图进行逐像素
				{

					if (phase_left.at<float>(row, col) < 0.01)//如果左相位图的像素值为0，跳过匹配
					{
						continue;
					}
					else
					{

						int col_r;

						if (i == 0)
						{
							col_r = col + d;//负视差的时候，右图视差和左图视差之间的关系
						}
						else
						{
							col_r = col - d;//正视差的时候，右图视差和左图视差之间的关系
						}
						int aaaa = row;
						int bbbb = col;
						if (phase_right.at<float>(row, col_r) < 0.01)
						{
							continue;
						}
						else
						{
							vec_cost_disparity_maps[i][d].at<float>(row, col) = compute_ad_census_cost(row, col, row, col_r, phase_left, phase_right, sad_win_size);
						/*	float aa = compute_sad_census_cost(row, col, row, col_r, phase_left, phase_right, sad_win_size);
							int lll = 0;*/
						}

					}
				}

			}

		}
		
	}
	time_cost_computation_end = clock();
	cout << "代价计算用时：" << double((time_cost_computation_end - time_cost_computation_start) / CLOCKS_PER_SEC)<<endl;
	/*********************************************************************代价计算结束***************************************************/
	
	Mat dst01 = vec_cost_disparity_maps[0][0];
	Mat dst02 = vec_cost_disparity_maps[0][20];
	Mat dst03 = vec_cost_disparity_maps[0][30];
	Mat dst04 = vec_cost_disparity_maps[0][40];
	Mat dst05 = vec_cost_disparity_maps[0][50];
	Mat dst06 = vec_cost_disparity_maps[0][60];
	Mat dst07 = vec_cost_disparity_maps[0][70];
	Mat dst08 = vec_cost_disparity_maps[0][80];
	Mat dst09 = vec_cost_disparity_maps[0][90];
	Mat dst10 = vec_cost_disparity_maps[0][95];


	Mat dst11 = vec_cost_disparity_maps[1][10];
	Mat dst12 = vec_cost_disparity_maps[1][20];
	Mat dst13 = vec_cost_disparity_maps[1][30];
	Mat dst14 = vec_cost_disparity_maps[1][40];
	Mat dst15 = vec_cost_disparity_maps[1][50];
	Mat dst16 = vec_cost_disparity_maps[1][60];
	Mat dst17 = vec_cost_disparity_maps[1][70];
	Mat dst18 = vec_cost_disparity_maps[1][80];
	Mat dst19 = vec_cost_disparity_maps[1][90];
	Mat dst20 = vec_cost_disparity_maps[1][95];
 	int a1111 = 0;

	/*********************************计算积分图*****************************************************88*/
	//积分图的定义
	clock_t time_integral_image_start, time_integral_image_end;
	time_integral_image_start = clock();

	vector<vector<Mat>>integral_images;//积分图矩阵
	integral_images.resize(2);//存放一个负视差的，一个正视差的；因为vector下标不支持负数的索引
	integral_images[0].resize(abs(min_disparity) + 1);//最小视差数加一个0视差
	integral_images[1].resize(abs(max_disparity) + 1);//最大视差数加一个0视差
	for (int i = 0; i < integral_images.size(); i++)
	{
		for (int j = 0; j < integral_images[i].size(); j++)
		{
			integral_images[i][j] = Mat::zeros(phase_left.size(), CV_32FC1);//指定每个积分图图的大小和输入相位图一样
		}
	}

	for (int i = 0; i < 2; i++)//进行两次循环，一次正视差，一次负视差
	{
		int d_length = 0;
		if (i == 0)//第一次先进行负视差匹配
		{
			d_length = fabs(min_disparity);
		}
		else
		{
			d_length = fabs(max_disparity);
		}
		for (int d = 0; d < d_length; d++)
		{
			//下面三次for循环得到积分图
			for (int row = sad_win_size / 2 + 1; row < height - sad_win_size / 2 - 1; row++)
			{
				for (int col = sad_win_size / 2 + 1; col < width - sad_win_size / 2 - 1; col++)//两次循环，对相位图进行逐像素
				{
					//第一步，计算水平积分图
					//第二步，根据水平积分图，根据q点的水平臂长先计算q点的水平代价聚合E
					//第三步，把水平代价聚合当做输入，得到竖直积分图S
					//第四步，由竖直积分图S得到代价值
						
				}

			}

		}

	}
	time_integral_image_end = clock();
	cout << "计算积分图用时：" << double((time_integral_image_end - time_integral_image_start) / CLOCKS_PER_SEC) << endl;
	

	/******************************************************************************************8*/
	
	//代价聚合矩阵的定义
	vector<vector<Mat>>aggregate_vec_cost_disparity_maps;//DSI
	aggregate_vec_cost_disparity_maps.resize(2);//存放一个负视差的，一个正视差的；因为vector下标不支持负数的索引
	aggregate_vec_cost_disparity_maps[0].resize(abs(min_disparity) + 1);//最小视差数加一个0视差
	aggregate_vec_cost_disparity_maps[1].resize(abs(max_disparity) + 1);//最大视差数加一个0视差
	for (int i = 0; i < aggregate_vec_cost_disparity_maps.size(); i++)
	{
		for (int j = 0; j < aggregate_vec_cost_disparity_maps[i].size(); j++)
		{
			aggregate_vec_cost_disparity_maps[i][j] = Mat::zeros(phase_left.size(), CV_32FC1);//指定每个视差图的大小
		}
	}



	///////***************888888888888888888888888888888888888888888**********************************************8888888888888888888888888888888
	/////*********************************************************************第二种聚合方法，使用自适应窗口*******************************************/
	clock_t time_aggregation_start, time_aggregation_end;
	time_aggregation_start = clock();
	Aggregation aggregation(phase_left, phase_right, 1.2, 0.6, 0.3,11, 7);
	for (int i = 0; i < 2; i++)//进行两次循环，一次正视差，一次负视差
	{
		int d_length = 0;
		if (i == 0)//第一次先进行负视差聚合
		{
			d_length = fabs(min_disparity);
		}
		else
		{
			d_length = fabs(max_disparity);
		}
		//下面三次for循环得到分别对视差进行聚合
		for (int d = 0; d < d_length; d++)
		{
			for (int row = sad_win_size / 2; row < height - sad_win_size / 2; row++)
			{
				for (int col = sad_win_size / 2; col < width - sad_win_size / 2; col++)//两次循环，对相位图进行逐像素聚合
				{
					if (vec_cost_disparity_maps[i][d].at<float>(row, col) < 0.000001)//如果左相位图的像素值为0，跳过匹配
					{
						continue;
					}
					else
					{
						aggregation.Aggregation2D(vec_cost_disparity_maps[i][d], true, 0);

					}

				}

			}
		}
	}
	time_aggregation_end = clock();
	cout << "代价聚合用时：" << double((time_aggregation_end - time_aggregation_start) / CLOCKS_PER_SEC);
	///////***************************************************88888888888888888888888888888888第二种聚合方法到此结束******************************************************************************************


	///////***************888888888888888888888888888888888888888888**********************************************8888888888888888888888888888888
	/////*********************************************************************第一种聚合方法，使用固定窗口*******************************************/
	//for (int i = 0; i < 2; i++)//进行两次循环，一次正视差，一次负视差
	//{
	//	int d_length = 0;
	//	if (i == 0)//第一次先进行负视差聚合
	//	{
	//		d_length = fabs(min_disparity);
	//	}
	//	else
	//	{
	//		d_length = fabs(max_disparity);
	//	}
	//	//下面三次for循环得到分别对视差进行聚合
	//	for (int d = 0; d < d_length; d++)
	//	{
	//		for (int row = sad_win_size / 2; row < height - sad_win_size / 2; row++)
	//		{
	//			for (int col = sad_win_size / 2; col < width - sad_win_size / 2; col++)//两次循环，对相位图进行逐像素聚合
	//			{
	//				if (vec_cost_disparity_maps[i][d].at<float>(row, col) < 0.000001)//如果左相位图的像素值为0，跳过匹配
	//				{
	//					continue;
	//				}
	//				else
	//				{
	//					Mat kernel = vec_cost_disparity_maps[i][d](Rect(col - sad_win_size / 2, row - sad_win_size / 2, sad_win_size, sad_win_size));
	//					Scalar value = sum(kernel);
	//					float temp_value = value[0];
	//					aggregate_vec_cost_disparity_maps[i][d].at<float>(row, col) = temp_value;

	//				}

	//			}

	//		}
	//	}
	//}
	///////***************************************************88888888888888888888888888888888第一种聚合方法到此结束******************************************************************************************


	Mat aggregate_dst01 = aggregate_vec_cost_disparity_maps[0][0];
	Mat aggregate_dst02 = aggregate_vec_cost_disparity_maps[0][20];
	Mat aggregate_dst03 = aggregate_vec_cost_disparity_maps[0][30];
	Mat aggregate_dst04 = aggregate_vec_cost_disparity_maps[0][40];
	Mat aggregate_dst05 = aggregate_vec_cost_disparity_maps[0][50];
	Mat aggregate_dst06 = aggregate_vec_cost_disparity_maps[0][60];
	Mat aggregate_dst07 = aggregate_vec_cost_disparity_maps[0][70];
	Mat aggregate_dst08 = aggregate_vec_cost_disparity_maps[0][80];
	Mat aggregate_dst09 = aggregate_vec_cost_disparity_maps[0][90];
	Mat aggregate_dst10 = aggregate_vec_cost_disparity_maps[0][95];


	Mat aggregate_dst11 = aggregate_vec_cost_disparity_maps[1][10];
	Mat aggregate_dst12 = aggregate_vec_cost_disparity_maps[1][20];
	Mat aggregate_dst13 = aggregate_vec_cost_disparity_maps[1][30];
	Mat aggregate_dst14 = aggregate_vec_cost_disparity_maps[1][40];
	Mat aggregate_dst15 = aggregate_vec_cost_disparity_maps[1][50];
	Mat aggregate_dst16 = aggregate_vec_cost_disparity_maps[1][60];
	Mat aggregate_dst17 = aggregate_vec_cost_disparity_maps[1][70];
	Mat aggregate_dst18 = aggregate_vec_cost_disparity_maps[1][80];
	Mat aggregate_dst19 = aggregate_vec_cost_disparity_maps[1][90];
	Mat aggregate_dst20 = aggregate_vec_cost_disparity_maps[1][95];

	clock_t time_disparity_computation_start, time_disparity_computation_end;
	time_disparity_computation_start = clock();
	//视差计算
	int d_length = 0;
	for (int row = 0; row < height; row++)
	{
		for (int col = 0; col < width; col++)
		{
			float cost_value = 1000000;

			if (phase_left.at<float>(row, col) < 0.01)
			{
				continue;
			}
			else
			{
				for (int i = 0; i < 2; i++)
				{
					if (i == 0)
					{
						d_length = fabs(min_disparity);
					}
					else
					{
						d_length = fabs(max_disparity);
					}

					for (int d = 0; d < d_length; d++)
					{

						if ((aggregate_vec_cost_disparity_maps[i][d].at<float>(row, col) < 0.00001))
						{
							continue;
						}
						else
						{
							if (aggregate_vec_cost_disparity_maps[i][d].at<float>(row, col) < cost_value)
							{
								cost_value = aggregate_vec_cost_disparity_maps[i][d].at<float>(row, col);
								if (i == 0)
								{
									map_disparity_left.at<float>(row, col) = -d;
								}
								else
								{
									map_disparity_left.at<float>(row, col) = d;
								}
							}

						}
					}

				}

			}

		}
	}
	time_disparity_computation_end = clock();
	cout << "从DSI得到视差图用时：" << double((time_disparity_computation_end - time_disparity_computation_start) / CLOCKS_PER_SEC) << endl;

	clock_t time_2piont_cloud_start, time_2point_cloud_end;
	time_2piont_cloud_start = clock();
	//下面是由视差图计算点云
	for (int col = 0; col < width; col++)
	{
		for (int row = 0; row < height; row++)
		{
			if (map_disparity_left.at<float>(row, col) == 0)
			{
				continue;
			}
			else
			{
				//计算点云XYZ
				pcl::PointXYZ tep_points;
				Vec4d xyd1 = Vec4d(row, col, map_disparity_left.at<float>(row, col), 1);
				Matx44d _Q;
				Q.convertTo(_Q, CV_64F);
				Vec4d XYZW = _Q * xyd1;
				tep_points.x = XYZW[0] / XYZW[3];
				tep_points.y = XYZW[1] / XYZW[3];
				tep_points.z = XYZW[2] / XYZW[3];
				//points_cloud->points.push_back(tep_points);
				points_cloud.points.push_back(tep_points);
			}

		}
	}
	time_2point_cloud_end = clock();
	cout << "从视差图得到点云图用时：" << double((time_2point_cloud_end - time_2piont_cloud_start) / CLOCKS_PER_SEC);


	/*points_cloud->width = points_cloud->points.size();
	points_cloud->height = 1;
	pcl::io::savePCDFile("cloud2.pcd", *points_cloud);*/
	points_cloud.width = points_cloud.points.size();
	points_cloud.height = 1;
	pcl::io::savePCDFile("phone_SadRank.pcd", points_cloud);
	//return points_cloud;
}

///**********************************************************************************************************
//* 第二版计算视差的SAD函数 
//* 对左右相位图先取窗口聚合—>取左右差值最小得粗视差—>唯一性检测—>左右一致性检查得到LRCZ_map_disparity_right—>输出点云
//*function: 从相位图中计算视差，然后由视差得到点云数据
//* input：   phase1——左相机绝对相位图
//*           phase2——右相机绝对相位图
//* output：  输出点云XYZ数据坐标
//* history:  2020,7.26
//***********************************************************************************************************/
////pcl::PointCloud<pcl::PointXYZ>::Ptr phaseSad2Pcd(Mat phase_left, Mat phase_right, Mat Q)
 void phaseSad2Pcd(Mat phase_left, Mat phase_right, Mat Q)
{
	Mat cost_map_left=Mat::zeros(phase_left.size(), CV_32FC1);
	Mat cost_map_right=Mat::zeros(phase_left.size(), CV_32FC1);
	int width = phase_left.cols;
	int height = phase_left.rows;
	int sad_win_size = 5;

	Mat map_disparity_left=Mat::zeros(phase_left.size(),CV_32FC1);//视差图
	Mat map_disparity_right = Mat::zeros(phase_left.size(), CV_32FC1);//视差图
	Mat LRC_map_disparity_right = Mat::zeros(phase_left.size(), CV_32FC1);//;左右一致性检查后的视差图

    //pcl::PointCloud<pcl::PointXYZ>::Ptr points_cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>points_cloud;

	/****************由于SAD的特殊性，先聚合再匹配****************************************************************/
	for (int row = 0; row < height; row++)
	{
		for (int col = 0; col < width; col++)
		{
			/***************************************************************************************/
			//这一步是必须的，因为在下面sum取窗口的时候会出错，输入的相位图在小于0的像素值为-4.32×10^8，求和就不对了
			if (phase_left.at<float>(row, col) < 0.01)
			{
				phase_left.at<float>(row, col) = 0;
			}
			if (phase_right.at<float>(row, col) < 0.01)
			{
				phase_right.at<float>(row, col) = 0;
			}
		}
	}
	for (int row = 0; row < height; row++)
	{
		for (int col = 0; col < width; col++)
		{
			/****************************************************************************************************/

			bool out = ((row - sad_win_size / 2) <= 0 || (row + sad_win_size / 2) >= height || (col - sad_win_size / 2) <= 0 || (col + sad_win_size / 2) >= width);

			/*计算左边的SAD,由于SAD的特殊性，先聚合再相减*/
			bool zero_left = (phase_left.at<float>(row, col)< 0.01);
			if (out || zero_left)//如果处于边界或像素值为0，则跳过
			{
				continue;
			}
			else
			{
				Mat kernel_left = phase_left(Rect(col - sad_win_size / 2, row - sad_win_size / 2, sad_win_size, sad_win_size));
				Scalar temp_left = sum(kernel_left);
				cost_map_left.at<float>(row, col) = temp_left[0];
			}
		}
	}

	/*计算右边的SAD，先聚合*/
	for (int row = 0; row < height; row++)
	{
		for (int col = 0; col < width; col++)
		{
			bool out = ((row - sad_win_size / 2) <= 0 || (row + sad_win_size / 2) >= height || (col - sad_win_size / 2) <= 0 || (col + sad_win_size / 2) >= width);
		
			bool zero_right =( phase_right.at<float>(row, col) <0.01);
			if (out || zero_right)//如果处于边界或像素值为0，则跳过
			{
				continue;
			}
			else
			{
				Mat kernel_right = phase_right(Rect(col - sad_win_size / 2, row - sad_win_size / 2, sad_win_size, sad_win_size));
				Scalar temp_right = sum(kernel_right);
				cost_map_right.at<float>(row, col) = temp_right[0];
			}
		}
	}

	/*****************************************下面开始计算视差***********************************/
	/**********************************先计算左相机视差图，再计算右相机视差图**************************************/
 	
	for (int row = 0; row < height; row++)//左相机视差图

	{
		int start_coordinate = 0;//如果（row,col）的视差为d=col-k(row,k)，则（row,col+1）的视差一定在(row,k)的右边。

		for (int col = 0; col < width; col++)
		{
			if (cost_map_left.at<float>(row, col) < 0.01)
			{
				continue;
			}
			else
			{
				float min_cost_value=20000;//随便指定的一个比较大的数
				float best_disparity=0;
				float sec_min_cost = 20000;
				float sec_best_disparity=0;

				
				for (int k = start_coordinate; k < width; k++)//在同一个y坐标进行最佳视差搜索
				{
				    if (cost_map_right.at<float>(row, k) < 0.01)//如果右边视差为0，跳过该点
					{
						continue;
					}
					else
					{
						float temp_value = fabs(cost_map_left.at<float>(row, col) - cost_map_right.at<float>(row, k));
						if (temp_value < min_cost_value)
						{
							min_cost_value = temp_value;
							best_disparity = col - k;
							//point = Point2f(k, row);
							
						}
					}
				}

				//计算第二小的代价值
				
				for (int w = 0; w < width; w++)
				{
					if ((cost_map_right.at<float>(row, w) <0.01)||(w==col-best_disparity))//如果右边视差为0，跳过该点，最佳视差点也跳过
					{
						continue;
					}
					else
					{
						float temp_value = fabs(cost_map_left.at<float>(row, col) - cost_map_right.at<float>(row, w));
						if (temp_value < sec_min_cost)
						{
							sec_min_cost = temp_value;
							sec_best_disparity = col - w;
							//point = Point2f(k, row);
						}
					}
				}
				if (fabs(min_cost_value - sec_min_cost) < 5)//如果最小代价值和次最小代价值的差异不是特别大，则视为无效视差<<<视差唯一性>>>
				{
					continue;//视为无效视差，直接跳过
				}
				else
				{
					map_disparity_left.at<float>(row, col) = best_disparity;
				}

			}
		}
	}

	for (int row = 0; row < height; row++)//右相机视差图
	{
		//计算右相机视差图的时候，由于重合区域在两边，需要进行水平镜像翻转
		Mat remap_cost_map_left = Mat::zeros(cost_map_left.size(), CV_32FC1);
		Mat remap_cost_map_right = Mat::zeros(cost_map_left.size(), CV_32FC1);
		for (int col = 0; col < width; col++)//进行镜像翻转
		{
			for (int row = 0; row < height; row++)
			{
				remap_cost_map_left.at<float>(row, col) = cost_map_left.at<float>( row, width - col-1);//水平镜像翻转
				remap_cost_map_right.at<float>(row, col) = cost_map_right.at<float>(row, width - col-1);
			}
		}
		int start_coordinate = 0;//如果（row,col）的视差为d=col-k(row,k)，则（row,col+1）的视差一定在(row,k)的右边。

		for (int col = 0; col < width; col++)
		{
			if (remap_cost_map_right.at<float>(row, col) < 0.01)
			{
				continue;
			}
			else
			{
				float min_cost_value = 20000;//随便指定的一个比较大的数
				float best_disparity = 0;
				float sec_min_cost = 20000;
				float sec_best_disparity = 0;


				for (int k = start_coordinate; k < width; k++)//在同一个y坐标进行最佳视差搜索
				{
					if (remap_cost_map_left.at<float>(row, k) < 0.01)//如果左边代价为0，跳过该点
					{
						continue;
					}
					else
					{
						float temp_value = fabs(remap_cost_map_right.at<float>(row, col) - remap_cost_map_left.at<float>(row, k));
						if (temp_value < min_cost_value)
						{
							min_cost_value = temp_value;
							best_disparity = col - k;
							//point = Point2f(k, row);
							
						}
					}
				}

				//计算次最小的代价值

				for (int w = 0; w < width; w++)
				{
					if ((remap_cost_map_left.at<float>(row, w) < 0.01) || (w == col - best_disparity))//如果左边视差为0或位于次最小代价值点，跳过该点
					{
						continue;
					}
					else
					{
						float temp_value = fabs(remap_cost_map_right.at<float>(row, col) - remap_cost_map_left.at<float>(row, w));
						if (temp_value < sec_min_cost)
						{
							sec_min_cost = temp_value;
							sec_best_disparity = col - w;
							//point = Point2f(k, row);
						}
					}
				}
				if (fabs(min_cost_value - sec_min_cost) < 5)//如果最小代价值和次最小代价值的差异不是特别大，说明极有可能是收到噪声的污染，导致二者相近，则视为无效视差<<<视差唯一性>>>
				{
					continue;//视为无效视差，直接跳过
				}
				else
				{
					map_disparity_right.at<float>(row,width- col-1) = best_disparity;//再翻转回来
				}

			}
		}
	}

	//左右一致性检查
	for (int row = 0; row < height; row++)
	{
		
		for (int col = 0; col < width; col++)
		{
			if (cost_map_left.at<float>(row, col) < 0.01)//忽略被掩膜的区域
			{
				continue;
			}
			else
			{
				float d_left = map_disparity_left.at<float>(row, col);
				//float temp11 = width - (col - d_left);//由于镜像之后，左图视差右图对应d的坐标为temp11
				float temp11 = col - d_left;
				if ((temp11 > 0) && (temp11 < width))
				{
					float d_right = map_disparity_right.at<float>(row, temp11);
					if (fabs(d_left - d_right) < 1)//满足左=右，说明视差是正确的
					{
						LRC_map_disparity_right.at<float>(row, col) = map_disparity_left.at<float>(row, col);
					}
				}
			}
		}
	}

					//视差精细化
				 //   float c1 = fabs(cost_map_left.at<float>(row, col) - cost_map_right.at<float>(row, col- best_disparity - 1));
					//float c2 = fabs(cost_map_left.at<float>(row, col) - cost_map_right.at<float>(row, col - best_disparity));
					//float c3 = fabs(cost_map_left.at<float>(row, col) - cost_map_right.at<float>(row, col - best_disparity + 1));
					//float a = (c3 + c1 - 2 * c2);
					//float precise_disparity = best_disparity - (c3 - c1) /2*a;


	//下面是由视差图计算点云
	for (int col = 0; col < width; col++)
	{
		for (int row = 0; row < height; row++)
		{
			if (LRC_map_disparity_right.at<float>(row, col) == 0)
			{
				continue;
			}
			else
			{
				//计算点云XYZ
				pcl::PointXYZ tep_points;
				Vec4d xyd1 = Vec4d(row, col, LRC_map_disparity_right.at<float>(row, col), 1);
				Matx44d _Q;
				Q.convertTo(_Q, CV_64F);
				Vec4d XYZW = _Q * xyd1;
				tep_points.x = XYZW[0] / XYZW[3];
				tep_points.y = XYZW[1] / XYZW[3];
				tep_points.z = XYZW[2] / XYZW[3];
				//points_cloud->points.push_back(tep_points);
				points_cloud.points.push_back(tep_points);
			}
		
		}
	}


				
	/*points_cloud->width = points_cloud->points.size();
	points_cloud->height = 1;
	pcl::io::savePCDFile("cloud2.pcd", *points_cloud);*/
	points_cloud .width = points_cloud .points.size();
	points_cloud.height = 1;
	pcl::io::savePCDFile("phone_SadUniquenessLrc.pcd", points_cloud);
	//return points_cloud;
}

//得到Rank_transform的等级
inline int Rank_value(float diff)
{

	int Crank = 0;
	if (diff <= u)
	{
		if (diff > -u)     //-u<diff<=u
		{
			return 0;
		}
		else
		{
			if (diff <= -v)   //diff<=-v
			{
				return -2;
			}
			else                 //-v<diff<=u
			{
				return -1;
			}
		}
	}
	else
	{
		if (diff >= v)     //diff>=v
		{
			return 2;
		}
		else                // u<diff<v
		{
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

/***********************************************AD-Census代价计算**********************************************************************/
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


//使用tangulapoints来计算点云
void UsingTrangulPoints()
{
		////+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
		//	/************************************************************************
		//	* 第一版输出点云的方法，使用triangulatePoints
		//	* 使用了自己后面定义的两个函数computeSAD（） 和  Q2pcd（）
		//	***************************************************************************
		//	*/
			//输入两张相位图和SAD窗口大小，得到两个vector，里面是点对
			//vector<Point2f>vecL;
			//vector<Point2f>vecR;
			//clock_t startTime6, endTime6;
			//startTime6 = clock();

			// computeSAD(L_phase, R_phase, vecL, vecR, Size(7, 7));

			//endTime6 = clock();
			//cout << "SAD匹配成功" << endl;
			//cout << "SAD时间：" << double((endTime6 - startTime6) / CLOCKS_PER_SEC) << endl;

			//Mat P1, P2;
			//Mat point4D( 4, vecL.size(), CV_32F);

			//fin["P1"] >> P1;
			//fin["P2"] >> P2;

			//clock_t startTime7, endTime7;
			//startTime7 = clock();
			////计算出Point4D矩阵，该函数为opencv自带API
			//triangulatePoints(P1, P2, vecL, vecR, point4D);
			//endTime7 = clock();
			//cout << "triangulate时间：" << double((endTime7- startTime7) / CLOCKS_PER_SEC) << endl;
			//string saveLoad = "triangulatePoints.pcd";
			//Q2pcd(point4D, saveLoad);
			//cout << "成功输出点云" << endl;
}

/**********************************************************************
/*
* 这个是只有SAD，没有唯一性检测和LRC
* 
*************************************************************************/

void phaseSad2PcdSadonly(Mat phase_left, Mat phase_right, Mat Q )
{
	Mat cost_map_left = Mat::zeros(phase_left.size(), CV_32FC1);
	Mat cost_map_right = Mat::zeros(phase_left.size(), CV_32FC1);
	int width = phase_left.cols;
	int height = phase_left.rows;
	int sad_win_size = 5;

	Mat LRC_map_disparity_right = Mat::zeros(phase_left.size(), CV_32FC1);//;左右一致性检查后的视差图

	pcl::PointCloud<pcl::PointXYZ>points_cloud;
	/****************由于SAD的特殊性，先聚合再匹配****************************************************************/
	for (int row = 0; row < height; row++){
		for (int col = 0; col < width; col++){
			/***************************************************************************************/
			//这一步是必须的，因为在下面sum取窗口的时候会出错，输入的相位图在小于0的像素值为-4.32×10^8，求和就不对了
			if (phase_left.at<float>(row, col) < 0.01){
				phase_left.at<float>(row, col) = 0;
			}
			if (phase_right.at<float>(row, col) < 0.01){
				phase_right.at<float>(row, col) = 0;
			}
		}
	}
	//针对上述问题，把左右两幅图像的聚合改为各自聚合
	for (int row = 0; row < height; row++){
		for (int col = 0; col < width; col++){
			bool out = ((row - sad_win_size / 2) <= 0 || (row + sad_win_size / 2) >= height || (col - sad_win_size / 2) <= 0 || (col + sad_win_size / 2) >= width);
			/*计算左边的SAD,由于SAD的特殊性，先聚合再相减*/
			bool zero_left = (phase_left.at<float>(row, col) < 0.01);
			if (out || zero_left){//如果处于边界或像素值为0，则跳过
				continue;
			}
			else{
				Mat kernel_left = phase_left(Rect(col - sad_win_size / 2, row - sad_win_size / 2, sad_win_size, sad_win_size));
				Scalar temp_left = sum(kernel_left);
				cost_map_left.at<float>(row, col) = temp_left[0];
			}
		}
	}

	for (int row = 0; row < height; row++){
		for (int col = 0; col < width; col++){
			/*计算右边的SAD，先聚合*/
			bool out = ((row - sad_win_size / 2) <= 0 || (row + sad_win_size / 2) >= height || (col - sad_win_size / 2) <= 0 || (col + sad_win_size / 2) >= width);
			bool zero_right = (phase_right.at<float>(row, col) < 0.01);
			if (out || zero_right){//如果处于边界或像素值为0，则跳过
				continue;
			}
			else{
				Mat kernel_right = phase_right(Rect(col - sad_win_size / 2, row - sad_win_size / 2, sad_win_size, sad_win_size));
				Scalar temp_right = sum(kernel_right);
				cost_map_right.at<float>(row, col) = temp_right[0];
			}
		}
	}

	/*****************************************下面开始计算视差***********************************/
	for (int row = 0; row < height; row++){//左相机视差图{
		int start_coordinate = 0;//如果（row,col）的视差为d=col-k(row,k)，则（row,col+1）的视差一定在(row,k)的右边。

		for (int col = 0; col < width; col++){
			if (cost_map_left.at<float>(row, col) < 0.01){
				continue;
			}
			else{
				float min_cost_value = 20000;//随便指定的一个比较大的数
				float best_disparity = 0;
				int a = row;
				int b = col;
				for (int k = start_coordinate; k < width; k++){//在同一个y坐标进行最佳视差搜索
					if (cost_map_right.at<float>(row, k) < 0.01){//如果右边视差为0，跳过该点
						continue;
					}
					else{
						float temp_value = fabs(cost_map_left.at<float>(row, col) - cost_map_right.at<float>(row, k));
						if (temp_value < min_cost_value){
							min_cost_value = temp_value;
							best_disparity = col - k;
						}
					}
				}
				if (best_disparity != 0){
					LRC_map_disparity_right.at<float>(row, col) = best_disparity;
				}
			}		
		}
	}
	//下面是由视差图计算点云
	for (int col = 0; col < width; col++){
		for (int row = 0; row < height; row++){
			if (LRC_map_disparity_right.at<float>(row, col) == 0){
				continue;
			}
			else{
				//计算点云XYZ
				pcl::PointXYZ tep_points;
				Vec4d xyd1 = Vec4d(row, col, LRC_map_disparity_right.at<float>(row, col), 1);
				Matx44d _Q;
				Q.convertTo(_Q, CV_64F);
				Vec4d XYZW = _Q * xyd1;
				tep_points.x = XYZW[0] / XYZW[3];
				tep_points.y = XYZW[1] / XYZW[3];
				tep_points.z = XYZW[2] / XYZW[3];
				//points_cloud->points.push_back(tep_points);
				points_cloud.points.push_back(tep_points);
			}
		}
	}
	points_cloud.width = points_cloud.points.size();
	points_cloud.height = 1;
	pcl::io::savePCDFile("phone_Sad_only.pcd", points_cloud);
	getchar();
}




/******************************************************************************************
*function:第一版用trianguPoints时使用的SAD计算点对的函数
*input: left---左相机相位图
*       right---右相机相位图
*       vecL----使用引用输出左相机的点
*       vecR----使用引用输出右相机的点
*       winSize---SAD窗口的大小
*output:
history:2020.7.25
********************************************************************************************/
//输入左右相位图片和SAD窗口大小（一般取7x7)，得到对应的左右点集vecL,vecR
void computeSAD(Mat &left, Mat&right, vector<Point2f>&vecL, vector<Point2f>&vecR, Size winSize)
{
	Mat cost_map_left = Mat::zeros(left.size(), CV_32FC1);
	Mat cost_map_right = Mat::zeros(left.size(), CV_32FC1);
	//由于SAD得特殊性，先聚合再算差进行匹配
	for (int row = 0; row < left.rows; row++)
	{
		for (int col = 0; col < left.cols; col++)
		{
			/***************************************************************************************/
			//这一步是必须的，因为在下面sum取窗口的时候会出错，输入的相位图在小于0的像素值为-4.32×10^8，求和就不对了
			if (left.at<float>(row, col) < 0.01)
			{
				left.at<float>(row, col) = 0;
			}
			if (right.at<float>(row, col) < 0.01)
			{
				right.at<float>(row, col) = 0;
			}
		}
	}
			/****************************************************************************************************/
	for (int row = 0; row < left.rows; row++)
	{
		for (int col = 0; col < left.cols; col++)
		{
			bool out = ((row - winSize.height / 2) <= 0 || (row + winSize.height / 2) >= left.rows || (col - winSize.height / 2) <= 0 || (col + winSize.height / 2) >= left.cols);

			/*计算左边的SAD,由于SAD的特殊性，先聚合再相减*/
			bool zero_left = (left.at<float>(row, col) < 0.01);//用<0.01代替==0比较好
			if (out || zero_left)//如果处于边界或像素值为0，则跳过
			{
				continue;
			}
			else
			{

				Mat kernel_left = left(Rect(col - winSize.height / 2, row - winSize.height / 2, winSize.height, winSize.width));
				Scalar temp_left = sum(kernel_left);
				float value = temp_left[0];
				cost_map_left.at<float>(row, col) = temp_left[0];
			}

		}
	}
	for (int row = 0; row < left.rows; row++)
	{
		for (int col = 0; col < left.cols; col++)
		{
			bool out = ((row - winSize.height / 2) <= 0 || (row + winSize.height / 2) >= left.rows || (col - winSize.height / 2) <= 0 || (col + winSize.height / 2) >= left.cols);
			/*计算右边的SAD，先聚合*/
			bool zero_right = (right.at<float>(row, col) < 0.01);
			if (out || zero_right)//如果处于边界或像素值为0，则跳过
			{
				continue;
			}
			else
			{
				Mat kernel_right = right(Rect(col - winSize.height / 2, row - winSize.height / 2, winSize.height, winSize.width));
				Scalar temp_right = sum(kernel_right);
				cost_map_right.at<float>(row, col) = temp_right[0];
			}
		}
	}
	for (int col = 0; col < left.cols; col++)
	{
		for (int row = 0; row < left.rows; row++)
		{
			if (cost_map_left.at<float>(row, col) < 0.01)//代价图左图视差为零，跳过
			{
				continue;
			}
			else
			{
				int best_disparity = 0;
				float value = 10000;

				for (int k = 0; k < right.cols; k++)
				{
					if (cost_map_right.at<float>(row, k) < 0.01)//如果代价图右图的值为零，跳过
					{
						continue;
					}
					else
					{
						float temp_value = fabs(cost_map_right.at<float>(row, k) - cost_map_left.at<float>(row, col));
						if (temp_value < value)
						{
							value = temp_value;
							best_disparity = col - k;
						/*	int a = row;
							int b = col;
							int c = 0;
						*/
						}
						else
						{
							continue;
						}
					}
				}

				vecL.push_back(Point2f(col, row));
				vecR.push_back(Point2f(col - best_disparity, row));

			}
		}
	}
	
	
	//下面是我第一遍写得，取窗口再计算差，再聚合，贼慢
	//for (float i = winSize.width; i < left.cols - winSize.width; i++)
	//{
	//	for (float j = winSize.height; j < left.rows - winSize.height; j++)
	//	{
	//		if (left.at<float>(j, i) < 0.01)//像素值为零，直接跳过匹配
	//		{
	//			continue;
	//		}
	//		else
	//		{
	//			map<float, Point>minPoint;
	//			for (float k = winSize.width; k < right.cols - winSize.width; k++)
	//			{
	//				if (right.at<float>(j, k) < 0.01)
	//				{
	//					continue;
	//				}
	//				else
	//				{
	//					Mat kernel_L = Mat(winSize, CV_32FC1, Scalar(0));
	//					Mat kernel_R = Mat(winSize, CV_32FC1, Scalar(0));
	//					kernel_L = left(Rect(i - winSize.width / 2, j - winSize.height / 2, winSize.width, winSize.height));
	//					kernel_R = right(Rect(k - winSize.width / 2, j - winSize.height / 2, winSize.width, winSize.height));
	//					Mat difference;
	//					::absdiff(kernel_L, kernel_R, difference);
	//					Scalar D = sum(difference);
	//					float  aaa = D[0];
	//					minPoint.insert(make_pair(aaa, Point(k, j)));

	//				}
	//			}
	//			if (minPoint.empty())
	//			{
	//				continue;
	//			}
	//			else
	//			{
	//				map<float, Point>::iterator iter = minPoint.begin();
	//				Point tempMinPoint = iter->second;
	//				float temNum = iter->first;
	//				for (map<float, Point>::iterator it = minPoint.begin(); it != minPoint.end(); it++)
	//				{
	//					if (it->first < temNum)
	//					{
	//						tempMinPoint = it->second;
	//					}
	//					else
	//					{
	//						continue;
	//					}
	//				}
	//				minPoint.clear();
	//				vecL.push_back(Point(i, j));
	//				vecR.push_back(tempMinPoint);
	//			}
	//		}

	//	}
	//}
}


/******************************************************************************************
*function:第一版用trianguPoints时使用的点云输出函数
*input: Point4D---是trianguPoint输出的Q矩阵
*       pcdFileName---是pcd点云的保存路径
*output:输出PCD点云到磁盘
history:2020,7.25
********************************************************************************************/
void Q2pcd(Mat Point4D, string pcdFileName)//Q为trigulatepoints输出的4xN的矩阵，这个函数把Q矩阵转化为PCD文件
{
	pcl::PointCloud < pcl::PointXYZ > cloud;
	cloud.width = Point4D.cols;
	cloud.height = 1;
	cloud.points.resize(cloud.width*cloud.height);

	float*Q_row1 = Point4D.ptr<float>(0);
	float*Q_row2 = Point4D.ptr<float>(1);
	float*Q_row3 = Point4D.ptr<float>(2);
	float*Q_row4 = Point4D.ptr<float>(3);
	for (int i = 0; i < Point4D.cols; i++)
	{
		float point3D_data4 = *(Q_row4 + i);
		float point3D_data1 = *(Q_row1 + i) / point3D_data4;
		float point3D_data2 = *(Q_row2 + i) / point3D_data4;
		float point3D_data3 = *(Q_row3 + i) / point3D_data4;
		if (i < cloud.points.size())
		{
			cloud.points[i].x = point3D_data1;
			cloud.points[i].y = point3D_data2;
			cloud.points[i].z = point3D_data3;
		}
	}
	pcl::io::savePCDFileASCII(pcdFileName, cloud);
}



//读文件流得到标定需要的点集的函数
Size read_file(ifstream & ifile, vector<vector<Point2f>>&imagePointVec, vector<vector<Point3f>>&realPointVec,
	Size&nei_jiaodian_number1, Size&square_size1)
{
	Size t;
	
	while (!ifile.eof())
	{
		vector<Point2f>image_point;
		string filename;
		getline(ifile, filename);
		Mat src = imread(filename);
		if (src.empty())
		{
			printf("找不到图片");
			break;
		}
		else
		{
			if (0 == findChessboardCorners(src, nei_jiaodian_number1, image_point))
			{
				cout << "找不到角点";
			}
			else
			{
				Mat gray;
				cvtColor(src, gray, COLOR_RGB2GRAY);
				TermCriteria criteria = TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1);
				cornerSubPix(gray, image_point, Size(5, 5), Size(-1, -1), criteria);
				drawChessboardCorners(src, nei_jiaodian_number1, image_point, true);
				//获取object_point点
				vector<Point3f>temp_object_point;

				for (int i = 0; i < nei_jiaodian_number1.height; ++i)
				{
					for (int j = 0; j < nei_jiaodian_number1.width; ++j)
					{
						Point3f XYZ;
						XYZ.x = square_size1.height*j;
						XYZ.y = square_size1.width*i;
						XYZ.z = 0;
						temp_object_point.push_back(XYZ);
					}
				}

				realPointVec.push_back(temp_object_point);//棋盘上的坐标点，假设z=0;
			}

			imagePointVec.push_back(image_point);//检测到的亚像素角点
		}
		//
		//namedWindow("1", WINDOW_AUTOSIZE);
		//imshow("1", src);
		//waitKey(500);
		t = src.size();
	}
	return t;
}