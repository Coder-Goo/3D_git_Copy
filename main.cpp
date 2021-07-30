
#include"calibration.h"
#include"aggregation.h"
#include"phase_unwrapping2.h"
#include"ad_census.h"

using namespace cv;
using namespace std;

Size read_file(ifstream & ifile, vector<vector<Point2f>>&imagePointVec, vector<vector<Point3f>>&realPointVec, Size&nei_jiaodian_number1, Size&square_size1);
//标定
int CalibrationImage();//标定
//计算相位图
int CalculatePhaseImage(Mat &phase_left, Mat &phase_right);
Mat phaseSad2PcdSadonly(const Mat &_phase_left, const Mat &_phase_right, int );
//AD计算视差
Mat ADCalculateDisparity(const Mat &phase_left, const Mat &phase_right);
//由视差计算点云
void CalculatePointCloud(const Mat &disparity, const Mat &phase_left, const Mat &Q, String &file_name);
float GetPreciseDisparity(int best_disparity, int row, int col);
float GetPreciseDisparityFromAD(Point2f x1, Point2f x2, Point2f x3);
String pcd_file_name = "AD点云";
string calib_file_name = "./Calibdata.xml";

int main()
{
	//AD-CENSUS还是有问题的
	bool dsi = false;
	bool sad = false;
	Mat Q;     //投影矩阵Q
	CalibrationImage();//标定
	FileStorage fin(calib_file_name, FileStorage::READ);
	fin["Q"] >> Q;
	//cout << Q << endl;

  	Mat phase_left,phase_right;//左右相位图
	vector<vector<Mat>>vec_cost_maps; //代价DSI
	vector<vector<Mat>>vec_aggregation_maps;//聚合DSI
	vector<vector<Mat>>vec_aggregation_right_maps;//右边的DSI
	Mat left_disparity,right_disparity;
	
	std::cout << CalculatePhaseImage(phase_left, phase_right) << endl;//解相位

	if(dsi)//使用DSI 计算视差图
	{
		CostCalculation(vec_cost_maps, phase_left, phase_right, cost_window_size);
		CostAggregation(fixed_window, vec_cost_maps, vec_aggregation_maps, phase_left, phase_right, aggeragation_window_size);
		/*vector<vector<Mat>>right_dsi;
		CalculateRightDsi(right_dsi, vec_aggregation_maps,phase_left,phase_right);*/
		left_disparity=DisparityCalculation( vec_aggregation_maps, phase_left, phase_right);
		//right_disparity= DisparityCalculation(right_dsi, phase_left, phase_right);
		pcd_file_name = "AD-CENSUS点云";
		CalculatePointCloud(left_disparity, phase_left, Q, pcd_file_name);
	}
	else//不使用DSI计算视差图
	{
		if (sad) {
			pcd_file_name = "SAD点云";
			Mat disparity =  phaseSad2PcdSadonly(phase_left,phase_right, 5);
			CalculatePointCloud(disparity, phase_left, Q, pcd_file_name);
		}
		else {
			pcd_file_name = "AD点云";
			Mat disparity = ADCalculateDisparity(phase_left, phase_right);
			CalculatePointCloud(disparity, phase_left, Q, pcd_file_name);
		}
	}
	
	fin.release();

	return 0;
}


//AD+THRESHOLD
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

#ifdef PAPER
	imwrite("E:/科研/论文/改师兄的论文/论文/流程图片集/disparity.bmp", disparity);
#endif

	return disparity;
}

/**********************************************************************
/*
* 这个是只有SAD，没有唯一性检测和LRC
* 
*************************************************************************/
Mat phaseSad2PcdSadonly(const Mat &phase_left, const Mat &phase_right, int sad_win_size = 5)
{
	clock_t time_start, time_end;
	time_start = clock();
	int width = phase_left.cols;
	int height = phase_left.rows; 
	Mat disparity(height, width, CV_32FC1, DEFAULT_DISPARITY);
	Mat aggre_phase_left = Mat::zeros(phase_left.size(), CV_32FC1);
	Mat aggre_phase_right = Mat::zeros(phase_left.size(), CV_32FC1);

	//针对上述问题，把左右两幅图像的聚合改为各自聚合
	Mat kernel;
	for (int row = sad_win_size / 2; row < height- sad_win_size / 2; row++){
		for (int col = sad_win_size / 2; col < width- sad_win_size / 2; col++){
			if (phase_left.at<float>(row,col)<0.01){//如果像素值为0，则跳过
				continue;
			}
			else{
				kernel = phase_left(Rect(col - sad_win_size / 2, row - sad_win_size / 2, sad_win_size, sad_win_size));
				Scalar temp_left = sum(kernel);
				aggre_phase_left.at<float>(row, col) = temp_left[0];
			}
		}
	}
	for (int row = sad_win_size / 2; row < height - sad_win_size / 2; row++) {
		for (int col = sad_win_size / 2; col < width - sad_win_size / 2; col++) {
			if (phase_right.at<float>(row, col) < 0.01) {//如果处于边界或像素值为0，则跳过
				continue;
			}
			else{
				kernel = phase_right(Rect(col - sad_win_size / 2, row - sad_win_size / 2, sad_win_size, sad_win_size));
				Scalar temp_right = sum(kernel);
				aggre_phase_right.at<float>(row, col) = temp_right[0];
			}
		}
	}

	/*****************************************下面开始计算视差***********************************/
	for (int row = 0; row < height; row++){//左相机视差图{
		const float * left_ptr = aggre_phase_left.ptr<float>(row);
		const float* right_ptr = aggre_phase_right.ptr<float>(row);
		float* disparity_ptr = disparity.ptr<float>(row);
		//int start_coordinate = 0;//如果（row,col）的视差为d=col-k(row,k)，则（row,col+1）的视差一定在(row,k)的右边。
		for (int col = 0; col < width; col++){
				if (left_ptr[col] < 0.01)  continue;

				float min_cost_value = 20000;//随便指定的一个比较大的数
				float temp_value = 0;
				int best_disparity = DEFAULT_DISPARITY;
				for (int k = 0; k < width; k++){//在同一个y坐标进行最佳视差搜索
					if (right_ptr[k] > 0.01){//如果右边视差为0，跳过该点
						temp_value = fabs(left_ptr[col]- right_ptr[k]);
						if (temp_value < min_cost_value){
							min_cost_value = temp_value;
							best_disparity = col - k;
						}
					}
				}
				if (best_disparity != DEFAULT_DISPARITY && min_cost_value < 1){
					disparity_ptr[col] = best_disparity;
				}		
		}
	}

	/**********下面输出看结果**********/
	/*FileStorage fout("C:\\Program Files\\MATLAB\\R2016b\\bin\\StereoMatching/aggregation_phase_left.xml", FileStorage::WRITE);
	fout << "L_phase" << aggre_phase_left;
	fout.release();
	FileStorage fout1("C:\\Program Files\\MATLAB\\R2016b\\bin\\StereoMatching/aggregation_phase_right.xml", FileStorage::WRITE);
	fout1 << "R_phase" << aggre_phase_right;
	fout1.release();*/


	time_end = clock();
	std::cout << "从SAD计算得到视差图用时：" << double((time_end - time_start) / CLOCKS_PER_SEC) << endl;
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
void CalculatePointCloud(const Mat &disparity, const Mat & phase_left, const Mat &Q, String &file_name)
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
	pcl::io::savePCDFile(file_name + ".pcd", points_cloud);
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
	Calibration calibration(calib_file_name, image_point_vecL, image_point_vecR, object_point_vec, imageSize);
	calibration.calibration();//完成标定和矫正,输出标定参数到yml文件中。
	endTime1 = clock();
	cout << "标定用时：" << double((endTime1 - startTime1) / CLOCKS_PER_SEC) << "s" << endl;
	return 0;
}
//计算相位图
int CalculatePhaseImage(Mat &phase_left, Mat &phase_right)
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
	threshold(src11, src111, 50, 255, THRESH_BINARY | THRESH_OTSU);
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

#ifdef PAPER
	imwrite("E:/科研/论文/改师兄的论文/论文/流程图片集/mask_L.bmp", mask_L);
	imwrite("E:/科研/论文/改师兄的论文/论文/流程图片集/mask_R.bmp", mask_R);
#endif
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
		/*imshow("mask之后的图片", srcTepL);
		waitKey();*/

		
		left_img.push_back(srcTepL.clone());
		string str_right;
		getline(right_phase, str_right);
		Mat dst = imread(str_right);
		Mat srcTepR;
		dst.copyTo(srcTepR, mask_R);
		right_img.push_back(srcTepR.clone());

#ifdef PAPER
		imwrite("E:/科研/论文/改师兄的论文/论文/流程图片集/left_image_masked.bmp", srcTepL);
		imwrite("E:/科研/论文/改师兄的论文/论文/流程图片集/right_image_masked.bmp", srcTepR);
#endif
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

#ifdef PAPER
	imwrite("E:/科研/论文/改师兄的论文/论文/流程图片集/rectified_right1.bmp", rightImg[0]);
	imwrite("E:/科研/论文/改师兄的论文/论文/流程图片集/rectified_left1.bmp", leftImg[0]);
#endif

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


#ifdef PAPER
	imwrite("E:/科研/论文/改师兄的论文/论文/流程图片集/left_phase.bmp", L_phase);
	imwrite("E:/科研/论文/改师兄的论文/论文/流程图片集/right_phase.bmp", R_phase);
#endif

	/***********************************用来显示看是否正确,得先把图片变为uint类型*************************************/

	//FileStorage fout("C:\\Program Files\\MATLAB\\R2016b\\bin\\StereoMatching/phase_left.xml", FileStorage::WRITE);
	//fout << "L_phase" << L_phase;
	//fout.release();
	//FileStorage fout1("C:\\Program Files\\MATLAB\\R2016b\\bin\\StereoMatching/phase_right.xml", FileStorage::WRITE);
	//fout1 << "R_phase" << R_phase;
	//fout1.release();

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