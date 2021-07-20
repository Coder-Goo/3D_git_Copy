#include "calibration.h"

using namespace std;
using namespace cv;

Calibration::Calibration(string calib_file_name, vector<vector<Point2f>>imagePointVecL, vector<vector<Point2f>>imagePointVecR, vector<vector<Point3f>>realPointVec, Size imageSize)
{
	this->imagePointVecL = imagePointVecL;
	this->imagePointVecR = imagePointVecR;
	this->realPointVec= realPointVec;
	this->imagesize = imageSize;
	this->calib_file_name = calib_file_name;
}
int Calibration::calibration()
{

	FileStorage fout(calib_file_name, FileStorage::WRITE);//输出相机的内外参数
	//read_file函数，第一个参数image_point_vecL为检测到的亚像素角点，第二个参数 object_point_vec就是实物上的z=0的点
	//第三个参数为输入的文件流，第四个为内角点数，第五个为棋盘格大小。参数全为引用
	
	Mat camereMatrix1 = Mat::eye(3, 3, CV_64FC1);
	Mat distcoffs1;
	Mat camereMatrix2 = Mat::eye(3, 3, CV_64FC1);
	Mat distcoffs2;
	Mat K1;//左相机旋转向量，
	Mat K2;//右相机旋转向量，
	Mat M1;//左相机平移向量，
	Mat M2;//右相机平移向量，
	//cout << "开始左右相机的单独标定" << endl;
	calibrateCamera(realPointVec, imagePointVecL, imagesize, camereMatrix1, distcoffs1, K1, M1, 0);
	calibrateCamera(realPointVec , imagePointVecR, imagesize, camereMatrix2, distcoffs2, K2, M2, 0);

	///*输出相机标定参数，*/
	//double fovxL, fovyL, focalLengthL, aspectRatioL;
	//Point2d principalPointL;
	//calibrationMatrixValues(camereMatrix1, imagesize, 0.0052, 0.0052, fovxL, fovyL, focalLengthL, principalPointL, aspectRatioL);
	//double fovxR, fovyR, focalLengthR, aspectRatioR;
	//Point2d principalPointR;
	//calibrationMatrixValues(camereMatrix2, imagesize, 0.0052, 0.0052, fovxR, fovyR, focalLengthR, principalPointR, aspectRatioR);
	//
	//ofstream outfile;
	//outfile.open("内外参数.txt");
	//outfile << "cameraMatrix1;" << camereMatrix1 << endl;
	//outfile << "fovxL:" << fovxL << endl;
	//outfile << "fovyL:" << fovyL << endl;
	//outfile << "focalLengthL:" << focalLengthL << endl;
	//outfile << "principalPoint" << principalPointL << endl;
	//outfile << "aspectRatioL" << aspectRatioL << endl;

	//outfile << "cameraMatrix2;" << camereMatrix2 << endl;
	//outfile << "fovxR:" << fovxR << endl;
	//outfile << "fovyR:" << fovyR << endl;
	//outfile << "focalLengthR:" << focalLengthR << endl;
	//outfile << "principalPointR" << principalPointR << endl;
	//outfile << "aspectRatioR" << aspectRatioR << endl;
	//
	//////cout << "左右相机单独标定完成" << endl;
	//////fout << "camereMatrix1\n" << camereMatrix1 << endl;

	Mat R;
	Mat T;
	Mat E;
	Mat F;
	//cout << "立体标定开始" << endl;
	//TermCriteria criteria = TermCriteria(TermCriteria::COUNT| TermCriteria::EPS,100,0.000001);
	int flags = CALIB_FIX_ASPECT_RATIO +CALIB_SAME_FOCAL_LENGTH +CALIB_ZERO_TANGENT_DIST +CALIB_RATIONAL_MODEL +CALIB_FIX_K3 + CALIB_FIX_K4 + CALIB_FIX_K5;
	double error= stereoCalibrate(realPointVec , imagePointVecL, imagePointVecR, camereMatrix1, distcoffs1, camereMatrix2,
		distcoffs2, imagesize, R, T, E, F, flags, TermCriteria(TermCriteria::COUNT | TermCriteria::EPS, 100, 1e-5));
	cout << "标定误差" << error << endl;
	//stereoRectify();
	//cout << "立体标定结束" << endl;
	//waitKey(0);
	Mat R1;
	Mat R2;
	Mat P1;
	Mat P2;
	Mat Q;
	stereoRectify(camereMatrix1, distcoffs1, camereMatrix2, distcoffs2, imagesize, R, T, R1, R2, P1, P2, Q, 0);
	
	Mat map11;
	Mat map12;
	initUndistortRectifyMap(camereMatrix1, distcoffs1, R1, P1, imagesize, CV_16SC2, map11, map12);
	Mat map21;
	Mat map22;
	initUndistortRectifyMap(camereMatrix2, distcoffs2, R2, P2, imagesize, CV_16SC2, map21, map22);

	//看看输出的结果，使用文件输出流
	ofstream ofile("./输出的内外参数txt.txt");
	ofile << "camereMatrix1" << camereMatrix1<<endl;
	ofile << "distcoffs1" << distcoffs1<<endl;
	ofile << "camereMatrix2" << camereMatrix2 <<endl;
	ofile << "distcoffs2" << distcoffs2 << endl;

	ofile << "R" << R << endl;
	ofile << "T" << T << endl;
	ofile << "E" << E << endl;
	ofile << "F" << F << endl;

	ofile << "R1" << R1 << endl;
	ofile << "R2" << R2 << endl;
	ofile << "P1" << P1 << endl;
	ofile << "P2" << P2 << endl;
	ofile << "Q" << Q << endl;

	ofile << "map11" << map11<<endl << endl << endl;
	ofile << "map12" << map12 << endl << endl << endl;
	ofile << "map21" << map21 << endl << endl << endl;
	ofile << "map22" << map22 << endl << endl << endl;

	//输出相机的内外参数到外部yml文件中去
	fout << "camereMatrix1" << camereMatrix1;
	fout << "distcoffs1" << distcoffs1;
	fout << "camereMatrix2" << map21;
	fout << "distcoffs2" << map22;

	fout << "R" << R;
	fout << "T" << T;
	fout << "E" << E;
	fout << "F" << F;

	fout << "R1" << R1;
	fout << "R2" << R2;
	fout << "P1" << P1;
	fout << "P2" << P2;
	fout << "Q" << Q;

	fout << "map11" << map11;
	fout << "map12" << map12;
	fout << "map21" << map21;
	fout << "map22" << map22;

	fout.release();
	cout << "标定结束" << endl;
	waitKey(500);
	return 0;
}

