
//������������ı궨ͼƬ��ȡ�Ľǵ�vector����Ӧ����������㣬�õ�һ���궨�����ļ�
#pragma once
#ifndef JIAOZHENG
#define JIAOZHENG
#include"biao_tou.h"
using namespace cv;
using namespace std;


class Calibration
{
public:
	Calibration(string calib_file_name, vector<vector<Point2f>>imagePointVecL, vector<vector<Point2f>>imagePointVecR, vector<vector<Point3f>>realPointVec, Size imageSize);//���캯����������ļ���
	int calibration();//�궨�ͽ��������������ӳ�����yml�ļ���ȥ
private:
	vector<vector<Point2f>>imagePointVecL;
	vector<vector<Point2f>>imagePointVecR;
	vector<vector<Point3f>>realPointVec;
	Size imagesize;
	string calib_file_name;
};
#endif
