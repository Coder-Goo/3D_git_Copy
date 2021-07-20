#pragma once
#ifndef PHASE2PCD
#define PHASE2PCD
#include"biao_tou.h"

using namespace cv;
using namespace std;

class Phase2Pcd
{
	Phase2Pcd();//���캯����ʼ��Q
	~Phase2Pcd();
	//pcl::PointCloud<pcl::PointXYZ>::Ptr Phase2Pcd
	Mat ComputeCost(Mat phase_left, Mat phase_right);//�������
	Mat CostAggregation(Mat cost_map);//���۾ۺ�
	vector<Vec4d> ComputeDisparity(Mat aggregated_cost_map);//�����Ӳ�:�������Ӳϸ��������һ���Լ��
	pcl::PointCloud<pcl::PointXYZ>::Ptr Disparity2PCD(vector<Vec4d> points);

};

#endif