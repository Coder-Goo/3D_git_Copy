#pragma once
#ifndef PHASE2PCD
#define PHASE2PCD
#include"biao_tou.h"

using namespace cv;
using namespace std;

class Phase2Pcd
{
	Phase2Pcd();//构造函数初始化Q
	~Phase2Pcd();
	//pcl::PointCloud<pcl::PointXYZ>::Ptr Phase2Pcd
	Mat ComputeCost(Mat phase_left, Mat phase_right);//计算代价
	Mat CostAggregation(Mat cost_map);//代价聚合
	vector<Vec4d> ComputeDisparity(Mat aggregated_cost_map);//计算视差:包括：视差精细化，左右一致性检查
	pcl::PointCloud<pcl::PointXYZ>::Ptr Disparity2PCD(vector<Vec4d> points);

};

#endif