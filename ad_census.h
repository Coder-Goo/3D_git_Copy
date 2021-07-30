#pragma once
#include"calibration.h"
#include"aggregation.h"
#include"phase_unwrapping2.h"

const uchar  cost_window_size =7;
const int aggeragation_window_size = 7;
const float LMADA_AD = 1.0;
const float  LMADA_CENSUS = 70.0;
const float LMADA_RANK = 10.0;
//视差范围
const int min_disparity = -50;
const int max_disparity = 900;
//自适应窗口阈值
const double color_threshold1 = 0.6;
const double color_threshold2 = 0.8;
const double color_threshold3 = 0.2;
const int maxlength1 = 9;
const int maxlength2 = 5;//maxlength2<maxlength1且color_threshold1>color_threshold

const int INVALID_DISPARITY = 6550;

const bool ad_census = true;
const bool fixed_window = false;

inline float compute_ad_census_cost(int& left_row, int& left_col, int &right_row, int& right_col,const  Mat &phase_left, const  Mat &phase_right,const uchar &win_size);

//计算代价
void CostCalculation(vector<vector<Mat>>&vec_cost_maps, const Mat &phase_left, const Mat &phase_right, const uchar &cost_window_size);
//计算左视图的DSI
void CalculateRightDsi(vector<vector<Mat>>&vec_right_dsi, const vector<vector<Mat>>&vec_aggregation_maps, const Mat& phase_left, const Mat &phase_right);
//代价聚合
void CostAggregation(bool fixed_window, vector<vector<Mat>>&vec_cost_maps, vector<vector<Mat>>&vec_aggregation_maps, const Mat &phase_left, const Mat &phase_right, const uchar &aggregation_window_size);
//视差计算
Mat DisparityCalculation(const vector<vector<Mat>>&vec_aggregation_maps, const Mat &phase_left, const Mat &phase_right);