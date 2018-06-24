/*
Senior Project
#21 Team WillChair
Fall 2017 to Spring 2018

Computer vision software part by
Zhuohao Yang
*/

#include "stdafx.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/features2d/features2d.hpp"
#include <string>
#include <vector>
#include <iostream>
#define _USE_MATH_DEFINES
#include <math.h>
#include <stack>
#include <fstream>

using namespace cv;
using namespace std;

// helper functions 
void convertHSL(int b, int g, int r, int& hue, int& sat, int& lit);

// edge detection main functions
void SobelEdgeDetection(const Mat& src, Mat& rst, int thresh_hue);

// FloodFill function
void myFloodFill(const Mat &img, vector<Point2i> &blob);

// Template match function
void tpmatch(Mat& final_display, Point& match_start, Point& match_end, const Mat& templ1, const Mat& templ2, const Mat& templ3, const Mat& cmp, const int& draw_ctr, double& rst_val);

// The main function
int main()
{
	const double max_img_pixel = 600.0;
	const int window_choice = 0;
	const int write_ctr = 0;				// save the output pictures to files
	const int draw_box_ctr = 1;				// draw the bounding box of the handle found
	const int draw_largest_blob = 1;		// find the 'door' or 'all found componets'
	const int hue_refine = 0;				// 
	const int open_ctr = 1;					// dilate then erode

	VideoCapture cap(0);					// this is the default camera
	//VideoCapture cap(1);

	// if not successful, exit program
	if (!cap.isOpened())
	{
		cout << "Cannot open the video cam" << endl;
		return -1;
	}

	//create a window called "MyVideoFrame0"
	namedWindow("MyVideo0", WINDOW_AUTOSIZE);
	Mat frame0;

	// read a new frame from video
	bool bSuccess0 = cap.read(frame0);

	//if not successful, break loop
	if (!bSuccess0)
	{
		cout << "Cannot read a frame from video stream" << endl;
	}

	//show the frame in "MyVideo" window
	imshow("MyVideo0", frame0);

	ifstream in_status("status.txt");
	int in_state;
	in_status >> in_state;
	if (in_state == 0) {
		cout << "State is 0, set it to 1 to start" << endl;
		return 0;
	}
	in_status.close();

	Mat tempr_3 = imread("./door/template_doorknob_right.png");
	Mat templ_3 = imread("./door/template_doorknob_left.png");
	Mat sticker3 = imread("./door/template_sticker1.png");
	Mat tempr_2, tempr_4, templ_2, templ_4, sticker1, sticker2;

	// Scale the template image
	int col_r2 = tempr_3.cols * 3 / 4; 	int row_r2 = tempr_3.rows * 3 / 4;
	int col_r4 = tempr_3.cols * 4 / 3; 	int row_r4 = tempr_3.rows * 4 / 3;
	Size sizer_2(row_r2, col_r2);
	Size sizer_4(row_r4, col_r4);
	resize(tempr_3, tempr_2, sizer_2);
	resize(tempr_3, tempr_4, sizer_4);

	int col_l2 = templ_3.cols * 3 / 4; 	int row_l2 = templ_3.rows * 3 / 4;
	int col_l4 = templ_3.cols * 4 / 3; 	int row_l4 = templ_3.rows * 4 / 3;
	Size sizel_2(row_l2, col_l2);
	Size sizel_4(row_l4, col_l4);
	resize(templ_3, templ_2, sizel_2);
	resize(templ_3, templ_4, sizel_4);

	int col_s2 = sticker3.cols * 3 / 4; 	int row_s2 = sticker3.rows * 3 / 4;
	int col_s1 = sticker3.cols * 9 /16; 	int row_s1 = sticker3.rows * 9 /16;
	Size sizes_2(row_s2, col_s2);
	Size sizes_1(row_s1, col_s1);
	resize(sticker3, sticker2, sizes_2);
	resize(sticker3, sticker1, sizes_1);

	while (1)
	{
		// read a new frame from video
		Mat original;
		bool bSuccess = cap.read(original);
		imshow("MyVideo0", original);
		//if not successful, break loop
		if (!bSuccess)
		{
			cout << "Cannot read a frame from video stream" << endl;
			break;
		}

		// Scale the input image if the size is too large 
		double scale = 1.0;
		int max_side = original.cols;
		if (original.rows > original.cols)
			max_side = original.rows;
		if (max_side > max_img_pixel)
			scale = max_img_pixel / max_side;
		int resize_row = original.rows*scale;
		int resize_col = original.cols*scale;
		Size our_resize_num(resize_col, resize_row);
		resize(original, original, our_resize_num);
		//*/

		if (window_choice == 1) {
			namedWindow("Detected_region", WINDOW_NORMAL);
			namedWindow("Edge", WINDOW_NORMAL);
			namedWindow("Largest_Flood_Object", WINDOW_NORMAL);
			namedWindow("Output", WINDOW_NORMAL);
		}
		else if (window_choice == 2) {
			namedWindow("Detected_region", WINDOW_AUTOSIZE);
			namedWindow("Edge", WINDOW_AUTOSIZE);
			namedWindow("Largest_Flood_Object", WINDOW_AUTOSIZE);
			namedWindow("Output", WINDOW_AUTOSIZE);
		}
		else {

		}

		// use black and white version to create the edge picture
		Mat img_edge;
		img_edge.create(original.rows, original.cols, CV_8UC1);
		Mat gray; cvtColor(original, gray, cv::COLOR_BGR2GRAY);
		SobelEdgeDetection(gray, img_edge, 50);
		//imshow("Edge", img_edge);

		
		// flood fill the binary edge image and store pixels in vector 'blobs'
		vector<Point2i> blob;
		myFloodFill(img_edge, blob);

		// used the vector 'blob' to create a flood fill result mat
		Mat largest_blob = Mat::zeros(img_edge.size(), CV_8UC3);
		// if hue refine needed
		Mat img_hue = Mat::zeros(img_edge.size(), CV_8UC1);
		if (hue_refine == 1) {
			for (int i = 0; i < img_edge.rows; i++) {
				for (int j = 0; j < img_edge.cols; j++) {
					int r, g, b, h, s, l;
					b = original.at<Vec3b>(i, j)[0];
					g = original.at<Vec3b>(i, j)[1];
					r = original.at<Vec3b>(i, j)[2];
					convertHSL(b, g, r, h, s, l);
					img_hue.at<uchar>(i, j) = h;
				}
			}
		}
		int door_hue_avg = 0;
		// Color the found largest blob
		Scalar color = Scalar(57, 169, 40);
		for (int k = 0; k < blob.size(); k++) {
			if (hue_refine == 1)
				door_hue_avg += img_hue.at<uchar>(blob[k].x, blob[k].y);
			largest_blob.at<Vec3b>(blob[k].x, blob[k].y)[0] = (uchar) color[0];
			largest_blob.at<Vec3b>(blob[k].x, blob[k].y)[1] = (uchar) color[1];
			largest_blob.at<Vec3b>(blob[k].x, blob[k].y)[2] = (uchar) color[2];
		}
		// find the 'average hue'
		if (hue_refine == 1) {
			door_hue_avg /= (int) blob.size();
		}
		//imshow("Largest_Flood_Object", largest_blob);
		
		
		// refine the flood fill output mat with 'open' operation
		if (open_ctr == 1) {
			// Closing operation for better output
			int erosion_size = 1;
			int dilation_size = 1;
			Mat element = getStructuringElement(MORPH_RECT,
				Size(2 * erosion_size + 1, 2 * erosion_size + 1),
				Point(erosion_size, erosion_size));
			dilate(largest_blob, largest_blob, element);
			dilate(largest_blob, largest_blob, element);
			dilate(largest_blob, largest_blob, element);
			erode(largest_blob, largest_blob, element);
			erode(largest_blob, largest_blob, element);
		}
		//imshow("Largest_Flood_Object", largest_blob);

		
		// Template match to find the doorknob
		Point match_start, match_end, temp_start, temp_end; 
		double rst_val1 = 1.0, rst_val2 = 1.0;

		tpmatch(original, temp_start, temp_end, templ_2, templ_3, templ_4, largest_blob, draw_box_ctr, rst_val2);
		cout << "left handle match: " << rst_val2 << "\t\t";
		//bashHistory << "left handle match: " << rst_val2 << "\t\t";

		tpmatch(original, match_start, match_end, tempr_2, tempr_3, tempr_4, largest_blob, draw_box_ctr, rst_val1);
		cout << "right handle match: " << rst_val1 << endl;
		//bashHistory << "right handle match: " << rst_val1 << "endl;
		if (rst_val2 < rst_val1) { 
			match_start.x = temp_start.x; match_start.y = temp_start.y;  
			match_end.x = temp_end.x; match_end.y = temp_end.y;
		}
		bool cantfind = false;
		if (rst_val1 > 0.3 && rst_val2 > 0.3)
			cantfind = true;

		// Template match to find the sticker
		Mat gray3c;
		cvtColor(gray, gray3c, cv::COLOR_GRAY2BGR);
		Point sticker_start, sticker_end; 
		double sticker_rst_val;
		tpmatch(original, sticker_start, sticker_end, sticker1, sticker2, sticker3, gray3c, draw_box_ctr, sticker_rst_val);
		//imshow("Edge", original);


		if (!cantfind) {
			cout << " found " << endl;

			int diff = 0, dir = 0, magni = 10, out_state;
			int sticker_pixel_loc_x, sticker_pixel_loc_y;
			int handle_pixel_loc_x, handle_pixel_loc_y;
			int sticker_mid_x = (sticker_start.x + sticker_end.x) / 2;
			int handle_mid_x = (match_start.x + match_end.x) / 2;
			int handle_mid_y = (match_start.y + match_end.y) / 2;

			ofstream status("status.txt");
			if (in_state == 1) {
				status << 2 << " " << sticker_mid_x << " " << sticker_end.y << endl;
				cout << 2 << " " << sticker_mid_x << " " << sticker_end.y << endl;
			}
			else if (in_state == 2) {
				ofstream instructions("instructions.txt");
				ifstream in_status("status.txt");
				int dummy, prev_x, prev_y;
				in_status >> dummy >> prev_x >> prev_y;
				int pixel_scale = 1 + prev_y - sticker_end.y;

				int diff, magni;
				diff = sticker_end.y - match_start.y;
				if (diff > 0) {													// stikcer is below the handle
					magni = (10 * diff) / pixel_scale;
					instructions << 1 << " split " << magni << " split ";		// move up
					cout << 1 << " " << magni << " --- ";
				}
				else {
					instructions << 1 << " split " << 0 << " split ";			// don't move
					cout << 1 << " " << 0 << " --- ";
				}

				diff = sticker_mid_x - handle_mid_x;
				if (diff > 0) {													// sticker is on the right of the handle
					magni = (10 * diff) / pixel_scale;
					instructions << 3 << " split " << magni << " split ";		// move left
					cout << 3 << " " << magni << " --- ";
				}
				else {															// sticker is on the left of the handle
					diff = 0 - diff;
					magni = (10 * diff) / pixel_scale;
					instructions << 4 << " split " << magni << " split ";		// move right
					cout << 4 << " " << magni << " --- ";
				}

				diff = handle_mid_y - match_start.y;
				magni = (10 * diff) / pixel_scale;
				instructions << 2 << " split " << magni;						// move down
				cout << 1 << " " << magni << endl;

				instructions.close();
				status << 3;
			}
			else {
				cout << " default state " << endl;
				status << 1 << endl;
			}
			status.close();
		}

		if (waitKey(30) == 27)
		{
			cout << "esc key is pressed by user" << endl;
			break;
		}
	}
	cap.release();
	return 0;
}

void SobelEdgeDetection(const Mat& src, Mat& rst, int thresh_sob) {
	rst = Mat::zeros(src.size(), CV_8UC1);
	for (int i = 1; i < src.rows - 1; i++) {
		for (int j = 1; j < src.cols - 1; j++) {
			int gray1 = src.at<uchar>(i - 1, j - 1);
			int gray2 = src.at<uchar>(i - 1, j);
			int gray3 = src.at<uchar>(i - 1, j + 1);
			int gray4 = src.at<uchar>(i, j - 1);
			int gray5 = src.at<uchar>(i, j);
			int gray6 = src.at<uchar>(i, j + 1);
			int gray7 = src.at<uchar>(i + 1, j - 1);
			int gray8 = src.at<uchar>(i + 1, j);
			int gray9 = src.at<uchar>(i + 1, j + 1);
			int sob_x = abs(gray1 + 2 * gray4 + gray7 - gray3 - 2 * gray6 - gray9);
			int sob_y = abs(gray7 + 2 * gray8 + gray9 - gray1 - 2 * gray2 - gray3);
			int sob_res = (int) sqrt(sob_x * sob_x + sob_y * sob_y);
			//rst.at<uchar>(i, j) = sob_res;
			if (sob_res > thresh_sob)
				rst.at<uchar>(i, j) = 255;
		}
	}

	for (int i = 0; i < src.rows; i++) {
		rst.at<uchar>(i, 0) = 255;
		rst.at<uchar>(i, src.cols - 1) = 255;
	}
	for (int j = 0; j < src.cols; j++) {
		rst.at<uchar>(0, j) = 255;
		rst.at<uchar>(src.rows - 1, j) = 255;
	}
}

void myFloodFill(const Mat &img, vector<Point2i> &blob) {
	blob.clear();
	Mat imgcopy;
	img.copyTo(imgcopy);
	unsigned char wanted_val = 0;
	stack<Point2i> mystack;
	int max_size = 0;
	int curr_size = 0;
	vector<Point2i> temp_blob;

	for (int i = 0; i < imgcopy.rows; i++) {
		for (int j = 0; j < imgcopy.cols; j++) {
			if (imgcopy.ptr<uchar>(i)[j] == wanted_val) {
				curr_size = 1;
				imgcopy.at<uchar>(i, j) = 255 - wanted_val;
				Point2i myp;
				myp.x = i;
				myp.y = j;
				mystack.push(myp);
				temp_blob.clear();
				temp_blob.push_back(myp);
				Point2i pt2, pt4, pt5, pt6, pt8;
				while (!mystack.empty()) {
					pt5 = mystack.top();
					mystack.pop();
					if (pt5.x > 0) {
						pt2.x = pt5.x - 1;
					}
					else {
						pt2.x = pt5.x;
					}
					if (pt5.x < imgcopy.rows - 1) {
						pt8.x = pt5.x + 1;
					}
					else {
						pt8.x = pt5.x;
					}
					if (pt5.y > 0) {
						pt4.y = pt5.y - 1;
					}
					else {
						pt4.y = pt5.y;
					}
					if (pt5.y < imgcopy.cols - 1) {
						pt6.y = pt5.y + 1;
					}
					else {
						pt6.y = pt5.y;
					}
					pt4.x = pt5.x;			pt2.y = pt5.y;
					pt6.x = pt5.x;			pt8.y = pt5.y;

					if (imgcopy.at<uchar>(pt2.x, pt2.y) == wanted_val) {
						imgcopy.at<uchar>(pt2.x, pt2.y) = 255 - wanted_val;
						mystack.push(pt2);
						temp_blob.push_back(pt2);
						curr_size++;
					}
					if (imgcopy.at<uchar>(pt4.x, pt4.y) == wanted_val) {
						imgcopy.at<uchar>(pt4.x, pt4.y) = 255 - wanted_val;
						mystack.push(pt4);
						temp_blob.push_back(pt4);
						curr_size++;
					}
					if (imgcopy.at<uchar>(pt6.x, pt6.y) == wanted_val) {
						imgcopy.at<uchar>(pt6.x, pt6.y) = 255 - wanted_val;
						mystack.push(pt6);
						temp_blob.push_back(pt6);
						curr_size++;
					}
					if (imgcopy.at<uchar>(pt8.x, pt8.y) == wanted_val) {
						imgcopy.at<uchar>(pt8.x, pt8.y) = 255 - wanted_val;
						mystack.push(pt8);
						temp_blob.push_back(pt8);
						curr_size++;
					}
				}
				//cout << "blob size=" << size << endl;;
				//store the blob in the vector of blobs
				if (curr_size > max_size) {
					max_size = curr_size;
					blob = temp_blob;
				}
			}
		}
	}
	//cout << "door pixel size = " << blob.size() << endl;
	//namedWindow("Trial");
	//imshow("Trial", imgcopy);
}

void convertHSL(int b, int g, int r, int& hue, int& sat, int& lit) {

	double B = b / 255.0;
	double G = g / 255.0;
	double R = r / 255.0;
	double hsl_max, hsl_min, delta;

	// find hue_max
	if (B >= G && B >= R) hsl_max = B;
	else if (G >= B && G >= R) hsl_max = G;
	else hsl_max = R;

	// find hue_min
	if (B <= G && B <= R) hsl_min = B;
	else if (G <= B && G <= R) hsl_min = G;
	else hsl_min = R;

	delta = hsl_max - hsl_min;

	lit = (int) (hsl_max + hsl_min) * 50;
	double litt = (hsl_max + hsl_min) / 2;
	if (delta == 0) sat = 0;
	else sat = (int) (100 * delta / (1 - abs(litt * 2 - 1)));

	if (delta == 0)	hue = 0;
	else if (hsl_max == R) {
		double temp = (G - B) / delta;
		hue = (int) (60 * temp);
	}
	else if (hsl_max == G)
		hue = (int) (60 * (((B - R) / delta) + 2));
	else if (hsl_max == B)
		hue = (int) (60 * (((R - G) / delta) + 4));
	else
		cout << "HUE function error" << endl;

	return;
}

void tpmatch(Mat& final_display, Point& match_start, Point& match_end, const Mat& templ1, const Mat& templ2, const Mat& templ3, const Mat& cmp, const int& draw_ctr, double& rst_val) {

	/// Create the result matrix
	Mat result1;
	Mat result2;
	Mat result3;
	int result_cols1 = final_display.cols - templ1.cols + 1;
	int result_rows1 = final_display.rows - templ1.rows + 1;
	int result_cols2 = final_display.cols - templ2.cols + 1;
	int result_rows2 = final_display.rows - templ2.rows + 1;
	int result_cols3 = final_display.cols - templ3.cols + 1;
	int result_rows3 = final_display.rows - templ3.rows + 1;
	result1.create(result_rows1, result_cols1, CV_8UC1);
	result2.create(result_rows2, result_cols2, CV_8UC1);
	result3.create(result_rows3, result_cols3, CV_8UC1);

	/// Do the Matching and Normalize
	int match_method = 1;
	matchTemplate(cmp, templ1, result1, match_method);
	matchTemplate(cmp, templ2, result2, match_method);
	matchTemplate(cmp, templ3, result3, match_method);

	/// Localizing the best match with minMaxLoc
	double minVal1; double maxVal1; Point minLoc1; Point maxLoc1;
	double minVal2; double maxVal2; Point minLoc2; Point maxLoc2;
	double minVal3; double maxVal3; Point minLoc3; Point maxLoc3;
	minMaxLoc(result1, &minVal1, &maxVal1, &minLoc1, &maxLoc1, Mat());
	minMaxLoc(result2, &minVal2, &maxVal2, &minLoc2, &maxLoc2, Mat());
	minMaxLoc(result3, &minVal3, &maxVal3, &minLoc3, &maxLoc3, Mat());

	// template 1 is the best match
	if (minVal1 < minVal2 && minVal1 < minVal3) {
		rst_val = minVal1;
		match_start = minLoc1;
		match_end.x = match_start.x + templ1.cols;
		match_end.y = match_start.y + templ1.rows;
		//rectangle(final_display, matchLoc, Point(matchLoc.x + templ1.cols, matchLoc.y + templ1.rows), Scalar::all(255), 2, 8, 0);
		if (draw_ctr == 1)
			rectangle(final_display, match_start, match_end, Scalar::all(255), 2, 8, 0);
	}
	// template 2 is the best match
	else if (minVal2 < minVal1 && minVal2 < minVal3) {
		rst_val = minVal2;
		match_start = minLoc2;
		match_end.x = match_start.x + templ2.cols;
		match_end.y = match_start.y + templ2.rows;
		//rectangle(final_display, matchLoc, Point(matchLoc.x + templ2.cols, matchLoc.y + templ2.rows), Scalar::all(255), 2, 8, 0);
		if (draw_ctr == 1)
			rectangle(final_display, match_start, match_end, Scalar::all(255), 2, 8, 0);
	}
	// template 3 is the best match
	else {
		rst_val = minVal3;
		match_start = minLoc3;
		match_end.x = match_start.x + templ3.cols;
		match_end.y = match_start.y + templ3.rows;
		//rectangle(final_display, matchLoc, Point(matchLoc.x + templ3.cols, matchLoc.y + templ3.rows), Scalar::all(255), 2, 8, 0);
		if (draw_ctr == 1)
			rectangle(final_display, match_start, match_end, Scalar::all(255), 2, 8, 0);
	}

}
