// Region_Growing_3D_VS2012.cpp : Defines the entry point for the console application.
//



#include "stdafx.h"


#ifdef _DEBUG
#pragma comment(lib, "opencv_core249d.lib")
#pragma comment(lib, "opencv_imgproc249d.lib")
//#pragma comment(lib, "opencv_objdetect249d.lib")
//#pragma comment(lib, "opencv_gpu246d.lib")
//#pragma comment(lib, "opencv_features2d249d.lib")
#pragma comment(lib, "opencv_highgui249d.lib")
//#pragma comment(lib, "opencv_flann249d.lib")
#else
#pragma comment(lib, "opencv_core249.lib")
#pragma comment(lib, "opencv_imgproc249.lib")
//#pragma comment(lib, "opencv_objdetect246.lib")
//#pragma comment(lib, "G:\\OPENCV\\lib\\Release\\opencv_gpu246.lib")
//#pragma comment(lib, "opencv_features2d246.lib")
#pragma comment(lib, "opencv_highgui249.lib")
//#pragma comment(lib, "opencv_flann246.lib")
#endif

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
//#include <opencv2/gpu/gpu.hpp> 

#include <iostream>
#include <fstream>

using namespace cv;
using namespace std;

#define PLANE_SIZE 512
#define NEIGHBOURHOOD_SIZE 2 // (3 - 1)
#define THRESHOLD 100
#define THRESHOLD_DYNAMIC 4.0
#define POTENTIAL_POSITIVE_RADIUS 150
#define POTENTIAL_NEGATIVE_RADIUS 70

ofstream log_file;

//struct RecursiveData
//{
//	RecursiveData();
//	
//	RecursiveData(uchar* originalImage, float* potentialPlane, uchar* maskImage, int current_x, int current_y)
//	{
//	
//	};
//
//	uchar* originalImage;
//	float* potentialPlane;
//	uchar* maskImage;
//	int current_x;
//	int current_y;
//
//};


struct myPoint{
	
	myPoint()
	{
		x = 0;
		y = 0;
	};
	
	myPoint(int column, int row)
	{
		x = column;
		y = row;
	};

	int x;
	int y;
};

void RecursiveRegionGrowth (uchar* originalImage, float* potentialPlane, uchar* maskImage, int current_x, int current_y, int imgWidth, int imgHeight, float threshold)
{
	//cout << current_x << " " << current_y << endl;
	//log_file  << current_x << " " << current_y << endl;

	for (int i = -1; i < NEIGHBOURHOOD_SIZE; i++)
		for (int j = -1; j < NEIGHBOURHOOD_SIZE; j++)
			//begin if statement
				if (   (abs(originalImage[(current_y + i) * imgWidth + current_x + j] - originalImage[(current_y) * imgWidth + current_x])  // dynamic thresholding
						<																													// 
				   (threshold * potentialPlane[(current_y + i) * imgWidth + current_x + j]))										// 
				   //(originalImage[(current_y + i) * PLANE_SIZE + current_x + j] < THRESHOLD_DYNAMIC)										// const thresholding
				&& (maskImage[(current_y + i) * imgWidth + current_x + j] != 255)	// check if this is already processed or not
 				//&& (abs(i) != abs(j)) // for 4 pixel neighbourhood, if removed then 8 points neighbourhood
				)
			//end if statement
			{
				maskImage[(current_y + i) * imgWidth + current_x + j] = 255;

				RecursiveRegionGrowth(originalImage, potentialPlane, maskImage, current_x + j, current_y + i, imgWidth, imgHeight, threshold);
			};
};

/*
void RecursiveRegionGrowth_simple (uchar* originalImage, float* potentialPlane, uchar* maskImage, int current_x, int current_y)
{
	//cout << current_x << " " << current_y << endl;

	maskImage[(current_y) * PLANE_SIZE + current_x] = 255;
	
	current_x -= 1; //recurse left
	
	if ((originalImage[(current_y) * PLANE_SIZE + current_x] > THRESHOLD ) && (maskImage[(current_y) * PLANE_SIZE + current_x] != 255))
		RecursiveRegionGrowth_simple (originalImage, potentialPlane, maskImage, current_x, current_y);

	current_x += 2; //recursive right
	
	if ((originalImage[(current_y) * PLANE_SIZE + current_x] > THRESHOLD ) && (maskImage[(current_y) * PLANE_SIZE + current_x] != 255))
		RecursiveRegionGrowth_simple (originalImage, potentialPlane, maskImage, current_x, current_y);

	current_x -= 1; 
	current_y -= 1; //recursive up

	if ((originalImage[(current_y) * PLANE_SIZE + current_x] > THRESHOLD ) && (maskImage[(current_y) * PLANE_SIZE + current_x] != 255))
		RecursiveRegionGrowth_simple (originalImage, potentialPlane, maskImage, current_x, current_y);

	current_y += 2; //recursive down

	if ((originalImage[(current_y) * PLANE_SIZE + current_x] > THRESHOLD ) && (maskImage[(current_y) * PLANE_SIZE + current_x] != 255))
		RecursiveRegionGrowth_simple (originalImage, potentialPlane, maskImage, current_x, current_y);
};
*/

int PotentialPlaneGenerating (float* potentialPlane, vector <myPoint> positivePoints, vector <myPoint> negativePoints, int positiveRadius, int negativeRadius, int imgWidth, int imgHeight)
{
	for (int i = 0; i < positivePoints.size(); i++)
	{
		for (int y = -positiveRadius; y < positiveRadius;  y++)
			for (int x = -positiveRadius; x < positiveRadius; x++)
			{
				if ((positivePoints[i].x + x < 0) || (positivePoints[i].x + x > imgWidth) || (positivePoints[i].y + y < 0) || (positivePoints[i].y + y > imgHeight))
					continue;

				float distance = sqrt(x * x  + y * y);
				
				if (distance < positiveRadius)
					potentialPlane[(positivePoints[i].y + y) * imgWidth + positivePoints[i].x + x] += (positiveRadius - distance) / positiveRadius ;
				if (potentialPlane[(positivePoints[i].y + y) * imgWidth + positivePoints[i].x + x] > 1)
					potentialPlane[(positivePoints[i].y + y) * imgWidth + positivePoints[i].x + x] = 1.0;
			}
	}

	for (int i = 0; i < negativePoints.size(); i++)
	{


		for (int y = -negativeRadius; y < negativeRadius;  y++)
			for (int x = -negativeRadius; x < negativeRadius; x++)
			{
				if ((negativePoints[i].x + x < 0) || (negativePoints[i].x + x > imgWidth) || (negativePoints[i].y + y < 0) || (negativePoints[i].y + y > imgHeight))
					continue;

				float distance = sqrt(x * x  + y * y);
				
				if (distance < negativeRadius)
					{
						potentialPlane[(negativePoints[i].y + y) * imgWidth + negativePoints[i].x + x] -= (negativeRadius - distance) / negativeRadius ;
						if (potentialPlane[(negativePoints[i].y + y) * imgWidth + negativePoints[i].x + x] < 0) 
							potentialPlane[(negativePoints[i].y + y) * imgWidth + negativePoints[i].x + x] = 0;
					}
			}
	}

	return 1;
};

int _tmain(int argc, _TCHAR* argv[])
{
	log_file.open("log.txt");

	// The following 4 rows are for creating sample image
	Mat img(Mat::ones(PLANE_SIZE, PLANE_SIZE, CV_8UC1));

	for (int i = 100; i < img.rows - 100; i++)
		for (int j = 100; j < img.cols - 100; j++)
			img.at<unsigned char>(i, j) = 157;

	// Here, we don't pay attention at the initialized sample and read new image from file
	Mat dicomImage = imread("..//images//axial_slice_0003.png", 0);

	vector<myPoint> positive_points; // Vector for points with positive influence on potential plane
	vector<myPoint> negative_points; // Vector for points with negative influence on potential plane

	positive_points.push_back(myPoint(150, 220)); // initializing some sample points
	positive_points.push_back(myPoint(220, 150));
	
	negative_points.push_back(myPoint(27, 160));


	Mat potentialPlane(Mat::zeros(PLANE_SIZE, PLANE_SIZE, CV_32F)); // Potential plane is a float-type matrix initialized with zeros
	Mat maskImage(Mat::zeros(PLANE_SIZE, PLANE_SIZE, CV_8UC1)); // maskImage is an unsigned char matrix initialized with zeros
																// We need it for storing the segmentation results


	// Initializing potential plane according to preset radius and points coordinates
	PotentialPlaneGenerating((float*)(potentialPlane.data), positive_points, negative_points, POTENTIAL_POSITIVE_RADIUS, POTENTIAL_NEGATIVE_RADIUS, img.cols, img.rows);

	// We need to launch recursive region growing for each positive point
	for (int positivePointNumber = 0; positivePointNumber < positive_points.size(); positivePointNumber++)
		RecursiveRegionGrowth(dicomImage.data, (float*)(potentialPlane.data), maskImage.data, positive_points[positivePointNumber].x, positive_points[positivePointNumber].y, img.cols, img.rows, THRESHOLD_DYNAMIC);

	// Now we have the resulting segment in (uchar*)maskImage


	// The rest routines are for visualizng
	double minVal, maxVal;
	minMaxLoc(potentialPlane, &minVal, &maxVal); //find minimum and maximum intensities
	Mat drawPotential;
	potentialPlane.convertTo(drawPotential, CV_8U, 255.0/(maxVal - minVal), -minVal * 255.0/(maxVal - minVal));

	cvtColor(dicomImage, dicomImage, CV_GRAY2BGR);

	for (int positivePointNumber = 0; positivePointNumber < positive_points.size(); positivePointNumber++)
	{
		circle(dicomImage, Point(positive_points[positivePointNumber].x, positive_points[positivePointNumber].y), 3, Scalar(0, 255, 0), 2);
		circle(dicomImage, Point(positive_points[positivePointNumber].x, positive_points[positivePointNumber].y), POTENTIAL_POSITIVE_RADIUS, Scalar(0, 255, 0), 2);
	}

	for (int negativePointNumber = 0; negativePointNumber < negative_points.size(); negativePointNumber++)
	{
		circle(dicomImage, Point(negative_points[negativePointNumber].x, negative_points[negativePointNumber].y), 3, Scalar(0, 0, 255), 2);
		circle(dicomImage, Point(negative_points[negativePointNumber].x, negative_points[negativePointNumber].y), POTENTIAL_NEGATIVE_RADIUS, Scalar(0, 0, 255), 2);
	}

	imshow("output", dicomImage);
	imwrite("..//images//dicomImage.png", dicomImage);

	imshow("mask", maskImage);
	imwrite("..//images//segmentationMask.png", maskImage);

	imshow("Potential", drawPotential);
	imwrite("..//images//potential.png", drawPotential);
	waitKey();
	
	log_file.close();
	return 0;
}

