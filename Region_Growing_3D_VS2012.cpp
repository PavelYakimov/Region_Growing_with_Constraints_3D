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
#define THRESHOLD_DYNAMIC 2.0
#define POTENTIAL_POSITIVE_RADIUS 100
#define POTENTIAL_NEGATIVE_RADIUS 50

#define DIM_3 1

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
		z = 0;
	};
	
	myPoint(int column, int row)
	{
		x = column;
		y = row;
		z = 0;
	};

	myPoint(int column, int row, int z_coordinate)
	{
		x = column;
		y = row;
		z = z_coordinate;
	};

	int x;
	int y;
	int z;
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

void RecursiveRegionGrowth3D (uchar** originalImage, float** potentialPlane, uchar** maskImage, int current_x, int current_y, int current_z, int imgWidth, int imgHeight, int amountOfSlices, float threshold)
{
	//cout << current_x << " " << current_y << endl;
	//log_file  << current_x << " " << current_y << endl;
	if ((current_x < 1) || (current_x > imgWidth - 2) || (current_y < 1) || (current_y > imgHeight - 2) || (current_z < 1) || (current_z > amountOfSlices - 2)) return;

	for (int i = -1; i < NEIGHBOURHOOD_SIZE; i++)
		for (int j = -1; j < NEIGHBOURHOOD_SIZE; j++)
			for (int k = -1; k < NEIGHBOURHOOD_SIZE; k++)
				//begin if statement
					if ( (abs(originalImage[current_z + k][(current_y + i) * imgWidth + current_x + j] - originalImage[current_z][(current_y) * imgWidth + current_x])  // dynamic thresholding
						   <																													// 
						   (threshold * potentialPlane[current_z + k][(current_y + i) * imgWidth + current_x + j]))										// 
						 //(originalImage[(current_y + i) * PLANE_SIZE + current_x + j] < THRESHOLD_DYNAMIC)										// const thresholding
						 && (maskImage[current_z + k][(current_y + i) * imgWidth + current_x + j] != 255)	// check if this is already processed or not
 					   //&& (abs(i) != abs(j)) // for 4 pixel neighbourhood, if removed then 8 points neighbourhood
					   )
				//end if statement
					{
						maskImage[current_z + k][(current_y + i) * imgWidth + current_x + j] = 255;

						RecursiveRegionGrowth3D(originalImage, potentialPlane, maskImage, current_x + j, current_y + i, current_z + k, imgWidth, imgHeight, amountOfSlices, threshold);
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

int PotentialPlaneGenerating3Dnegative (float* potentialPlane, myPoint negativePoints3D, int negativeRadius, int imgWidth, int imgHeight, int amountOfSlices)
{
	for (int y = -negativeRadius; y < negativeRadius;  y++)
		for (int x = -negativeRadius; x < negativeRadius; x++)
		{
			if ((negativePoints3D.x + x < 0) || (negativePoints3D.x + x > imgWidth) || (negativePoints3D.y + y < 0) || (negativePoints3D.y + y > imgHeight))
				continue;

			float distance = sqrt(x * x  + y * y);
				
			if (distance < negativeRadius)
			{
				potentialPlane[(negativePoints3D.y + y) * imgWidth + negativePoints3D.x + x] -= (negativeRadius - distance) / negativeRadius ;
				if (potentialPlane[(negativePoints3D.y + y) * imgWidth + negativePoints3D.x + x] < 0) 
					potentialPlane[(negativePoints3D.y + y) * imgWidth + negativePoints3D.x + x] = 0;
			}
		}

	return 1;
};

int PotentialPlaneGenerating3Dpositive (float* potentialPlane, myPoint positivePoints3D, int positiveRadius, int imgWidth, int imgHeight, int amountOfSlices)
{
	for (int y = -positiveRadius; y < positiveRadius;  y++)
		for (int x = -positiveRadius; x < positiveRadius; x++)
		{
			if ((positivePoints3D.x + x < 0) || (positivePoints3D.x + x > imgWidth) || (positivePoints3D.y + y < 0) || (positivePoints3D.y + y > imgHeight))
				continue;

			float distance = sqrt(x * x  + y * y);
				
			if (distance < positiveRadius)
			{
				potentialPlane[(positivePoints3D.y + y) * imgWidth + positivePoints3D.x + x] += (positiveRadius - distance) / positiveRadius ;
				if (potentialPlane[(positivePoints3D.y + y) * imgWidth + positivePoints3D.x + x] > 1)
					potentialPlane[(positivePoints3D.y + y) * imgWidth + positivePoints3D.x + x] = 1.0;
			}
		}
	return 1;
}

int _tmain(int argc, _TCHAR* argv[])
{
	log_file.open("log.txt");


// 3D ALGORITHM
#ifdef DIM_3
	int amount_of_slices = 170;
	int first_slice = 330;

	Mat* imageSeries = new Mat[amount_of_slices];
	
	char* slicename = new char[20];

	for (int i = 0; i < amount_of_slices; i++)
	{
		sprintf(slicename, "..//images//3D//axial0%d.png", first_slice + i);
		imageSeries[i] = imread(slicename, 0);
		//imshow("OrigData", (imageSeries[i]) );
		//waitKey();
	}

	vector<myPoint> positive_points_3d; // Vector for points with positive influence on potential plane
	vector<myPoint> negative_points_3d; // Vector for points with negative influence on potential plane

	// initializing some sample points
	positive_points_3d.push_back(myPoint(255, 311, 90)); // +
	positive_points_3d.push_back(myPoint(260, 260, 80)); // +
	positive_points_3d.push_back(myPoint(260, 240, 80)); // +

	//negative_points_3d.push_back(myPoint(210, 200, 80));  // -
	//negative_points_3d.push_back(myPoint(200, 265, 85));  // -

	//Initializing 3D dimensional plane with zeros
	Mat* potentialPlane3D = new Mat[amount_of_slices];
	for (int i = 0; i < amount_of_slices; i++)
	{
		potentialPlane3D[i] = Mat::zeros(imageSeries[i].rows, imageSeries[i].cols, CV_32F);
	}

	// Same as in 2D case, but now it is a vector of Mat-s
	// maskImage is an unsigned char matrix initialized with zeros
	// We need it for storing the segmentation results
	Mat* maskImage3D = new Mat[amount_of_slices];
	for (int i = 0; i < amount_of_slices; i++)
	{
		maskImage3D[i] = Mat::zeros(imageSeries[i].rows, imageSeries[i].cols, CV_8UC1);
	}

	// Initializing 3D potential plane according to preset radius and points coordinates
	// First initializing positive influence. The radius depends on the distance betwee current slice and the slice of z-coordinate of the positive point.
	for (int numOfPositivePoint = 0; numOfPositivePoint < positive_points_3d.size(); numOfPositivePoint++)
		for (int sliceNumber = (positive_points_3d[numOfPositivePoint].z - POTENTIAL_POSITIVE_RADIUS); sliceNumber < (positive_points_3d[numOfPositivePoint].z + POTENTIAL_POSITIVE_RADIUS); sliceNumber++)
		{
			if ((sliceNumber < 0) || (sliceNumber > amount_of_slices)) continue;
			int currentSlicePositiveRadius = POTENTIAL_POSITIVE_RADIUS - abs(sliceNumber - positive_points_3d[numOfPositivePoint].z);
			PotentialPlaneGenerating3Dpositive((float*)(potentialPlane3D[sliceNumber].data), positive_points_3d[numOfPositivePoint], currentSlicePositiveRadius, imageSeries[sliceNumber].cols, imageSeries[sliceNumber].rows, amount_of_slices);
		}
	// Now we add the negative influence in the same way as the positive one
	for (int numOfNegativePoint = 0; numOfNegativePoint < negative_points_3d.size(); numOfNegativePoint++)
		for (int sliceNumber = (negative_points_3d[numOfNegativePoint].z - POTENTIAL_NEGATIVE_RADIUS); sliceNumber < (negative_points_3d[numOfNegativePoint].z + POTENTIAL_NEGATIVE_RADIUS); sliceNumber++)
		{
			if ((sliceNumber < 0) || (sliceNumber > amount_of_slices)) continue;
			int currentSliceNegativeRadius = POTENTIAL_NEGATIVE_RADIUS - abs(sliceNumber - negative_points_3d[numOfNegativePoint].z);
			PotentialPlaneGenerating3Dnegative((float*)(potentialPlane3D[sliceNumber].data), negative_points_3d[numOfNegativePoint], currentSliceNegativeRadius, imageSeries[sliceNumber].cols, imageSeries[sliceNumber].rows, amount_of_slices);
		}

	//To prevent stack overflow error, we need get rid of extra Mat data. To do this, we convert an array of Mat-s to an array of arrays.
	unsigned char** pointerToOriginalData = new unsigned char*[amount_of_slices];
	unsigned char** pointerToMaskData = new unsigned char*[amount_of_slices];
	float** pointerToPotentialData = new float*[amount_of_slices];
	for (int i = 0; i < amount_of_slices; i++)
	{
		pointerToOriginalData[i] = (imageSeries[i].data);
		pointerToMaskData[i]	 = (maskImage3D[i].data);
		pointerToPotentialData[i] = (float*)(potentialPlane3D[i].data);
	}
		
	// We need to launch 3D recursive region growing for each positive point
	for (int positivePointNumber = 0; positivePointNumber < positive_points_3d.size(); positivePointNumber++)
	{
								// Passing pointers to original data, mask data and potential data
		RecursiveRegionGrowth3D (pointerToOriginalData, pointerToPotentialData, pointerToMaskData,
								// Passing coordinates of current point 
								 positive_points_3d[positivePointNumber].x, positive_points_3d[positivePointNumber].y, positive_points_3d[positivePointNumber].z, 
								 // Passing 3D image dimensions
								 imageSeries[positive_points_3d[positivePointNumber].z].cols, imageSeries[positive_points_3d[positivePointNumber].z].rows, amount_of_slices,
								 // Setting the dynamic threshold
								 THRESHOLD_DYNAMIC);
	}

	

	// Routines for visualizing and storing result 
	int startSlice = 0;
	int stopSlice  = 100;
	int step = 2;

	//Visualizing potential slices of 3D potentialPlane
	double minVal, maxVal;
	for (int i = startSlice; i < stopSlice; i += step)
	{
		minMaxLoc(potentialPlane3D[i], &minVal, &maxVal); //find minimum and maximum intensities
		Mat drawPotential;
		potentialPlane3D[i].convertTo(drawPotential, CV_8U, 255.0/(maxVal - minVal), -minVal * 255.0/(maxVal - minVal));
		
		sprintf(slicename, "..//images//potentialPlane_%d.png", i);
		//imshow("potential", drawPotential);
		imwrite(slicename, drawPotential);
		//waitKey();
	}
	
	// Visualizing segementation mask
	for (int i = startSlice; i < stopSlice; i += step)
	{
		sprintf(slicename, "..//images//segmentation_%d.png", i);
		//imshow("potential", maskImage3D[i]);
		imwrite(slicename, maskImage3D[i]);
		//waitKey();
	}

	// Visualizing original data with potential contours
	
	for (int i = startSlice; i < stopSlice; i += step)
	{
		cvtColor(imageSeries[i], imageSeries[i], CV_GRAY2BGR);

		for (int positivePointNumber = 0; positivePointNumber < positive_points_3d.size(); positivePointNumber++)
		{
			int currentSlicePositiveRadius = POTENTIAL_POSITIVE_RADIUS - abs(i - positive_points_3d[positivePointNumber].z);
			if (currentSlicePositiveRadius > 0)
				circle(imageSeries[i], Point(positive_points_3d[positivePointNumber].x, positive_points_3d[positivePointNumber].y), currentSlicePositiveRadius, Scalar(0, 255, 0), 2);
			circle(imageSeries[i], Point(positive_points_3d[positivePointNumber].x, positive_points_3d[positivePointNumber].y), 
				   (abs(positive_points_3d[positivePointNumber].z - i)>3 ? 0 : 3 - abs(positive_points_3d[positivePointNumber].z - i)), 
				   Scalar(0, 255, 0), 2);
		}

		for (int negativePointNumber = 0; negativePointNumber < negative_points_3d.size(); negativePointNumber++)
		{
			int currentSliceNegativeRadius = POTENTIAL_NEGATIVE_RADIUS - abs(i - negative_points_3d[negativePointNumber].z);
			if (currentSliceNegativeRadius > 0)
				circle(imageSeries[i], Point(negative_points_3d[negativePointNumber].x, negative_points_3d[negativePointNumber].y), currentSliceNegativeRadius, Scalar(0, 0, 255), 2);
			circle(imageSeries[i], Point(negative_points_3d[negativePointNumber].x, negative_points_3d[negativePointNumber].y), 
				   (abs(negative_points_3d[negativePointNumber].z - i)>3 ? 0 : 3 - abs(negative_points_3d[negativePointNumber].z - i)), 
				   Scalar(0, 0, 255), 2);
		}

		sprintf(slicename, "..//images//segmentation_%d_orig.png", i);
		//imshow("orig", imageSeries[i]);
		imwrite(slicename, imageSeries[i]);
		//waitKey();
	}

// 2D ALGORITHM
#else
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
#endif	
	log_file.close();
	return 0;

}

