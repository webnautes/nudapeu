#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>


using namespace cv;
using namespace std;


int main()
{
	Mat image;
	image = imread("house.jpg", IMREAD_COLOR);
	if (image.empty())
	{
		cout << "Could not open or find the image" << endl;
		return -1;
	}
	
	namedWindow("Original", WINDOW_AUTOSIZE);
	imshow("Original", image);
	
	waitKey(0);
}
