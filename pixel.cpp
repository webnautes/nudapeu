#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;



int main()
{
	Mat img_color = imread("test.jpg", IMREAD_COLOR);


	int height = img_color.rows;
	int width = img_color.cols;


	int x = 50;
	int y = 50;


	int b = img_color.at<Vec3b>(y, x)[0];
	int g = img_color.at<Vec3b>(y, x)[1];
	int r = img_color.at<Vec3b>(y, x)[2];

	cout << b << " " << g << " " << r <<endl;


	uchar *data = img_color.data;
	b = data[y * width * 3 + x * 3];
	g = data[y * width * 3 + x * 3 + 1];
	r = data[y * width * 3 + x * 3 + 2];

	cout << b << " " << g << " " << r <<endl;


	return 0;
}
