#include <opencv2/opencv.hpp>
#include <iostream>


using namespace cv;
using namespace std;


typedef struct _PointInfo
{
    int x;
    int y;
    int sad;
}PointInfo;


int main()
{

    Mat img_color = imread("image.jpg", IMREAD_COLOR);
    Mat img_original;
    cvtColor(img_color, img_original, COLOR_BGR2GRAY);

    Mat img_template = imread("template.jpg", IMREAD_GRAYSCALE);


    int original_h = img_original.rows;
    int original_w = img_original.cols;
    int template_h = img_template.rows;
    int template_w = img_template.cols;


    int best_position_y = 0;
    int best_position_x = 0;
    int best_position_sad = 100000;

    vector<PointInfo> point;


    // 원본 이미지 스캔
    for ( int original_y = 0; original_y < original_h - template_h; original_y++)
    {
        uchar *data_original = img_original.data;

        for (int original_x = 0; original_x < original_w - template_w; original_x++)
        {

            int SAD = 0;

            //템플릿 이미지 스캔
            for (int template_y = 0; template_y < template_h; template_y++)
            {
                uchar *data_template = img_template.data;

                for (int template_x = 0; template_x < template_w; template_x++)
                {

                    int original_pixel = data_original[(original_y+template_y)*original_w + (original_x+template_x)];
                    int template_pixel = data_template[template_y*template_w + template_x];
            
                    SAD += abs( original_pixel - template_pixel);
                }
            }


            // 최소 SAD 지점 찾기  
            if (best_position_sad > SAD)
            {
                best_position_sad = SAD;             
                best_position_y = original_y;
                best_position_x = original_x;
            }

            PointInfo pointInfo = {original_x, original_y, SAD};
            point.push_back(pointInfo);

        }
    }



    for (int i = 0; i < point.size(); i++)
    {

        if (abs(best_position_sad - point[i].sad) < 100){

            cout << abs(best_position_sad - point[i].sad) << endl;
            rectangle(img_color, Point(point[i].x, point[i].y), Point(point[i].x+template_w, point[i].y+template_h), Scalar(255, 0, 0), 3 );;
            
        }
    }

    imshow("result", img_color);
    waitKey(0);
}
