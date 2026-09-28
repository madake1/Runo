#include <iostream>
#include <opencv2/opencv.hpp>

int main()
{
    cv::VideoCapture camera(0);

    if (!camera.isOpened())
    {
        std::cerr << "Failed to open camera!" << std::endl;
        return -1;
    }

    cv::Mat frame;

    while (true)
    {
        camera >> frame;

        if (frame.empty())
        {
            std::cerr << "Failed to read frame!" << std::endl;
            break;
        }

        cv::imshow("Runo Camera", frame);

        if (cv::waitKey(1) == 27)
        {
            break;
        }
    }

    return 0;
}