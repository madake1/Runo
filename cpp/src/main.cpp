// Desktop OpenCV camera preview entry point for the standalone C++ project.
#include <iostream>
#include <opencv2/opencv.hpp>

// Open the default camera, show frames until Esc, and report camera failures.
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
