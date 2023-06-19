#include <chrono>
#include <iostream>
#include <memory>
#include <opencv4/opencv2/opencv.hpp>
#include <thread>

#include "realsense_device.hpp"

int main(int argc, char *argv[])
{
    std::unique_ptr<gca::device> rs_cam(new gca::realsense_device());
    rs_cam->device_start();

    const gca::intrinsics &in = rs_cam->get_color_intrinsics();

    std::cout << in.fx << in.fy << in.cx << in.cy << std::endl;

    const gca::extrinsics &ex_c_to_d = rs_cam->get_color_to_depth_extrinsics();
    for (auto r : ex_c_to_d.rotation)
    {
        std::cout << r << std::endl;
    }

    for (auto t : ex_c_to_d.translation)
    {
        std::cout << t << std::endl;
    }

    cv::namedWindow("display", cv::WINDOW_AUTOSIZE);
    while (true)
    {
        rs_cam->receive_data();
        cv::Mat color = rs_cam->get_color_cv_mat();
        cv::Mat depth = rs_cam->get_depth_cv_mat();

        const void *testptr = rs_cam->get_color_raw_data();
        std::cout << testptr << std::endl;

        cv::imshow("display", color);
        if (cv::waitKey(30) == 27) // Wait for 'esc' key press to exit
        {
            break;
        }
    }
    cv::destroyAllWindows();

    return 0;
}
