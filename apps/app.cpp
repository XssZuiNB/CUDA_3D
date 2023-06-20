#include <chrono>
#include <iostream>
#include <memory>
#include <opencv4/opencv2/opencv.hpp>
#include <thread>

#include "realsense_device.hpp"

void transform_point_to_point(float to_point[3], const gca::extrinsics *extrin,
                              const float from_point[3])
{
    to_point[0] = extrin->rotation[0] * from_point[0] + extrin->rotation[1] * from_point[1] +
                  extrin->rotation[2] * from_point[2] + extrin->translation[0];
    to_point[1] = extrin->rotation[3] * from_point[0] + extrin->rotation[4] * from_point[1] +
                  extrin->rotation[5] * from_point[2] + extrin->translation[1];
    to_point[2] = extrin->rotation[6] * from_point[0] + extrin->rotation[7] * from_point[1] +
                  extrin->rotation[8] * from_point[2] + extrin->translation[2];
}

void uv_to_xyz(const float uv[2], float depth, float xyz[3], float depth_scale,
               const gca::intrinsics &depth_in)
{
    auto z = depth / depth_scale;
    xyz[2] = z;
    xyz[0] = (uv[0] - depth_in.cx) * z / depth_in.fx;
    xyz[1] = (uv[1] - depth_in.cy) * z / depth_in.fy;
}

void xyz_to_uv(float uv[2], const float xyz[3], const gca::intrinsics &color_in)
{

    uv[0] = (xyz[0] * color_in.fx / xyz[2]) + color_in.cx;
    uv[1] = (xyz[1] * color_in.fy / xyz[2]) + color_in.cy;
}

int main(int argc, char *argv[])
{
    std::unique_ptr<gca::device> rs_cam(new gca::realsense_device());
    rs_cam->device_start();

    const gca::intrinsics &c_in = rs_cam->get_color_intrinsics();
    const gca::intrinsics d_in = rs_cam->get_depth_intrinsics();
    const gca::extrinsics &ex_c_to_d = rs_cam->get_color_to_depth_extrinsics();
    const gca::extrinsics &ex_d_to_c = rs_cam->get_depth_to_color_extrinsics();

    auto depth_scale = rs_cam->get_depth_scale();

    while (true)
    {
        rs_cam->receive_data();
        cv::Mat color = rs_cam->get_color_cv_mat();
        cv::Mat depth = rs_cam->get_depth_cv_mat();

        //平面点
        float pd_uv[2], pc_uv[2];
        //空间点定义
        float Pdc3[3], Pcc3[3];

        cv::Mat result = cv::Mat::zeros(cv::Size(640, 480), CV_8UC3);
        //对深度图像遍历
        for (int row = 0; row < depth.rows; row++)
        {
            for (int col = 0; col < depth.cols; col++)
            {
                //将当前的(x,y)放入数组pd_uv，表示当前深度图的点
                pd_uv[0] = col;
                pd_uv[1] = row;
                //取当前点对应的深度值
                uint16_t depth_value = depth.at<uint16_t>(row, col);
                //将深度图的像素点根据内参转换到深度摄像头坐标系下的三维点
                uv_to_xyz(pd_uv, depth_value, Pdc3, depth_scale, d_in);
                //将深度摄像头坐标系的三维点转化到彩色摄像头坐标系下
                transform_point_to_point(Pcc3, &ex_d_to_c, Pdc3);
                //将彩色摄像头坐标系下的深度三维点映射到二维平面上
                xyz_to_uv(pc_uv, Pcc3, c_in);

                //取得映射后的（u,v)
                auto x = (int)round(pc_uv[0]);
                auto y = (int)round(pc_uv[1]);
                //            if(x<0||x>color.cols)
                //                continue;
                //            if(y<0||y>color.rows)
                //                continue;
                //最值限定
                if (x < 0 || x > depth.cols - 1 || y < 0 || y > depth.rows - 1)
                {
                    continue;
                }
                //将成功映射的点用彩色图对应点的RGB数据覆盖
                for (int k = 0; k < 3; k++)
                {
                    //这里设置了只显示1米距离内的东西

                    result.at<cv::Vec3b>(y, x)[k] = color.at<cv::Vec3b>(y, x)[k];
                }
            }
        }

        cv::imshow("color", color);
        cv::imshow("result", result);

        if (cv::waitKey(30) == 27) // Wait for 'esc' key press to exit
        {
            break;
        }
    }
    cv::destroyAllWindows();

    return 0;
}
