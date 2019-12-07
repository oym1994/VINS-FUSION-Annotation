/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *
 * Author: Qin Tong (qintonguav@gmail.com)
 *******************************************************/

#include <stdio.h>
#include <queue>
#include <map>
#include <thread>
#include <mutex>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include "estimator/estimator.h"
#include "estimator/parameters.h"
#include "utility/visualization.h"
#include <QWidget>
#include <QApplication>
#include "qt_plot/widget.h"
#if show_state
extern ofstream DebugFile;
unsigned frontend_times =0;
extern ros::Time system_start_time;
bool start_count_time = 0;
bool refresh_flag =0;
double time_image =0;
#endif

#ifdef plot_time
int d=1;
QApplication a(d,NULL);
Widget w;
#endif


Estimator estimator;

queue<sensor_msgs::ImuConstPtr> imu_buf;
queue<sensor_msgs::PointCloudConstPtr> feature_buf;
queue<sensor_msgs::ImageConstPtr> img0_buf;     //queue队列,感觉有点类似vector
queue<sensor_msgs::ImageConstPtr> img1_buf;
std::mutex m_buf;
std::mutex flag_lock;
double start_timestamp =0 ;


//收到图像消息,将其放到缓存的队列buf里
void img0_callback(const sensor_msgs::ImageConstPtr &img_msg)
{
#if show_state
    if (!start_count_time){
        start_time.tic();
        start_count_time = 1;
        start_timestamp = img_msg->header.stamp.toSec();
    }
#ifdef plot_time

    time_image = double(img_msg->header.stamp.toSec() - start_timestamp);
    w.add_data(time_image,1.0);
    refresh_flag =1;
    printf("image time: %f s\n", img_msg->header.stamp.toSec() - start_timestamp);

#endif

#endif

    m_buf.lock();
    img0_buf.push(img_msg);
    m_buf.unlock();
}

void img1_callback(const sensor_msgs::ImageConstPtr &img_msg)
{
    m_buf.lock();
    img1_buf.push(img_msg);
    m_buf.unlock();
}


cv::Mat getImageFromMsg(const sensor_msgs::ImageConstPtr &img_msg)
{
    cv_bridge::CvImageConstPtr ptr;
    if (img_msg->encoding == "8UC1")
    {
        sensor_msgs::Image img;
        img.header = img_msg->header;
        img.height = img_msg->height;
        img.width = img_msg->width;
        img.is_bigendian = img_msg->is_bigendian;
        img.step = img_msg->step;
        img.data = img_msg->data;
        img.encoding = "mono8";
        ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
    }
    else
        ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);

    cv::Mat img = ptr->image.clone();
    return img;
}

// extract images with same timestamp from two topics
void sync_process()
{
#if show_state
   double total_time = 0 ;
#endif
    while(1)
    {
#if show_state
            TicToc start_time;
#endif
        if(STEREO)
        {
            cv::Mat image0, image1;
            std_msgs::Header header;
            double time = 0;
            m_buf.lock();
            if (!img0_buf.empty() && !img1_buf.empty())
            {
                double time0 = img0_buf.front()->header.stamp.toSec();  //front是buf里的第一个元素
                double time1 = img1_buf.front()->header.stamp.toSec();
                // 0.003s sync tolerance 如果时间间隔超过3ms,则不是同一帧的双目图像
                if(time0 < time1 - 0.003)
                {
                    img0_buf.pop();   //pop移除buf里的第一个
                    printf("throw img0\n");
                }
                else if(time0 > time1 + 0.003)
                {
                    img1_buf.pop();
                    printf("throw img1\n");
                }
                else
                {
                    time = img0_buf.front()->header.stamp.toSec();  //双目图像帧的时间设为cam0的时间
                    header = img0_buf.front()->header;
                    image0 = getImageFromMsg(img0_buf.front());
                    img0_buf.pop();
                    image1 = getImageFromMsg(img1_buf.front());
                    img1_buf.pop();
                    //printf("find img0 and img1\n");
                }
            }
            m_buf.unlock();
            if(!image0.empty())
                estimator.inputImage(time, image0, image1);  //
        }
        else
        {
            cv::Mat image;
            std_msgs::Header header;
            double time = 0;
            m_buf.lock();
            if(!img0_buf.empty())
            {
                time = img0_buf.front()->header.stamp.toSec();
                header = img0_buf.front()->header;
                image = getImageFromMsg(img0_buf.front());
                img0_buf.pop();
            }
            m_buf.unlock();
            if(!image.empty()){


            estimator.inputImage(time, image);  //单目前端

            }
        }

#if show_state
            if(start_time.toc()>1){
            frontend_times++;
             DebugFile<<"第"<<frontend_times<<"次前端消耗时间:"<< start_time.toc()<<"ms"<<endl;
             total_time +=start_time.toc();
             DebugFile<<"前"<<frontend_times<<"次前端平均耗时:"<<total_time/frontend_times<<"ms"<<endl;

            }
#endif

        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);  //睡眠2ms?
    }


}




void imu_callback(const sensor_msgs::ImuConstPtr &imu_msg)
{
    double t = imu_msg->header.stamp.toSec();
    double dx = imu_msg->linear_acceleration.x;
    double dy = imu_msg->linear_acceleration.y;
    double dz = imu_msg->linear_acceleration.z;
    double rx = imu_msg->angular_velocity.x;
    double ry = imu_msg->angular_velocity.y;
    double rz = imu_msg->angular_velocity.z;
    Vector3d acc(dx, dy, dz);
    Vector3d gyr(rx, ry, rz);
    estimator.inputIMU(t, acc, gyr);
    return;
}

void gyro_callback(const sensor_msgs::ImuConstPtr &imu_msg)
{

    double t = imu_msg->header.stamp.toSec();
    double rx = imu_msg->angular_velocity.x;
    double ry = imu_msg->angular_velocity.y;
    double rz = imu_msg->angular_velocity.z;
    double dx = 1000;
    double dy = 1000;
    double dz = 1000;
    Vector3d acc(dx, dy, dz);
    Vector3d gyr(rx, ry, rz);
#ifdef plot_time

    time_image = double(t - start_timestamp);
    w.add_data(time_image,-1.0);
#endif
//    printf("收到gyro消息");
    estimator.inputIMU(t, acc, gyr);
    return;
}


void acc_callback(const sensor_msgs::ImuConstPtr &imu_msg)
{
    double t = imu_msg->header.stamp.toSec();
    double dx = imu_msg->linear_acceleration.x;
    double dy = imu_msg->linear_acceleration.y;
    double dz = imu_msg->linear_acceleration.z;
    double rx = 1000;
    double ry = 1000;
    double rz = 1000;

#ifdef plot_time
    time_image = double(t - start_timestamp);
    w.add_data(time_image,-0.5);
#endif
    Vector3d acc(dx, dy, dz);
    Vector3d gyr(rx, ry, rz);
//    printf("收到acc消息");
    estimator.inputIMU(t, acc, gyr);
    return;
}



void feature_callback(const sensor_msgs::PointCloudConstPtr &feature_msg)
{
    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> featureFrame;
    for (unsigned int i = 0; i < feature_msg->points.size(); i++)
    {
        int feature_id = feature_msg->channels[0].values[i];
        int camera_id = feature_msg->channels[1].values[i];
        double x = feature_msg->points[i].x;
        double y = feature_msg->points[i].y;
        double z = feature_msg->points[i].z;
        double p_u = feature_msg->channels[2].values[i];
        double p_v = feature_msg->channels[3].values[i];
        double velocity_x = feature_msg->channels[4].values[i];
        double velocity_y = feature_msg->channels[5].values[i];
        if(feature_msg->channels.size() > 5)
        {
            double gx = feature_msg->channels[6].values[i];
            double gy = feature_msg->channels[7].values[i];
            double gz = feature_msg->channels[8].values[i];
            pts_gt[feature_id] = Eigen::Vector3d(gx, gy, gz);
            //printf("receive pts gt %d %f %f %f\n", feature_id, gx, gy, gz);
        }
        ROS_ASSERT(z == 1);
        Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
        xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
        featureFrame[feature_id].emplace_back(camera_id,  xyz_uv_velocity);
    }
    double t = feature_msg->header.stamp.toSec();
    estimator.inputFeature(t, featureFrame);
    return;
}

void restart_callback(const std_msgs::BoolConstPtr &restart_msg)
{
    if (restart_msg->data == true)
    {
        ROS_WARN("restart the estimator!");
        estimator.clearState();
        estimator.setParameter();
    }
    return;
}

void imu_switch_callback(const std_msgs::BoolConstPtr &switch_msg)
{
    if (switch_msg->data == true)
    {
        //ROS_WARN("use IMU!");
        estimator.changeSensorType(1, STEREO);
    }
    else
    {
        //ROS_WARN("disable IMU!");
        estimator.changeSensorType(0, STEREO);
    }
    return;
}

void cam_switch_callback(const std_msgs::BoolConstPtr &switch_msg)
{
    if (switch_msg->data == true)
    {
        //ROS_WARN("use stereo!");
        estimator.changeSensorType(USE_IMU, 1);
    }
    else
    {
        //ROS_WARN("use mono camera (left)!");
        estimator.changeSensorType(USE_IMU, 0);
    }
    return;
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "vins_estimator");  //ros初始化,从命令行传入参数,vins_estimator是节点名
    ros::NodeHandle n("~");                   //声明一个该ros项目的命名空间,但"~"代表啥?
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info); //设置ros控制台logger level

#if show_state
    DebugFile.open("/home/oym/VINS-FUSION-Annotation/src/debugfile.txt", ios::app);
    system_start_time = ros::Time::now();

#endif

    if(argc != 2)
    {
        printf("please intput: rosrun vins vins_node [config file] \n"
               "for example: rosrun vins vins_node "
               "~/catkin_ws/src/VINS-Fusion/config/euroc/euroc_stereo_imu_config.yaml \n");
        return 1;
    }

    string config_file = argv[1];
    printf("config_file: %s\n", argv[1]);

    readParameters(config_file);   //读取参数配置文件
    estimator.setParameter();      //设置系统参数

#ifdef EIGEN_DONT_PARALLELIZE
    ROS_DEBUG("EIGEN_DONT_PARALLELIZE");
#endif

    ROS_WARN("waiting for image and imu...");

    registerPub(n);  //注册发布话题
    //注册订阅消息,参数分别为:订阅消息名称,缓存,回调函数,ros::TransportHints().tcpNoDelay()与tcp连接时有用
    ros::Subscriber sub_imu = n.subscribe(IMU_TOPIC, 2000, imu_callback, ros::TransportHints().tcpNoDelay());
    ros::Subscriber sub_feature = n.subscribe("/feature_tracker/feature", 2000, feature_callback);
    ros::Subscriber sub_img0 = n.subscribe(IMAGE0_TOPIC, 100, img0_callback);
    ros::Subscriber sub_img1 = n.subscribe(IMAGE1_TOPIC, 100, img1_callback);
    ros::Subscriber sub_restart = n.subscribe("/vins_restart", 100, restart_callback);
    ros::Subscriber sub_imu_switch = n.subscribe("/vins_imu_switch", 100, imu_switch_callback);
    ros::Subscriber sub_cam_switch = n.subscribe("/vins_cam_switch", 100, cam_switch_callback);
    ros::Subscriber sub_acc = n.subscribe(ACC_TOPIC, 2000, acc_callback, ros::TransportHints().tcpNoDelay());
    ros::Subscriber sub_gyro = n.subscribe(GYRO_TOPIC, 2000, gyro_callback, ros::TransportHints().tcpNoDelay());


    std::thread sync_thread{sync_process}; //开启线程
//    std::thread refresh_thread{refreshplot}; //开启线程

    ros::spin();
#if show_state
    DebugFile.close();

#endif
    return 0;
}