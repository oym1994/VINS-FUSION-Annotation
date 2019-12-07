cmake_minimum_required(VERSION 2.8.3)
project(vins)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
#-DEIGEN_USE_MKL_ALL")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)



find_package(catkin REQUIRED COMPONENTS
    roscpp
    std_msgs
    geometry_msgs
    nav_msgs
    tf
    cv_bridge
    camera_models
    image_transport)

find_package(OpenCV REQUIRED)

find_package(Ceres REQUIRED)

set(CMAKE_PREFIX_PATH /home/oym/Qt5.7.0/5.7/gcc_64/lib/cmake/Qt5)


find_package(Qt5 COMPONENTS Core Gui Qml Quick REQUIRED)

qt5_wrap_cpp( MOC src/qt_plot/qcustomplot.h)
qt5_wrap_cpp( MOC src/qt_plot/widget.h)


qt5_generate_moc(src/qt_plot/qcustomplot.cpp src/qt_plot/qcustomplot.moc)
qt5_generate_moc(src/qt_plot/widget.cpp src/qt_plot/widget.moc)


find_package(Qt5PrintSupport REQUIRED)
if(Qt5PrintSupport_FOUND)
    message("Found Qt5PrintSupport")
else()
    message("Not Found Qt5PrintSupport")
endif()

include_directories(${catkin_INCLUDE_DIRS} ${CERES_INCLUDE_DIRS})

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(Eigen3)
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${Qt5PrintSupport_INCLUDE_DIRS}
  ${Qt5_INCLUDE_DIRS}
  ${CMAKE_CURRENT_BINARY_DIR}/..
  src/qt_plot
)


catkin_package()

add_library(vins_lib
    src/estimator/parameters.cpp
    src/estimator/estimator.cpp
    src/estimator/feature_manager.cpp
    src/factor/pose_local_parameterization.cpp
    src/factor/projectionTwoFrameOneCamFactor.cpp
    src/factor/projectionTwoFrameTwoCamFactor.cpp
    src/factor/projectionOneFrameTwoCamFactor.cpp
    src/factor/marginalization_factor.cpp
    src/utility/utility.cpp
    src/utility/visualization.cpp
    src/utility/CameraPoseVisualization.cpp
    src/initial/solve_5pts.cpp
    src/initial/initial_aligment.cpp
    src/initial/initial_sfm.cpp
    src/initial/initial_ex_rotation.cpp
    src/featureTracker/feature_tracker.cpp
    )
target_link_libraries(vins_lib ${catkin_LIBRARIES} ${OpenCV_LIBS} ${CERES_LIBRARIES} )

add_library(qtplot_lib
        src/qt_plot/qcustomplot.cpp
        src/qt_plot/widget.cpp
       )
target_link_libraries(qtplot_lib  ${catkin_LIBRARIES} ${Qt5PrintSupport_LIBRARIES} ${Qt5_LIBRARIES})



add_executable(vins_node src/rosNodeTest.cpp )
target_link_libraries(vins_node vins_lib qtplot_lib)

add_executable(kitti_odom_test src/KITTIOdomTest.cpp)
target_link_libraries(kitti_odom_test vins_lib) 

add_executable(kitti_gps_test src/KITTIGPSTest.cpp)
target_link_libraries(kitti_gps_test vins_lib) 

