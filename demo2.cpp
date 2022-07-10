#include <iostream>
#include <vector>
#include <string>
#include <ctime>
#include <pcl/io/pcd_io.h>
#include "LidarIris.h"
#include "Scancontext/Scancontext.h"

using namespace std;

void OneCoupleCompare(string cloudFileName1, string cloudFileName2)
{
    LidarIris iris(4, 18, 1.6, 0.75, 50);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud0(new pcl::PointCloud<pcl::PointXYZ>), cloud1(new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPCDFile(cloudFileName1, *cloud0) == -1)
    {
        abort();
    }
    if (pcl::io::loadPCDFile(cloudFileName2, *cloud1) == -1)
    {
        abort();
    }
    cout<<"cloud0 size : "<<cloud0->size()<<endl;
    cout<<"cloud1 size : "<<cloud1->size()<<endl;
    clock_t startTime = clock();

    cv::Mat1b li1 = LidarIris::GetIris(*cloud0);
    clock_t get_iris = clock();
    cv::Mat1b li2 = LidarIris::GetIris(*cloud1);
    clock_t get_iris2 = clock();

    LidarIris::FeatureDesc fd1 = iris.GetFeature(li1);
    clock_t get_feature1 = clock();
    LidarIris::FeatureDesc fd2 = iris.GetFeature(li2);
    clock_t get_feature2 = clock();

    int bias;
    auto dis = iris.Compare(fd1, fd2, &bias);

    clock_t endTime = clock();

    cout << "get iris 1 use " << (get_iris - startTime)/(double)CLOCKS_PER_SEC<< " s"<<endl;
    cout << "get iris 2 use " << (get_iris2 - get_iris)/(double)CLOCKS_PER_SEC << " s"<<endl;
    cout << "get feature 1 use "<<(get_feature1 - get_iris2)/(double)CLOCKS_PER_SEC << " s"<<endl;
    cout << "get feature 2 use "<<(get_feature2 - get_feature1)/(double)CLOCKS_PER_SEC << " s"<<endl;
    cout << "compare use "<<(endTime - get_feature2)/(double)CLOCKS_PER_SEC << " s"<<endl;

    cout << "try compare:" << endl
         << cloudFileName1 << endl
         << cloudFileName2 << endl;
    cout << "dis = " << dis << ", bias = " << bias << endl;
    cout << "times = " << (endTime - startTime) / (double)CLOCKS_PER_SEC << "s."<< endl;

    cv::Mat1b img_iris, img_T;
    cv::vconcat(fd1.img, fd2.img, img_iris);
    cv::imshow("LiDAR Iris before transformation", img_iris);
    // cv::imwrite("../img/before.bmp", img_iris);
    
    cv::Mat temp = LidarIris::circShift(fd1.img, 0, bias);
    cv::vconcat(temp, fd2.img, img_iris);
    cv::imshow("LiDAR Iris after transformation", img_iris);
    // cv::imwrite("../img/after.bmp", img_iris);

    cv::hconcat(fd1.T, fd2.T, img_T);
    cv::imshow("LiDAR Iris Template", img_T);
    // cv::imwrite("../img/temp.bmp", img_T);

    cv::waitKey(0);
}


void compareScanContext(string cloudFileName1, string cloudFileName2)
{
    SCManager sc_manager;
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud0(new pcl::PointCloud<pcl::PointXYZ>), cloud1(new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPCDFile(cloudFileName1, *cloud0) == -1)
    {
        abort();
    }
    if (pcl::io::loadPCDFile(cloudFileName2, *cloud1) == -1)
    {
        abort();
    }
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_0(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_1(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::copyPointCloud(*cloud0,*cloud_0);
    pcl::copyPointCloud(*cloud1,*cloud_1);

    clock_t start_time = clock();
    Eigen::MatrixXd sc1 = sc_manager.makeScancontext(*cloud_0);
    clock_t get_feature1 = clock();
    Eigen::MatrixXd sc2 = sc_manager.makeScancontext(*cloud_1);
    clock_t get_feature2 = clock();

    std::pair<double, int> sc_dist_result = sc_manager.distanceBtnScanContext(sc1,sc2);
    clock_t compare_time = clock();

    cout<<"get sc feature1 use : "<<(get_feature1 - start_time)/(double)CLOCKS_PER_SEC<<endl;
    cout<<"get sc feature2 use : "<<(get_feature2 - get_feature1)/(double)CLOCKS_PER_SEC<<endl;
    cout<<"compare use :"<<(compare_time - get_feature2)/(double)CLOCKS_PER_SEC<<endl;
    cout<<"bias : "<<sc_dist_result.second<<endl;

    cv::Mat sc_img,sc1_mat,sc2_mat;
    cv::eigen2cv(sc1,sc1_mat);
    cv::eigen2cv(sc2,sc2_mat);
    // cv::bitwise_not(sc1_mat,sc1_mat);
    // cv::bitwise_not(sc2_mat,sc2_mat);
    cv::vconcat(sc1_mat,sc2_mat,sc_img);

    cv::Mat dis(400,600,sc_img.type());
    cv::resize(sc_img,dis,dis.size());
    
    cv::imshow("sc befor compare", dis);

    cv::waitKey(0);


}

int main(int argc, char *argv[])
{

    if(argc != 3)
    {
        cerr << "usage: ./demo cloud1.pcd cloud2.pcd" << std::endl;
        return -1;
    }

    OneCoupleCompare(argv[1], argv[2]);
    compareScanContext(argv[1], argv[2]);

    return 0;
}   