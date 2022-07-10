#include <iostream>
#include <vector>
#include <string>
#include <ctime>
#include <pcl/io/pcd_io.h>
#include <pcl-1.10/pcl/kdtree/kdtree_flann.h>
#include "Scancontext/Scancontext.h"
#include "omp.h"

using namespace std;

// 05 -- 2761
// 00 -- 4541
// 08 -- 4071

// number of sequence
const int N = 4071;

// kitti sequence
const string seq = "08";

/*0 for kitti "00","05" only same direction loops;
1 for kitti "08" only reverse loops; 
2 for both same and reverse loops*/
const int loop_event = 0;

std::vector<vector<int>> getGTFromPose(const string& pose_path)
{
    std::ifstream pose_ifs(pose_path);
    std::string line;
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);

    int index = 1;
    while(getline(pose_ifs, line)) 
    {
        if(line.empty()) break;
        stringstream ss(line);
        float r1,r2,r3,t1,r4,r5,r6,t2,r7,r8,r9,t3;
        ss >> r1 >> r2 >> r3 >> t1 >> r4 >> r5 >> r6 >> t2 >> r7 >> r8 >> r9 >> t3;
        pcl::PointXYZI p;
        p.x = t1;
        p.y = 0;
        p.z = t3;
        p.intensity = index++;
        cout << p << endl;
        cloud->push_back(p);
    }

    pcl::io::savePCDFileASCII(seq +".pcd", *cloud);
    pcl::KdTreeFLANN<pcl::PointXYZI> kdtree;
    kdtree.setInputCloud(cloud);
    std::vector<vector<int>> res(5000);
    for(int i = 0; i < cloud->points.size(); i++)
    {
        float radius = 4;
        std::vector<int> ixi;
        std::vector<float> ixf;
        pcl::PointXYZI p = cloud->points[i];
        int cur = p.intensity;
        std::vector<int> nrs;
        kdtree.radiusSearch(p,radius,ixi,ixf);
        for(int j = 0; j < ixi.size(); j++)
        {
            if(cloud->points[ixi[j]].intensity == cur) continue;
            nrs.push_back(cloud->points[ixi[j]].intensity);
        }
        sort(nrs.begin(), nrs.end());
        res[cur] = nrs;
    }
    
    std::ofstream gt_ofs("../gt"+ seq + ".txt");

    for(int i =0; i < res.size(); i++)
    {
        gt_ofs << i << " ";
        for(int j = 0; j < res[i].size(); j++)
        {
            gt_ofs << res[i][j] << " ";
        }
        gt_ofs << endl;
    }
    return res;
}

int main(int argc, char *argv[])
{

    std::cout << "number of available processors: " << omp_get_num_procs()<< std::endl;
    std::cout << "number of threads: " << omp_get_max_threads() << std::endl;
    omp_set_num_threads(4);//设置线程数，一般设置的线程数不超过CPU核心数
    //kitti pose xx.txt
    auto gt = getGTFromPose("/home/cap/code/LiDAR-Iris/"+ seq +".txt");
    std::ofstream ofs("../test_res" + seq+".txt");

    SCManager sc_manager;

    #pragma omp paralell for
    for(int i =0; i <=N-1 ;i++)
    {
        std::stringstream ss;
        ss << setw(6) << setfill('0') << i;
        cout << ss.str()+".bin" << std::endl;

        // kitti velodyne bins
        std::string filename = "/media/cap/621418AD141885E7/dataset/kitti/" + seq + "/velodyne/" + ss.str() + ".bin";
        // cout<<"filename: "<<filename<<endl;
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud0(new pcl::PointCloud<pcl::PointXYZI>);
        std::fstream input(filename, std::ios::in | std::ios::binary);
        input.seekg(0, std::ios::beg);
        for (int ii=0; input.good() && !input.eof(); ii++) {
            pcl::PointXYZI point;
            input.read((char *) &point.x, 3*sizeof(float));
            float intensity;
            input.read((char *) &intensity, sizeof(float));
            cloud0->push_back(point);
        }
        std::cout<<"point size: "<<cloud0->size()<<std::endl;
        sc_manager.makeAndSaveScancontextAndKeys(*cloud0);
      
        float mindis = 1000;
        int loop_id = -1;

        auto sc_res = sc_manager.detectLoopClosureID();
        if(i>300)
        {
            
            loop_id = std::get<0>(sc_res);
        }
        if(loop_id != -1)
        {
            ofs << i+1 <<" " << loop_id+1 << " " << std::get<2>(sc_res)  << " " << 1 << std::endl;
            cout << i+1 << " " << loop_id+1 << " " << std::get<2>(sc_res) << " " << 1 << std::endl;
        }
        else
        {
            ofs<< i+1 << " " << loop_id+1 << " " << std::get<2>(sc_res) << " " << 0 << std::endl;
            cout << i+1 << " " << loop_id+1 << " " << std::get<2>(sc_res) << " " << 0 << std::endl; 
        }

        // if(loop_id == -1) continue;
        // if(std::find(gt[i+1].begin(),gt[i+1].end(),loop_id+1)!=gt[i+1].end())
        // {
        //     ofs << i+1 << " " << loop_id+1 << " " << sc_res.second << " " << 1 << std::endl; 
        //     cout << i+1 << " " << loop_id+1 << " " << sc_res.second << " " << 1 << std::endl; 
        // }
        // else 
        // {
        //     ofs << i+1 << " " << loop_id+1 << " " << sc_res.second << " " << 0 << std::endl; 
        //     cout << i+1 << " " << loop_id+1 << " " << sc_res.second << " " << 0 << std::endl; 

        // }


    }

    return 0;
}