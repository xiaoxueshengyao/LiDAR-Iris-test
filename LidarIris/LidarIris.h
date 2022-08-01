#ifndef _LIDAR_IRIS_H_
#define _LIDAR_IRIS_H_

#include <vector>
#include <opencv2/opencv.hpp>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>


class LidarIris
{
public:
    struct FeatureDesc
    {
        cv::Mat1b img;
        cv::Mat1b T;
        cv::Mat1b M;
    };

    /**
     * nscale:滤波器参数
     * 
     **/
    LidarIris(int nscale,           //分幾個尺度
              int minWaveLength,    //波長，用於計算中心頻率
              float mult,           //波長變換率
              float sigmaOnf,       //濾波器參數，k/ω0 用一个常量代替，以保证滤波器在不同中心频率时的对数频率尺度下形状不变
              int matchNum) 
              : _nscale(nscale),
                _minWaveLength(minWaveLength),
                _mult(mult),
                _sigmaOnf(sigmaOnf),
                _matchNum(matchNum)
{
    }
    LidarIris(const LidarIris &) = delete;
    LidarIris &operator=(const LidarIris &) = delete;

    static cv::Mat1b GetIris(const pcl::PointCloud<pcl::PointXYZ> &cloud);
    float Compare(const FeatureDesc &img1, const FeatureDesc &img2, int *bias = nullptr);

    FeatureDesc GetFeature(const cv::Mat1b &src);
    std::vector<cv::Mat2f> LogGaborFilter(const cv::Mat1f &src, unsigned int nscale, int minWaveLength, double mult, double sigmaOnf);
    void GetHammingDistance(const cv::Mat1b &T1, const cv::Mat1b &M1, const cv::Mat1b &T2, const cv::Mat1b &M2, int scale, float &dis, int &bias);
    
    static inline cv::Mat circRowShift(const cv::Mat &src, int shift_m_rows);
    static inline cv::Mat circColShift(const cv::Mat &src, int shift_n_cols);
    static cv::Mat circShift(const cv::Mat &src, int shift_m_rows, int shift_n_cols);

private:
    void LoGFeatureEncode(const cv::Mat1b &src, unsigned int nscale, int minWaveLength, double mult, double sigmaOnf, cv::Mat1b &T, cv::Mat1b &M);

    int _nscale;
    int _minWaveLength;
    float _mult;
    float _sigmaOnf;
    int _matchNum;
};

#endif
