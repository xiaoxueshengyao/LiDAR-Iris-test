#include "LidarIris.h"
#include "fftm/fftm.hpp"

//获得点云的俯视图
cv::Mat1b LidarIris::GetIris(const pcl::PointCloud<pcl::PointXYZ> &cloud)
{
    cv::Mat1b IrisMap = cv::Mat1b::zeros(80, 360);

    for (pcl::PointXYZ p : cloud.points)
    {
        float dis = sqrt(p.data[0] * p.data[0] + p.data[1] * p.data[1]);//x^2 +ｙ^2
        float yaw = (atan2(p.data[1], p.data[0]) * 180.0f / M_PI) + 180;//偏航角 0-360
        int Q_dis = std::min(std::max((int)floor(dis), 0), 79);//俯视图按距离划分矩阵行id，0-80
        int Q_arc = std::min(std::max((int)ceil(p.z + 5), 0), 7);//根据高度分成8份，这里的+5使得所有的点云z都大于0，另一个版本是通过俯仰角
        int Q_yaw = std::min(std::max((int)floor(yaw + 0.5), 0), 359);//根据偏航角划分列id
        IrisMap.at<uint8_t>(Q_dis, Q_yaw) |= (1 << Q_arc);//8位数据来表示一个bin，在该高度有点时就会置1，没有点时默认为0
    }

    return IrisMap;
}


float LidarIris::Compare(const LidarIris::FeatureDesc &img1, const LidarIris::FeatureDesc &img2, int *bias)
{
    if(_matchNum==2) //正向反向都有
    {
        auto firstRect = FFTMatch(img2.img, img1.img);
        int firstShift = firstRect.center.x - img1.img.cols / 2;
        float dis1;
        int bias1;
        GetHammingDistance(img1.T, img1.M, img2.T, img2.M, firstShift, dis1, bias1);
        
        auto T2x = circShift(img2.T, 0, 180);
        auto M2x = circShift(img2.M, 0, 180);
        auto img2x = circShift(img2.img, 0, 180);
        
        auto secondRect = FFTMatch(img2x, img1.img);
        int secondShift = secondRect.center.x - img1.img.cols / 2;
        float dis2 = 0;
        int bias2 = 0;
        GetHammingDistance(img1.T, img1.M, T2x, M2x, secondShift, dis2, bias2);
        
        if (dis1 < dis2)
        {
            if (bias)
                *bias = bias1;
            return dis1;
        }
        else
        {
            if (bias)
                *bias = (bias2 + 180) % 360;
            return dis2;
        }
    }
    if(_matchNum==1)//只有反向
    {
        auto T2x = circShift(img2.T, 0, 180);
        auto M2x = circShift(img2.M, 0, 180);
        auto img2x = circShift(img2.img, 0, 180);

        auto secondRect = FFTMatch(img2x, img1.img);
        int secondShift = secondRect.center.x - img1.img.cols / 2;
        float dis2 = 0;
        int bias2 = 0;
        GetHammingDistance(img1.T, img1.M, T2x, M2x, secondShift, dis2, bias2);
        if (bias)
            *bias = (bias2 + 180) % 360;
        return dis2;
    }
    if(_matchNum==0)
    {
        auto firstRect = FFTMatch(img2.img, img1.img);//得到偏移量
        int firstShift = firstRect.center.x - img1.img.cols / 2;
        float dis1;
        int bias1;
        if(firstShift >= 356)
        {
            firstShift = 356;
        }
        else if(firstShift <= -356)
        {
            firstShift = -356;
        }
        
        GetHammingDistance(img1.T, img1.M, img2.T, img2.M, firstShift, dis1, bias1);
        img1.M.convertTo(img1.M,CV_8UC1,255);
        img2.M.convertTo(img2.M,CV_8UC1,255);
        cv::imshow("img1M",img1.M);
        cv::imshow("img2M",img2.M);
        if (bias)
            *bias = bias1;
        std::cout<<"dis1:"<<dis1<<" bias1:"<<bias1<<std::endl;
        return dis1;

    }
}

std::vector<cv::Mat2f> LidarIris::LogGaborFilter(const cv::Mat1f &src, unsigned int nscale, int minWaveLength, double mult, double sigmaOnf)
{
    int rows = src.rows;
    int cols = src.cols;
    cv::Mat2f filtersum = cv::Mat2f::zeros(1, cols);
    std::vector<cv::Mat2f> EO(nscale);//4
    std::cout<<"EO size: "<<EO.size()<<std::endl;
    int ndata = cols;
    if (ndata % 2 == 1)//确保位偶数
        ndata--;
    cv::Mat1f logGabor = cv::Mat1f::zeros(1, ndata);
    cv::Mat2f result = cv::Mat2f::zeros(rows, ndata);
    cv::Mat1f radius = cv::Mat1f::zeros(1, ndata / 2 + 1);
    std::cout<<" radius size :　"<<radius.size()<<"\t"<<logGabor.size()<<std::endl;
    radius.at<float>(0, 0) = 1;
    for (int i = 1; i < ndata / 2 + 1; i++)
    {
        radius.at<float>(0, i) = i / (float)ndata;// i/cols
    }
    double wavelength = minWaveLength;//18
    std::cout<<"wavelength: "<<wavelength<<std::endl;
    for (int s = 0; s < nscale; s++)
    {
        //对于log-gabor的计算参考https://blog.csdn.net/StayAlive1/article/details/125412499
        double fo = 1.0 / wavelength;//频率 = 1 / 波长，滤波器中心频率
        double rfo = fo / 0.5;  //没用到
        //构建滤波器
        cv::Mat1f temp; //(radius.size());
        cv::log(radius / fo, temp);//元素绝对值的自然对数
        cv::pow(temp, 2, temp);
        cv::exp((-temp) / (2 * log(sigmaOnf) * log(sigmaOnf)), temp);
        temp.copyTo(logGabor.colRange(0, ndata / 2 + 1));
        //
        logGabor.at<float>(0, 0) = 0;
        cv::Mat2f filter;
        cv::Mat1f filterArr[2] = {logGabor, cv::Mat1f::zeros(logGabor.size())};
        cv::merge(filterArr, 2, filter);//通道合并
        filtersum = filtersum + filter;//这感觉不用也行啊
        //按行进行傅里叶变换
        for (int r = 0; r < rows; r++)
        {
            cv::Mat2f src2f;
            cv::Mat1f srcArr[2] = {src.row(r).clone(), cv::Mat1f::zeros(1, src.cols)};
            cv::merge(srcArr, 2, src2f);//合并成1个图
            cv::dft(src2f, src2f);//默认进行正向离散傅里叶变换
            cv::mulSpectrums(src2f, filter, src2f, 0);//对傅里叶频谱进行乘法
            cv::idft(src2f, src2f);//反向傅里叶
            src2f.copyTo(result.row(r));
        }
        EO[s] = result.clone();
        wavelength *= mult;//每次变波长
    }
    filtersum = circShift(filtersum, 0, cols / 2);//这玩意也没用到，在做行和列的shift操作
    return EO;
}

//log滤波器对图像进行编码
void LidarIris::LoGFeatureEncode(const cv::Mat1b &src, unsigned int nscale, int minWaveLength, double mult, double sigmaOnf, cv::Mat1b &T, cv::Mat1b &M)
{
    cv::Mat1f srcFloat;
    src.convertTo(srcFloat, CV_32FC1);//转为float
    auto list = LogGaborFilter(srcFloat, nscale, minWaveLength, mult, sigmaOnf);
    std::vector<cv::Mat1b> Tlist(nscale * 2), Mlist(nscale * 2);
    for (int i = 0; i < list.size(); i++)
    {
        cv::Mat1f arr[2];
        cv::split(list[i], arr);//滤波后的结果分成2通道
        Tlist[i] = arr[0] > 0;//二值化
        Tlist[i + nscale] = arr[1] > 0;
        cv::Mat1f m;
        cv::magnitude(arr[0], arr[1], m);//实部和虚部的幅值
        Mlist[i] = m < 0.0001;
        Mlist[i + nscale] = m < 0.0001;
    }
    cv::vconcat(Tlist, T);
    cv::vconcat(Mlist, M);
}

//获取iris描述子
LidarIris::FeatureDesc LidarIris::GetFeature(const cv::Mat1b &src)
{
    FeatureDesc desc;
    desc.img = src;
    LoGFeatureEncode(src, _nscale, _minWaveLength, _mult, _sigmaOnf, desc.T, desc.M);
    //这里的描述子其实就是傅里叶+滤波后实部虚部二值化后的特征图
    return desc;
}

//计算汉明距离
void LidarIris::GetHammingDistance(const cv::Mat1b &T1, const cv::Mat1b &M1, const cv::Mat1b &T2, const cv::Mat1b &M2, int scale, float &dis, int &bias)
{
    dis = NAN;
    bias = -1;
    std::cout<<"shift:"<<scale<<std::endl;
    for (int shift = scale - 2; shift <= scale + 2; shift++)
    {
        cv::Mat1b T1s = circShift(T1, 0, shift);//只在cols方向有位移
        cv::Mat1b M1s = circShift(M1, 0, shift);
        cv::Mat1b mask = M1s | M2;
        // cv::imshow("T1",T1);
        // cv::imshow("T1s",T1s);
        // cv::imshow("M1s",M1s);
        // cv::imshow("mask",mask);
        int MaskBitsNum = cv::sum(mask / 255)[0];
        std::cout<<"maskbitnum: "<<MaskBitsNum<<std::endl;
        int totalBits = T1s.rows * T1s.cols - MaskBitsNum;
        std::cout<<"totalBits: "<<totalBits<<std::endl;
        cv::Mat1b C = T1s ^ T2;//异或运算
        C = C & ~mask;
        int bitsDiff = cv::sum(C / 255)[0];
        std::cout<<"bitDiff:"<<bitsDiff<<std::endl;
        if (totalBits == 0)
        {
            dis = NAN;
        }
        else
        {
            float currentDis = bitsDiff / (float)totalBits;
            if (currentDis < dis || isnan(dis))
            {
                dis = currentDis;
                bias = shift;
            }
        }
    }
    return;
}

inline cv::Mat LidarIris::circRowShift(const cv::Mat &src, int shift_m_rows)
{
    if (shift_m_rows == 0)
        return src.clone();
    shift_m_rows %= src.rows;
    int m = shift_m_rows > 0 ? shift_m_rows : src.rows + shift_m_rows;
    cv::Mat dst(src.size(), src.type());
    src(cv::Range(src.rows - m, src.rows), cv::Range::all()).copyTo(dst(cv::Range(0, m), cv::Range::all()));
    src(cv::Range(0, src.rows - m), cv::Range::all()).copyTo(dst(cv::Range(m, src.rows), cv::Range::all()));
    return dst;
}

inline cv::Mat LidarIris::circColShift(const cv::Mat &src, int shift_n_cols)
{
    if (shift_n_cols == 0)
        return src.clone();
    shift_n_cols %= src.cols;
    int n = shift_n_cols > 0 ? shift_n_cols : src.cols + shift_n_cols;
    cv::Mat dst(src.size(), src.type());
    src(cv::Range::all(), cv::Range(src.cols - n, src.cols)).copyTo(dst(cv::Range::all(), cv::Range(0, n)));
    src(cv::Range::all(), cv::Range(0, src.cols - n)).copyTo(dst(cv::Range::all(), cv::Range(n, src.cols)));
    return dst;
}

cv::Mat LidarIris::circShift(const cv::Mat &src, int shift_m_rows, int shift_n_cols)
{
    return circColShift(circRowShift(src, shift_m_rows), shift_n_cols);
}