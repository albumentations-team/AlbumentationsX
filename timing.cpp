#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <vector>

int main()
{
    cv::setNumThreads(1);
    cv::RNG rng(137);
    const cv::Size fullHD(1920,1080), vga(640,480), square(224,224);
    const cv::Size sources[] = {vga, fullHD, vga, vga, vga, vga, vga};
    const cv::Size outputs[] = {square, cv::Size(910,512), cv::Size(320,240),
                               square, square, square, cv::Size(601,439)};
    const int types[] = {CV_8UC3,CV_8UC3,CV_8UC3,CV_8UC1,CV_8UC4,CV_32FC3,CV_8UC3};
    for (int i = 0; i < 7; ++i)
    {
        cv::Mat src(sources[i], types[i]), dst;
        rng.fill(src, cv::RNG::UNIFORM, 0, 256);
        const int repeat = i == 1 ? 20 : 180;
        std::vector<double> medians;
        for (int block = -1; block < 9; ++block)
        {
            const auto start = std::chrono::steady_clock::now();
            for (int j = 0; j < repeat; ++j)
                cv::resize(src, dst, outputs[i], 0, 0, cv::INTER_AREA);
            const auto end = std::chrono::steady_clock::now();
            if (block >= 0)
                medians.push_back(std::chrono::duration<double,std::micro>(end-start).count()/repeat);
        }
        std::sort(medians.begin(), medians.end());
        std::printf("%dx%d to %dx%d type %d: %.2f us\n",src.rows,src.cols,dst.rows,dst.cols,types[i],medians[4]);
    }
}
