#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include <cstdio>
#include <cstdlib>

int main(int argc, char** argv)
{
    cv::setNumThreads(argc > 1 ? std::atoi(argv[1]) : 1);
    cv::RNG rng(137);
    const int depths[] = {CV_8U, CV_16U, CV_16S, CV_32F, CV_64F};
    for (int i = 0; i < 180; ++i)
    {
        const int mode = i % 6;
        cv::Size from(rng.uniform(9, 1025), rng.uniform(9, 1025));
        cv::Size to(rng.uniform(1, from.width), rng.uniform(1, from.height));
        if (mode == 0) { from = cv::Size(640, 480); to = cv::Size(224, 224); }
        if (mode == 1) { from = cv::Size(1920, 1080); to = cv::Size(910, 512); }
        if (mode == 2) { from = cv::Size(1280, 720); to = cv::Size(1037, 583); }
        if (mode == 3) { from = cv::Size(640, 480); to = cv::Size(320, 240); }
        if (mode == 4) { from = cv::Size(640, 480); to = cv::Size(100, 76); }
        for (int depth : depths)
        {
            const int channels = 1 + (i / 6) % 4;
            cv::Mat storage(from.height + 2, from.width + 4, CV_MAKETYPE(depth, channels));
            cv::Mat src = storage(cv::Rect(1, 1, from.width, from.height)), dst;
            rng.fill(src, cv::RNG::UNIFORM,
                     depth == CV_16S || depth >= CV_32F ? -100.0 : 0.0,
                     depth == CV_8U ? 256.0 : 100.0);
            if (i % 2)
                src = src.clone();
            cv::resize(src, dst, to, 0, 0, cv::INTER_AREA);
            for (int y = 0; y < dst.rows; ++y)
                if (std::fwrite(dst.ptr(y), dst.elemSize(), static_cast<size_t>(dst.cols), stdout) != static_cast<size_t>(dst.cols))
                    return 1;
        }
    }
    return std::fflush(stdout) == 0 ? 0 : 1;
}
