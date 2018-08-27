#ifndef _RSA_FACE_DETECTION_MX_H_
#define _RSA_FACE_DETECTION_MX_H_
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <memory>
#include <algorithm>
#include <cmath>
#include <Eigen/LU>
#include <Eigen/Dense>
#include "gpu_nms.hpp"
#include "util.h"


class RsaFaceDetectorMx
{
  public:
    RsaFaceDetectorMx(const std::string &model_dir, int gpu_id);
    std::vector<Face> detect(cv::Mat image);
    void sfnProcess(const cv::Mat &img);
    void rsaProcess(void);
    void lrnProcess(std::vector<Face> &faces_out);

  private:
    Eigen::MatrixXd findNonreflectiveSimilarity(const cv::Point2f uv[], const cv::Point2f xy[]);
    cv::Mat getSimilarityTransform(const cv::Point2f uv[], const cv::Point2f xy[]);
    void getTripPoints(std::vector<cv::Point2f> &dst_rect, cv::Point2f src_key_point[]);

    int gpu_id_;
    // std::vector<std::shared_ptr<caffe::Blob<float>>> trans_featmaps;
    std::vector<Blob> trans_featmaps;
    // caffe::Blob<float> *sfn_net_output;
    // caffe::Blob<float> *input_layer;
    // caffe::Blob<float> *rsa_input_layer;
    // caffe::Blob<float> *lrn_input_layer;
    Blob* sfn_net_output;
    // Blob* input_layer;
    Blob* rsa_input_layer;
    Blob* lrn_input_layer;
    std::vector<cv::Mat> input_channels;
    double resize_factor;
    std::vector<float> anchor_box_len;
    double thresh_score;
    double stride;
    double anchor_center;
    std::vector<int> scale;

    Net *sfn_net;
    Net *rsa_net;
    Net *lrn_net;

    const float ANCHOR_BOX[4] = {-44.754833995939045, -44.754833995939045, 44.754833995939045, 44.754833995939045};
    const float ANCHOR_PTS[10] = {-0.1719448, -0.2204161, 0.1719447, -0.2261145, -0.0017059, -0.0047045, -0.1408936, 0.2034478, 0.1408936, 0.1977626};
    //?Why define two thresh value
    const float NMS_THRESH = 0.4;
    const float THRESH_CLS = 3.0;
    const float THRESH_SCORE = 8.0;
    const float ANCHOR_CENTER = 7.5;
    const float STRIDE = 16.0;
    const int MAX_IMG = 2048;
    const int MIN_IMG = 64;
};

#endif
