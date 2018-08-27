#include <iostream>
#include <string>
#include <fstream>
#include <memory>
#include <ctime>
#include <chrono>
#include "rsa_face_detection.hpp"

using milli = std::chrono::milliseconds;
int main(){
	std::string img_list = "./image/list.txt";
	std::ifstream list(img_list);
	std::string  img_name;
	RsaFaceDetectorMx *detector = new RsaFaceDetectorMx("./model/", 0);
	while(list >> img_name){
		std::cout << img_name << "\n";
		cv::Mat img = cv::imread(img_name);
		if(!img.empty()){
			std::cout << "Image width: " << img.cols << " Image height: " << img.rows << std::endl;
			auto start = std::chrono::high_resolution_clock::now();
			std::vector<Face> faces = detector->detect(img);	
			auto end = std::chrono::high_resolution_clock::now();
			for(int i = 0; i < faces.size(); ++i){
				cv::rectangle(img, cv::Point(faces[i].bbox[0], faces[i].bbox[1]),
						cv::Point(faces[i].bbox[2], faces[i].bbox[3]),
						cv::Scalar(0,0,255), 1, 1, 0);
				for(int j = 0; j < 5; ++j){
					cv::circle(img, faces[i].key_points[j], 1, cv::Scalar(0,255,0), 2);
				}
			}
			auto i = img_name.end() - 1;
			while(i >= img_name.begin() && *i != '/'){
				--i;
			}
			std::string output_name = "./image/output";
			for(auto j = i; j != img_name.end(); ++j){
				output_name += *j;
			}
			cv::imwrite(output_name, img);
			// cv::imshow("test", img);
			// cv::waitKey(0);
		}
		else{
			std::cerr << "Bad image: " << img_name << "\n";
		}
	}

	// cv::Mat img0 = cv::imread("image/test_image/1.jpg");
	// //warmup
	// for (int i=0; i<1; i++){
	// 	printf("start i = %f\n", i);
	// 	std::vector<Face> faces = detector->detect(img0);
	// 	faces.clear();
	// }
	// clock_t start_time, finish_time;
	// start_time=clock();
	// // auto start = std::chrono::high_resolution_clock::now();
	// std::vector<Face> faces = detector->detect(img0);
	// // auto end = std::chrono::high_resolution_clock::now();
	// finish_time=clock();
	// faces.clear();
	// // std::cout << "lrn took " << std::chrono::duration_cast<milli>(end - start).count() << " ms\n";
	// printf("time: %.0lfms\n", (finish_time-start_time)*1000.0 / CLOCKS_PER_SEC);
	// printf("faces.size()=%d", faces.size());
	// for(int i = 0; i < faces.size(); ++i){
	// 	cv::rectangle(img0, cv::Point(faces[i].bbox[0], faces[i].bbox[1]),
	// 			cv::Point(faces[i].bbox[2], faces[i].bbox[3]),
	// 			cv::Scalar(0,0,255), 1, 1, 0);
	// 	for(int j = 0; j < 5; ++j){
	// 		cv::circle(img0, faces[i].key_points[j], 1, cv::Scalar(0,255,0), 2);
	// 	}
	// }
	// cv::imwrite("image/output/1.jpg", img0);

	delete detector;

	return 0;
}

