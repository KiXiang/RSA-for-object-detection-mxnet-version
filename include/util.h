#ifndef _UTIL_H_
#define _UTIL_H_
#include <opencv2/opencv.hpp>
#include <mxnet-cpp/MxNetCpp.h>
#include <map>
#include <string>
#include <vector>

using namespace std;
using namespace mxnet::cpp;
using namespace cv;


typedef struct Face
{
    std::vector<double> bbox;
    std::vector<cv::Point2f> key_points;
    float score;
}Face;

class Blob:public NDArray {
public:
    Blob():mxnet::cpp::NDArray(){
        ;
    }
    Blob(const Shape &shape, const Context &context=Context::cpu()):
       NDArray(shape,context){
    }
    Blob(const std::vector<mx_uint> shape, const mxnet::cpp::Context &context=Context::cpu()):
       mxnet::cpp::NDArray(shape,context){
    }
    inline mx_float *cpu_data() const {
        return (mx_float *)GetData();
    }
    inline size_t count() const {
        return NDArray::Size();
    }
    inline size_t nums() const{
        return GetShape()[0];
    }
    inline size_t channels() const{
        return GetShape()[1];
    }
    inline size_t width() const{
        return GetShape()[3];
    }
    inline size_t height() const{
        return GetShape()[2];
    }
    inline std::vector<mx_uint> GetShape() const{
        return NDArray::GetShape();
    }
    inline void Reshape(int num,int ch,int width,int height){
        NDArray::Reshape(Shape(num, ch, width, height)); 
    }
    inline void AddNDArrayVector(std::vector<NDArray> data_vector){
        int w=width();
        int h=height();
        int ch=channels();
        int num=nums();
        int size=count()/num;
        Blob tmp(Shape(num,ch,w,h),Context::cpu());
        for (int i=0;i<data_vector.size();++i) {
            memcpy(&tmp.cpu_data()[i*size],data_vector[i].GetData(),size*sizeof(float));

        }
        tmp.CopyTo(this);
    }
    virtual ~Blob(){
    }
};

class Net{
public:
    Net(const std::string &symbol_name,const std::string &para_name,Shape shape=Shape(0,0,0,0),int n=0)
    {
        device_id_ = n;
        Symbol net;
        net_   = Symbol::Load(symbol_name);
        LoadParameters(GetContext(),para_name);
        executor_ =NULL;
        if (shape[2] && shape[3]) {
            args_map_["data"] = NDArray(shape,GetContext());
            // executor_ = net_.SimpleBind(GetContext(), args_map_);
            executor_ = net_.SimpleBind(GetContext(), args_map_, map<string, NDArray>(), map<string, OpReqType>(), aux_map_);
        }
    }
    Context  GetContext() const{
        if (devicetype_==mxnet::cpp::kGPU) {
            return Context::gpu(device_id_);
        }
        return Context::cpu(); 
    }
    const std::vector<mx_uint> & shape() {
        std::vector<mx_uint>  test=args_map_["data"].GetShape();
       return args_map_["data"].GetShape();
    }
    void Reshape(){
        //executor_->Reshape();
    }
	void ForwardEnd()
	{
	    if(executor_)
		    delete executor_;
		executor_=NULL;
	}
    void Forward(Blob *input=nullptr){
        bool is_train=false;
        if (input) {
            if(executor_)delete executor_;
            if (input->GetContext().GetDeviceType()==devicetype_) {
                args_map_["data"] = *input; 
            }else{
                Blob data(input->GetShape(),GetContext());
                input->CopyTo(&data);
                args_map_["data"] = data; 
            }
            // executor_ = net_.SimpleBind(GetContext(), args_map_);
            executor_ = net_.SimpleBind(GetContext(), args_map_, map<string, NDArray>(), map<string, OpReqType>(), aux_map_);
        }
        executor_->Forward(is_train); 
		out_put_flag = false;
    }
    void Forward_rsa(Blob *input = nullptr)
    {
        bool is_train = false;
        if (input)
        {
            if (executor_)
                delete executor_;
            if (input->GetContext().GetDeviceType() == devicetype_)
            {
                args_map_["res2b"] = *input;
            }
            else
            {
                Blob data(input->GetShape(), GetContext());
                input->CopyTo(&data);
                args_map_["res2b"] = data;
            }
            // executor_ = net_.SimpleBind(GetContext(), args_map_);
            executor_ = net_.SimpleBind(GetContext(), args_map_, map<string, NDArray>(), map<string, OpReqType>(), aux_map_);
        }
        executor_->Forward(is_train);
        out_put_flag = false;
    }
    inline const std::vector<Blob*> output_blobs()
	{
	    if(!out_put_flag)
		{
		    for(int i=0;i<out_blobs.size(); i++)
			{
			    delete out_blobs[i];
			}
			out_blobs.erase( out_blobs.begin(),out_blobs.end() );
			for (auto &output:executor_->outputs) 
			{
				Blob *data=new Blob(output.GetShape(),Context::cpu());
				output.SyncCopyToCPU(data->cpu_data(),data->Size());
				out_blobs.push_back(data); 			
			}
			
		}        
        out_put_flag = true;
        return out_blobs;
    }
    inline const std::vector<Blob*> input_blobs(){
        in_blobs.erase( in_blobs.begin(),in_blobs.end() );
        in_blobs.push_back((Blob*)&args_map_["data"]);
        return in_blobs;
    }
    void set_mode(DeviceType type)
    {
        devicetype_=type;
    }
    virtual ~Net()
	{
	     for(int i=0;i<out_blobs.size(); i++)
		{
		    delete out_blobs[i];
		}
        if(executor_)
            delete executor_;
    }

private:
    /*Fill the trained paramters into the model, a.k.a. net, executor*/
    void LoadParameters(Context ctx,string file) {
      map<string, NDArray> paramters;
      NDArray::Load(file, 0, &paramters);
      for (const auto &k : paramters) {
        if (k.first.substr(0, 4) == "aux:") {
          auto name = k.first.substr(4, k.first.size() - 4);
          aux_map_[name] = k.second.Copy(ctx);
        //   printf("aux name: -> %s\n", name.c_str());
        }
        if (k.first.substr(0, 4) == "arg:") {
          auto name = k.first.substr(4, k.first.size() - 4);
          args_map_[name] = k.second.Copy(ctx);
        //   printf("arg name: -> %s\n", name.c_str());
        }
      }
      /*WaitAll is need when we copy data between GPU and the main memory*/
      NDArray::WaitAll();
    }
    map<string, NDArray> args_map_;
    map<string, NDArray> aux_map_;
    Symbol net_;
    Executor * executor_;
    std::vector<Blob*>out_blobs;
    std::vector<Blob*>in_blobs;
    DeviceType devicetype_=mxnet::cpp::kGPU;
    int device_id_=0;
	bool out_put_flag;
};
const double dst_landmark[10] = {30.2946, 65.5318, 48.0252, 33.5493, 62.7299,51.6963, 51.5014, 71.7366, 92.3655, 92.2041 };

static bool CvMatToNDArraySignalChannel(const cv::Mat& cv_mat, Blob* data){
  if (cv_mat.empty())
    return false;

    int size = cv_mat.rows * cv_mat.cols * cv_mat.channels();
    mx_float* image_data= data->cpu_data();
    mx_float* ptr_image_b = image_data;
    mx_float* ptr_image_g = image_data + size / 3;
    mx_float* ptr_image_r = image_data + size / 3 * 2;

    for (int i = 0; i < cv_mat.rows; i++) {
        const float * data = cv_mat.ptr<float>(i);
        for (int j = 0; j < cv_mat.cols; j++) {
            *ptr_image_r++ = *data++;
            *ptr_image_g++ = *data++;
            *ptr_image_b++ = *data++;
        }
    }
    return true;
}

static bool MultyCvMatToNDArraySignalChannel(const std::vector<cv::Mat>& cv_mats, Blob* data){
  int size = cv_mats[0].rows * cv_mats[0].cols * cv_mats[0].channels();
  mx_float* image_data = data->cpu_data();

  for (int num=0; num<cv_mats.size(); num++){
    mx_float* ptr_image_r = image_data;
    mx_float* ptr_image_g = image_data + size / 3;
    mx_float* ptr_image_b = image_data + size / 3 *2;
    cv::Mat cv_mat = cv_mats[num];
    for (int i = 0; i < cv_mat.rows; i++) {
        const float * data = cv_mat.ptr<float>(i);
        for (int j = 0; j < cv_mat.cols; j++) {
            *ptr_image_r++ = *data++;
            *ptr_image_g++ = *data++;
            *ptr_image_b++ = *data++;
        }
    }
    image_data += size;
  }
  return true;
}


#endif