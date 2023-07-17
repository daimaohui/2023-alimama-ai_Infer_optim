#pragma once

#include <condition_variable>
#include <map>

#include "benchmark/common/double_buffer_reloader.h"
#include "benchmark/proto/bench_conf.pb.h"

#include "tensorflow/core/public/session.h"
#include "tensorflow/core/framework/tensor.h"
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>
#include <sys/stat.h>
#include<deque>
#include<fstream>
using namespace tensorflow;
namespace benchmark {
class PredictContext {
public:
    PredictContext(nvinfer1::IExecutionContext* m_CudaContext_1) {
       m_CudaContext=m_CudaContext_1;
       m_CudaStream=createCudaStream();
    }
    cudaStream_t createCudaStream() {
        cudaStream_t stream;
        cudaError_t cudaErr = cudaStreamCreate(&stream);
        if (cudaErr != cudaSuccess) {
            // 处理 CUDA 流创建失败的情况
            // ...
        }
        return stream;
    }
    nvinfer1::IExecutionContext* m_CudaContext;
    cudaStream_t m_CudaStream;
};

class Model {
 public:
  Model(const std::string& name, int predictor_num) {
    name_ = name;
    predictor_num_ = predictor_num;
  }
  ~Model() {
    
  }
  
  bool Loadonnx(const std::string& onnx_path);
  PredictContext* Borrow();
  bool loadTrt();
  bool run();
  const std::string& name() const { return name_; }
  bool ParseSamples(const std::string& sample_file);
  bool Warmup();
  void Return(PredictContext* context);
  static bool ifFileExists(const char *FileName)
  {
      struct stat my_stat;
      return (stat(FileName, &my_stat) == 0);
  }
  void GetOneInput(int* batchsize,float* comm,float* ncomm) const {
    static int idx = 0;
    int temp = idx++ % ncomm_input_data.size();
    *batchsize = inferred_batchsizes_[temp];
    comm=const_cast<float*>(comm_input_data[temp].data());
    ncomm=const_cast<float*>(ncomm_input_data[temp].data());
  }
 protected:
  std::string trt_name_;
  std::string name_;
  int predictor_num_;
  std::mutex mutex_;
  std::condition_variable cond_;
  std::deque<PredictContext*> PredictContexts;
  int maxBatchSize=0;
  std::vector<int> inferred_batchsizes_;
  std::vector<std::string> output_names_;
  std::vector<std::vector<float>> ncomm_input_data;
  std::vector<std::vector<float>> comm_input_data;
  
};

class ModelReloader : public DoubleBufferReloader<Model> {
 public:
  ModelReloader(const benchmark::BenchModelConfig& bench_model_config)
      : bench_model_config_(bench_model_config) {}

  virtual Model* CreateObject() override;

 private:
  benchmark::BenchModelConfig bench_model_config_;
};

class ModelSelector {
 public:
  ModelSelector() {}
  virtual ~ModelSelector() = default;

  void Start();
  void Stop();

  bool InitModel(const benchmark::BenchModelConfig& bench_model_config);
  std::shared_ptr<Model> GetModel(int idx) const;

 private:
  std::vector<std::shared_ptr<ModelReloader>> model_reloaders_;
  std::vector<int> switch_interval_;
  std::atomic_bool running_;
};

}  // namespace benchmark
