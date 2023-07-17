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
    PredictContext(nvinfer1::IExecutionContext* m_CudaContext_,int maxBatchSize_) {
       m_CudaContext=m_CudaContext_;
       m_CudaStream=createCudaStream();
       maxBatchSize=maxBatchSize_;
       cudaMalloc(&m_ArrayDevMemory[0],  176 * sizeof(float));
       cudaMalloc(&m_ArrayDevMemory[1], maxBatchSize_ * 384 * sizeof(float));
       cudaMalloc(&m_ArrayDevMemory[2], maxBatchSize_ * 2 * sizeof(float));
    }
    ~PredictContext(){
      for(auto &p:m_ArrayDevMemory){
        cudaFree(p);
      }
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
    void *m_ArrayDevMemory[3]{0};
    int maxBatchSize;
    cudaStream_t m_CudaStream;
};

class Int8EntropyCalibrator : public nvinfer1::IInt8EntropyCalibrator2
{
public:
    Int8EntropyCalibrator(std::vector<std::vector<float>> ncomm_input_data_,
    std::vector<std::vector<float>> comm_input_data_,
    std::vector<int> inferred_batchsizes_,int maxBatchSize_
    ) {
      m_CudaStream=createCudaStream();
      maxBatchSize=maxBatchSize_;
      cudaMalloc(&m_ArrayDevMemory[0],  176 * sizeof(float));
      cudaMalloc(&m_ArrayDevMemory[1], maxBatchSize_ * 384 * sizeof(float));
      cudaMalloc(&m_ArrayDevMemory[2], maxBatchSize_ * 2 * sizeof(float));
      ncomm_input_data=ncomm_input_data_;
      comm_input_data=comm_input_data_;
      inferred_batchsizes=inferred_batchsizes_;
    }
    virtual ~Int8EntropyCalibrator(){
      for(auto &p:m_ArrayDevMemory){
        free(p);
      }
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
    // 想要按照多少的batch进行标定
    int getBatchSize() const noexcept {
        return maxBatchSize;
    }

    bool next() {
        if(batchIndex>=inferred_batchsizes.size()){
          return false;
        }
        cudaMemcpyAsync(m_ArrayDevMemory[0], comm_input_data[batchIndex].data(), 176 * sizeof(float), cudaMemcpyHostToDevice, m_CudaStream);
        cudaMemcpyAsync(m_ArrayDevMemory[1], ncomm_input_data[batchIndex].data(), inferred_batchsizes[batchIndex]* 384 * sizeof(float), cudaMemcpyHostToDevice, m_CudaStream);
        batchIndex++;
        return true;
    }

    bool getBatch(void* bindings[], const char* names[], int nbBindings) noexcept {
        if (!next()) return false;
        bindings[0] = m_ArrayDevMemory[0];
        bindings[1] = m_ArrayDevMemory[1];
        return true;
    }

    const std::vector<uint8_t>& getEntropyCalibratorData() {
        return entropyCalibratorData_;
    }

    const void* readCalibrationCache(size_t& length) noexcept {
        if (fromCalibratorData_) {
            length = this->entropyCalibratorData_.size();
            return this->entropyCalibratorData_.data();
        }

        length = 0;
        return nullptr;
    }

    virtual void writeCalibrationCache(const void* cache, size_t length) noexcept {
        entropyCalibratorData_.assign((uint8_t*)cache, (uint8_t*)cache + length);
    }

private:
    void *m_ArrayDevMemory[3]{0};
    int maxBatchSize;
    cudaStream_t m_CudaStream;
    std::vector<uint8_t> entropyCalibratorData_;
    bool fromCalibratorData_ = false;
    std::vector<std::vector<float>> ncomm_input_data;
    std::vector<std::vector<float>> comm_input_data;
    std::vector<int> inferred_batchsizes;
    int batchIndex=0;
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
