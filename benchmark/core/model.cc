#include "benchmark/core/model.h"
#include "benchmark/proto/sample.pb.h"

#include "tensorflow/core/common_runtime/direct_session.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"
#include "tensorflow/core/protobuf/saved_model.pb.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/dump_graph.h"
#include "tensorflow/tools/graph_transforms/transform_utils.h"
#include "./logging.h"
#define USE_FP16
using namespace tensorflow;
namespace benchmark {
PredictContext *Model::Borrow() {
  std::unique_lock<std::mutex> lock(mutex_);
  // std::unique_lock<std::mutex> lock(mutex_);
  while (PredictContexts.empty()) {
    cond_.wait(lock); // 等待条件变量通知
  }
  PredictContext* context = PredictContexts.front();
  PredictContexts.pop_front();
  return context;
}

void Model::Return(PredictContext *predict_context) {
  std::unique_lock<std::mutex> lock(mutex_);
  PredictContexts.push_back(predict_context);
  cond_.notify_one(); // 唤醒一个等待线程
}

bool Model::ParseSamples(const std::string& sample_file) {
  if (sample_file.empty()) {
    LOG(ERROR) << "Samplefile path must not be empty: " << name();
    return false;
  }
  SamplesProto samples_proto;
  Status s = ReadBinaryProto(Env::Default(), sample_file.c_str(), &samples_proto);
  if (!s.ok()) {
    s = ReadTextProto(Env::Default(), sample_file.c_str(), &samples_proto);
    if (!s.ok()) {
      LOG(ERROR) << "Read sample_file failed: " << name() << ", " << s.ToString();
      return false;
    }
  }
  VLOG(2) << "Samples_proto: " << samples_proto.DebugString();
  for (int i = 0; i < samples_proto.sample_size(); ++i) {
    const InputsProto& inputs_proto = samples_proto.sample(i);
    int64 batchsize = 1;
    for (int i = 0; i < samples_proto.output_names_size(); ++i) {
         output_names_.push_back(samples_proto.output_names(i));
         LOG(INFO) <<samples_proto.output_names(i);
    }
    for (int j = 0; j < inputs_proto.input_size(); ++j) {

      const NamedTensorProto& input = inputs_proto.input(j);
      Tensor tensor;
      if (!tensor.FromProto(input.tensor())) { 
        LOG(ERROR) << "Init tensor from proto failed.";
        return false;
      }
      if (tensor.dims() >=1 && tensor.dim_size(0) > batchsize)
        batchsize = tensor.dim_size(0);
      if(input.name()=="ncomm"){
         LOG(INFO)<<"ncomm:"<<tensor.flat<float>().size();
        std::vector<float> vec(tensor.flat<float>().data(), tensor.flat<float>().data() + tensor.flat<float>().size());
        ncomm_input_data.push_back(vec);
      }else{
        LOG(INFO)<<"comm:"<<tensor.flat<float>().size();
        std::vector<float> vec(tensor.flat<float>().data(), tensor.flat<float>().data() + tensor.flat<float>().size());
        comm_input_data.push_back(vec);
      }
    }
    inferred_batchsizes_.emplace_back(batchsize);
    maxBatchSize=maxBatchSize>batchsize?maxBatchSize:batchsize;
    LOG(INFO) << "Parsed input, inferred batchsize = " << batchsize;
  }
  LOG(INFO) << "Parse input_samples success, total "<<ncomm_input_data.size()<<"samples";
  return true;
}

bool Model::Loadonnx(const std::string& onnx_path) {
  //加载onnx 模型
  LOG(INFO)<<"加载onnx模型";
  size_t sep_pos = onnx_path.find_last_of(".");
  trt_name_ = onnx_path.substr(0, sep_pos) + ".trt";
  LOG(INFO)<<trt_name_;
  if (ifFileExists(trt_name_.c_str()))
  {
      LOG(INFO)<<"已经存在trt文件,可以直接加载";
      return true;
  }

	Logger gLogger;

   auto builder =  nvinfer1::createInferBuilder(gLogger);
   auto network = builder->createNetworkV2(1U);
    // 解析模型
    auto parser = nvonnxparser::createParser(*network, gLogger);
    if(!parser->parseFromFile(onnx_path.c_str(), (int)nvinfer1::ILogger::Severity::kWARNING)){
        std::cout << " parse onnx file fail ..." << std::endl;
        return -1;
    }
    auto config = builder->createBuilderConfig();
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(1<<30);
    auto profile = builder->createOptimizationProfile();
    auto input_tensor=network->getInput(1);
    auto input_dims = input_tensor->getDimensions();
    input_dims.d[0] = 1;
    profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMIN, input_dims);
    profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kOPT, input_dims);
    input_dims.d[0] = maxBatchSize;
    profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMAX, input_dims);

    config->addOptimizationProfile(profile);

#ifdef USE_FP16
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
#endif
#ifdef USE_INT8
    config->setFlag(nvinfer1::BuilderFlag::kINT8);
#endif
    auto engine = builder->buildEngineWithConfig(*network, *config);
    assert(engine);
    auto trtModelStream = engine->serialize(); //序列化 保存trt
    std::ofstream out(trt_name_.c_str(), std::ios::binary);
    if (!out.is_open()){
     std::cout << "打开文件失败!" <<std:: endl;
    }
    out.write(reinterpret_cast<const char*>(trtModelStream->data()), trtModelStream->size());
    out.close();
    parser->destroy();
    engine->destroy();
    network->destroy();
    config->destroy();
    builder->destroy();
  LOG(INFO)<<"加载完成";
  return true;
}
bool Model::loadTrt()
{
    Logger gLogger;
    nvinfer1::IRuntime *runtime = nvinfer1::createInferRuntime(gLogger);
    std::ifstream fin(trt_name_);
    std::string cached_engine = "";
    while (fin.peek() != EOF)
    {
        std::stringstream buffer;
        buffer << fin.rdbuf();
        cached_engine.append(buffer.str());
    }
    fin.close();
    auto m_CudaEngine = runtime->deserializeCudaEngine(cached_engine.data(), cached_engine.size(), nullptr);
    

    auto dims_i = m_CudaEngine->getBindingDimensions(0);
    LOG(INFO) << "input dims " << dims_i.d[0] << " " << dims_i.d[1];
    dims_i = m_CudaEngine->getBindingDimensions(1);
    LOG(INFO) << "input dims " << dims_i.d[0] << " " << dims_i.d[1];
    dims_i = m_CudaEngine->getBindingDimensions(2);
    LOG(INFO) << "output dims " << dims_i.d[0] << " " << dims_i.d[1];
    for(int i=0;i<predictor_num_;i++){
      auto m_CudaContext = m_CudaEngine->createExecutionContext();
      PredictContexts.push_back(new PredictContext(m_CudaContext));

    }
    runtime->destroy();
}
bool Model::Warmup() {
  LOG(INFO)<<"开始预热";
  //加载之前的result.txt里面的数据
  std::ifstream file("result.txt");
  std::vector<std::vector<float>> tensorflowResult;
  std::string line;
  while (std::getline(file, line)) {
    std::vector<float> row;
    std::stringstream ss(line);
    float value;

    while (ss >> value) {
      row.push_back(value);
    }

    tensorflowResult.push_back(row);
  }
  file.close();
  // for (const auto& row : tensorflowResult) {
  //   for (const auto& value : row) {
  //     std::cout << value << " ";
  //   }
  //   std::cout << std::endl;
  // }
  for(auto e:PredictContexts){
      auto m_CudaContext=e->m_CudaContext;
      auto m_CudaStream=e->m_CudaStream;
      long long sumtime=0;
      for(int j=0;j<10;j++){
        auto bef = std::chrono::high_resolution_clock::now();
        int target_num=0;
        float error=0.0;
        for(int i=0;i<ncomm_input_data.size();i++){
          void *m_ArrayDevMemory[3]{0};
          cudaMalloc(&m_ArrayDevMemory[1], inferred_batchsizes_[i] * 384 * sizeof(float));
          cudaMemcpyAsync(m_ArrayDevMemory[1], ncomm_input_data[i].data(), inferred_batchsizes_[i]* 384 * sizeof(float), cudaMemcpyHostToDevice, m_CudaStream);
          cudaMalloc(&m_ArrayDevMemory[0], 176 * sizeof(float));
          cudaMemcpyAsync(m_ArrayDevMemory[0], comm_input_data[i].data(), 176 * sizeof(float), cudaMemcpyHostToDevice, m_CudaStream);
          cudaMalloc(&m_ArrayDevMemory[2], inferred_batchsizes_[i]*2 * sizeof(float));
          nvinfer1::Dims dims5; 
          dims5.d[0] = inferred_batchsizes_[i];    // replace dynamic batch size with 1
          dims5.d[1] = 384;
          dims5.nbDims = 2;
          m_CudaContext->setBindingDimensions(1,dims5);
          m_CudaContext->executeV2(m_ArrayDevMemory);
          void* result=malloc(inferred_batchsizes_[i]*2 * sizeof(float));
          cudaMemcpyAsync(result, m_ArrayDevMemory[2], inferred_batchsizes_[i]*2 * sizeof(float), cudaMemcpyDeviceToHost, m_CudaStream);
          cudaStreamSynchronize(m_CudaStream);
          for (auto &p : m_ArrayDevMemory)
          {
              cudaFree(p);
              p = nullptr;
          }
          float* result_t=(float*)result;
          std::vector<float> row=tensorflowResult[i];
          for(int k=0;k<row.size();k++){
            error+=(abs(row[k]-(*(result_t+k)))/abs(row[k]));
            target_num++;
          }
        }
        LOG(INFO)<<"推理结果误差："<<error/target_num*100<<"%";
        auto aft = std::chrono::high_resolution_clock::now();
        auto dur = std::chrono::duration_cast<std::chrono::microseconds>(aft - bef).count();
        sumtime+=dur;
        LOG(INFO)<<ncomm_input_data.size()*1e6/dur;
    }
    LOG(INFO)<<10*ncomm_input_data.size()*1e6/sumtime;

  }
  LOG(INFO)<<"预热阶段结束";
  return true;
}

Model* ModelReloader::CreateObject() {
  Model *model= new Model(bench_model_config_.name(), bench_model_config_.predictor_num());
  if (!model->ParseSamples(bench_model_config_.sample_file())) {
    LOG(ERROR) << "Read sample_file failed: " << bench_model_config_.sample_file() << "," 
               << bench_model_config_.name();
    delete model;
    return nullptr;
  }
  if(!model->Loadonnx(bench_model_config_.frozen_graph())){
    delete model;
    return nullptr;
  }
  if(!model->loadTrt()){
    LOG(ERROR) << "loadTrt: " << bench_model_config_.name();
    return nullptr;
  }
  
  // Warmup
  if (!model->Warmup()) {
    LOG(ERROR) << "Warmup failed: " << bench_model_config_.name();
    delete model;
    return nullptr;
  }

  LOG(INFO) << "Init and warmup model complete: " << bench_model_config_.name();
  return model;
}

bool ModelSelector::InitModel(
    const benchmark::BenchModelConfig& bench_model_config) {
  std::shared_ptr<ModelReloader> model_reloader =
      std::make_shared<ModelReloader>(bench_model_config);
  bool success = model_reloader->Switch();
  if (!success) {
    return false;
  }
  model_reloaders_.emplace_back(model_reloader);
  switch_interval_.emplace_back(bench_model_config.switch_interval());
  return true;
}

std::shared_ptr<Model> ModelSelector::GetModel(int idx) const {
  auto model_reloader = model_reloaders_[idx];
  return model_reloader->Instance();
}

void ModelSelector::Start() {
  running_ = true;
  std::vector<int> left_time_to_switch(switch_interval_);
  while (running_.load()) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
    for (int i = 0; i < left_time_to_switch.size(); ++i) {
      left_time_to_switch[i]--;
      if (left_time_to_switch[i] <= 0) {
        LOG(INFO) << "Begin switch model.";
        bool success = model_reloaders_[i]->Switch();
        if (!success) {
          LOG(ERROR) << "Switch model failed.";
          continue;
        }
        LOG(INFO) << "Switch model successfully.";
        left_time_to_switch[i] = switch_interval_[i];
      }
    }
  }
}

void ModelSelector::Stop() { running_ = false; }

}  // namespace benchmark