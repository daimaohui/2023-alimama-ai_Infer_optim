#include <fstream>
#include "benchmark/core/predict_request_consumer.h"

using namespace tensorflow;
namespace benchmark {

PredictRequestConsumer::PredictRequestConsumer(
    benchmark::ModelSelector *model_selector,
    benchmark::PredictRequestQueue *predict_queue,
    benchmark::Metrics *metrics, int max_queue_size) {
  model_selector_ = model_selector;
  predict_queue_ = predict_queue;
  metrics_ = metrics;
  max_queue_size_ = max_queue_size;
}

void PredictRequestConsumer::Start() {
  while (!metrics_->IsStopped()) {
    PredictRequest *predict_request = predict_queue_->Dequeue();
    if (!predict_request) {
      std::this_thread::sleep_for(std::chrono::microseconds(1000));
      continue;
    }

    int model_idx = predict_request->model_idx;
    std::shared_ptr<Model> model = model_selector_->GetModel(model_idx);
    if (!model) {
      LOG(ERROR) << "model_idx: " << model_idx << " out of range.";
      return;
    }
    if (max_queue_size_ > 0 && predict_queue_->size() > max_queue_size_) {
      metrics_->UpdateGetPredictorFailures(model->name());
      VLOG(2) << "Drop request: number of outstanding requests exceeds max_queue_size.";
      continue;
    }
    PredictContext* predict_context = model->Borrow();
    if (!predict_context) {
      predict_queue_->Enqueue(predict_request);
      continue;
    }
    auto bef = std::chrono::high_resolution_clock::now();
    int batchsize = 0;
    float* comm;
    float* ncomm;
    model->GetOneInput(&batchsize,comm,ncomm);
    if (this->PredictImpl(predict_context, &batchsize,comm,ncomm)) {
      metrics_->UpdateThroughput(model->name());
    } else {
      metrics_->UpdateFailures(model->name());
    }
    auto aft = std::chrono::high_resolution_clock::now();
    auto dur = std::chrono::duration_cast<std::chrono::microseconds>(aft - bef).count();
    metrics_->UpdateLatency(model->name(), dur);
    metrics_->UpdateBatchsize(model->name(), batchsize);
    model->Return(predict_context);
  }
}

bool PredictRequestConsumer::PredictImpl(
    benchmark::PredictContext* predict_context, int* batchsize,float* comm,float* ncomm) {
    auto m_CudaContext=predict_context->m_CudaContext;
    auto m_CudaStream=predict_context->m_CudaStream;
    auto m_ArrayDevMemory=predict_context->m_ArrayDevMemory;
    cudaMemcpyAsync(m_ArrayDevMemory[1], ncomm, *batchsize* 384 * sizeof(float), cudaMemcpyHostToDevice, m_CudaStream);
    cudaMemcpyAsync(m_ArrayDevMemory[0], comm, 176 * sizeof(float), cudaMemcpyHostToDevice, m_CudaStream);
    // m_CudaContext->executeV2(m_ArrayDevMemory);
    m_CudaContext->enqueueV2(m_ArrayDevMemory, m_CudaStream, nullptr);
    void* result=malloc(*batchsize*2 * sizeof(float));
    cudaMemcpyAsync(result, m_ArrayDevMemory[2], *batchsize*2 * sizeof(float), cudaMemcpyDeviceToHost, m_CudaStream);
    cudaStreamSynchronize(m_CudaStream);
    // float* result=(fli)
    free(result);
    return true;
}

}  // namespace benchmark
