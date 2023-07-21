file(GLOB_RECURSE SOURCE
        "common/*.cc"
        "core/*.cc")

set(proto_gen_folder "${PROJECT_BINARY_DIR}/benchmark/proto")
file(GLOB_RECURSE proto_files "proto/*.proto")
protobuf_generate_cpp_py(${proto_gen_folder} proto_srcs proto_hdrs proto_python
        "${PROJECT_SOURCE_DIR}/benchmark" "proto" ${proto_files})
list(APPEND SOURCE ${proto_srcs})
MESSAGE(${proto_srcs})


include_directories(/usr/local/cuda-10.2/include)
link_directories(/usr/local/cuda-10.2/lib64)
include_directories(/root/TensorRT-8.5.1.7/include)
link_directories(/root/TensorRT-8.5.1.7/lib)
include_directories(/usr/local/cuda-10.2/targets/x86_64-linux/include/)


add_executable(benchmark ${SOURCE})

target_link_libraries(benchmark
    tensorflow_cc
    tensorflow_framework
    cppmetrics
    boost_chrono
    boost_system
    boost_thread
    glog
    pthread
    nvinfer
    nvonnxparser
    -L/usr/local/cuda/lib64/ -lcudart
    )
if (USE_CUDA)
    target_link_libraries(benchmark cudart)
endif ()
