# 2023-alimama-ai_Infer_optim
2023年alimama AI推理优化赛道 复赛



运行方式
bash init.sh 
初始化环境，这一步是服务器重新启动的时候必须做的一步，只需要在服务器重启之后执行一次
bash Build.sh 
编译代码，其中在model.cc 存在参数是否开启FP16优化以及INT8优化，默认使用FP16优化
INT8可以运行，但是量化误差比较大，并且实际测试速度没有FP16快,目前还未找到原因(问了一个大佬，tensorrt版本太低，唉！！！！)

bash run.sh
运行代码，在warmup阶段进行量化误差
***请注意，由于onnx ---> trt 时间比较长，
***程序会先判断是否存在model.trt，如果存在的话，那么直接读取model.trt文件，不存在，进行onnx ---> trt 过程
***如果测试了FP16之后，想要测试INT8的效果，那么请将blaze-benchmark目录下的model.trt删除，重新编译运行。



2023-7-10 完成tensorflow ---> onnx ---> trt 动态推理 FP16 FP32
导出命令：
python -m tf2onnx.convert --graphdef frozen_graph_origin.pb --inputs comm:0  --outputs output:0 --opset 11  --output model.onnx
2023-7-11 完成trt consumer的实现,改写了推理池逻辑，使用队列进行实现
trt 单线程 1700
trt 推理池逻辑，使用队列进行实现 单线程 2200-2400不等
2023-7-15 完成了误差计算，这里计算的是相对误差平均值
trt                                                             相对误差                   速度
FP16                                                              0.2%                   2300
FP32                                                              0.000132406%           2014
固定尺寸推理+FP16                                                  0.294638%              4673
2023-7-16 计算误差有问题，重新计算，99%的数据量的不超过1%相对误差，
trt                          相对误差                                超过1%相对误差的数据占比        速度
FP16                          0.2%                                     0.0%                       2300
FP32                          0.000132406%                             0.0%                       2014
固定尺寸推理+FP16*             0.294638%                                0.0%                       4673
FP16+设置最优推理尺度*         0.294638%                                 0.0%                      5300+
INT8                          8.1867%                                  0.0%                       4000+
分析带*的主要提升是在于在推理器进行初始化的时候，
我们在model.h/PredictContext的构造函数中生成最大显存，
这样每次使用该推理器的时候就不会重复申请和释放显存了
并且每个推理器都使用自己的内存区域，不会出现线程互斥的问题
在析构函数里面释放了显存，防止内存泄露。
2023-7-16
查看很多tensorrt官方文档，官方不建议使用多线程的tensorrt推理，建议使用多 batchSize的推理
github 上面有大佬做过对比试验
https://github.com/axxx-xxxa/Yolo-deploy-framework.git
发现多线程不能提高推理的效果
我们为了兼容显卡驱动，使用tensorrt7.2.1.6，到官方的github去提问，也是让先使用tensorrt8.6
我们也尝试安装tensorrt8.6和cuda11.2,cudann8.6 但是在onnx ---> trt 会报错，并且比赛快要结束，害怕破坏整个环境，没有继续深入修改liunx环境
其实，关于多线程推理，网上的资料很少，我们也是按照官方提供的生产者和消费者去更改的。
https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-803/best-practices/index.html#thread-safety
按照Tensorrt官方的文档,Tensorrt的builder生成的context是线程不安全，于是我们对于每一个PredictContext都去使用不同的builder，并且使用的context都是用不同的builder生成的，但是还是推理速度没有任何提升。目前问题卡在这里了。
2023-7-16
睡觉之前看了一眼文档，发现仅enqueue/enqueueV2 异步接口支持多线程，但是我一直使用的是execute/executeV2,修改之后发现单线程从5300 ---> 5500了,多线程 2700 ---> 3500 说明在tensorrt内部推理的过程中，存在资源的互斥现象。
trt                          相对误差                                超过1%相对误差的数据占比        速度
FP16                          0.2%                                     0.0%                       2300
FP32                          0.000132406%                             0.0%                       2014
固定尺寸推理+FP16*             0.294638%                                0.0%                       4673
FP16+设置最优推理尺度*         0.294638%                                0.0%                       5300+
INT8                          8.1867%                                  0.0%                       4000+
FP16+最优尺度+enqueueV2        0.294638%                                0.0%                       5500+
FP16+最优尺度+enqueueV2+多线程  0.294638%                                0.0%                       3500+
