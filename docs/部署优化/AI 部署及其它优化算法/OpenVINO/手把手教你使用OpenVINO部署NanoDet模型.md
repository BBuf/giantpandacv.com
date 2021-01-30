【GiantPandaCV】**本文为大家介绍了一个手把手使用OpenVINO部署NanoDet的教程，并开源部署的全部代码，在Intel i7-7700HQ CPU做到了6ms一帧的速度。**

# 0x0. nanodet简介

NanoDet （https://github.com/RangiLyu/nanodet）是一个速度超快和轻量级的Anchor-free 目标检测模型。想了解算法本身的可以去搜一搜之前机器之心的介绍。

# 0x1. 环境配置
- Ubuntu：18.04

- OpenVINO：2020.4

- OpenCV：3.4.2

- OpenVINO和OpenCV安装包（编译好了，也可以自己从官网下载自己编译）可以从链接: https://pan.baidu.com/s/1zxtPKm-Q48Is5mzKbjGHeg 密码: gw5c下载

- OpenVINO安装

```cpp
tar -xvzf l_openvino_toolkit_p_2020.4.287.tgz
cd l_openvino_toolkit_p_2020.4.287
sudo ./install_GUI.sh 一路next安装
cd /opt/intel/openvino/install_dependencies
sudo ./install_openvino_dependencies.sh
vi ~/.bashrc
```

- 把如下两行放置到bashrc文件尾

```cpp
source /opt/intel/openvino/bin/setupvars.sh
source /opt/intel/openvino/opencv/setupvars.sh
```

- source ~/.bashrc 激活环境

- 模型优化配置步骤

```cpp
cd /opt/intel/openvino/deployment_tools/model_optimizer/install_prerequisites
sudo ./install_prerequisites_onnx.sh（模型是从onnx转为IR文件，只需配置onnx依赖）
```

- OpenCV配置

tar -xvzf opencv-3.4.2.zip 解压OpenCV到用户根目录即可，以便后续调用。（这是我编译好的版本，有需要可以自己编译）

# 0x2. NanoDet模型训练和转换ONNX

- git clone https://github.com/Wulingtian/nanodet.git 

- cd nanodet

- cd config 配置模型文件，训练模型

- 定位到nanodet目录，进入tools目录，打开export.py文件，配置cfg_path model_path out_path三个参数

- 定位到nanodet目录，运行 python tools/export.py 得到转换后的onnx模型

- python /opt/intel/openvino/deployment_tools/model_optimizer/mo_onnx.py --input_model onnx模型 --output_dir 期望模型输出的路径。得到IR文件


# 0x3. NanoDet模型部署

- sudo apt install cmake 安装cmake

- git clone https://github.com/Wulingtian/nanodet_openvino.git （求star！）

- cd nanodet_openvino 打开CMakeLists.txt文件，修改OpenCV_INCLUDE_DIRS和OpenCV_LIBS_DIR，之前已经把OpenCV解压到根目录了，所以按照你自己的路径指定

- 定位到nanodet_openvino，cd models 把之前生成的IR模型（包括bin和xml文件）文件放到该目录下

- 定位到nanodet_openvino， cd test_imgs 把需要测试的图片放到该目录下

- 定位到nanodet_openvino，编辑main.cpp，xml_path参数修改为"../models/你的模型名称.xml"

- 编辑 num_class 设置类别数，例如：我训练的模型是安全帽检测，只有1类，那么设置为1

- 编辑 src 设置测试图片路径，src参数修改为"../test_imgs/你的测试图片"

- 定位到nanodet_openvino

- mkdir build; cd build; cmake .. ;make

- ./detect_test 输出平均推理时间，以及保存预测图片到当前目录下，至此，部署完成！


# 0x4. 核心代码一览

```cpp
//主要对图片进行预处理，包括resize和归一化
std::vector<float> Detector::prepareImage(cv::Mat &src_img){

    std::vector<float> result(INPUT_W * INPUT_H * 3);
    float *data = result.data();
    float ratio = float(INPUT_W) / float(src_img.cols) < float(INPUT_H) / float(src_img.rows) ? float(INPUT_W) / float(src_img.cols) : float(INPUT_H) / float(src_img.rows);
    cv::Mat flt_img = cv::Mat::zeros(cv::Size(INPUT_W, INPUT_H), CV_8UC3);
    cv::Mat rsz_img = cv::Mat::zeros(cv::Size(src_img.cols*ratio, src_img.rows*ratio), CV_8UC3);
    cv::resize(src_img, rsz_img, cv::Size(), ratio, ratio);

    rsz_img.copyTo(flt_img(cv::Rect(0, 0, rsz_img.cols, rsz_img.rows)));
    flt_img.convertTo(flt_img, CV_32FC3);

    int channelLength = INPUT_W * INPUT_H;
    std::vector<cv::Mat> split_img = {
            cv::Mat(INPUT_W, INPUT_H, CV_32FC1, data + channelLength * 2),
            cv::Mat(INPUT_W, INPUT_H, CV_32FC1, data + channelLength),
            cv::Mat(INPUT_W, INPUT_H, CV_32FC1, data)
    };

    cv::split(flt_img, split_img);
    for (int i = 0; i < 3; i++) {

        split_img[i] = (split_img[i] - img_mean[i]) / img_std[i];
    }

    return result;
}

//加载IR模型，初始化网络
bool Detector::init(string xml_path,double cof_threshold,double nms_area_threshold,int input_w, int input_h, int num_class, int r_rows, int r_cols, std::vector<int> s, std::vector<float> i_mean,std::vector<float> i_std){
    _xml_path = xml_path;
    _cof_threshold = cof_threshold;
    _nms_area_threshold = nms_area_threshold;
    INPUT_W = input_w;
    INPUT_H = input_h;
    NUM_CLASS = num_class;
    refer_rows = r_rows;
    refer_cols = r_cols;
    strides = s;
    img_mean = i_mean;
    img_std = i_std;
    Core ie;
    auto cnnNetwork = ie.ReadNetwork(_xml_path); 

    InputsDataMap inputInfo(cnnNetwork.getInputsInfo());
    InputInfo::Ptr& input = inputInfo.begin()->second;
    _input_name = inputInfo.begin()->first;
    input->setPrecision(Precision::FP32);
    input->getInputData()->setLayout(Layout::NCHW);
    ICNNNetwork::InputShapes inputShapes = cnnNetwork.getInputShapes();
    SizeVector& inSizeVector = inputShapes.begin()->second;
    cnnNetwork.reshape(inputShapes);
    _outputinfo = OutputsDataMap(cnnNetwork.getOutputsInfo());
    for (auto &output : _outputinfo) {
        output.second->setPrecision(Precision::FP32);
    }
    _network =  ie.LoadNetwork(cnnNetwork, "CPU");
    return true;
}
//模型推理及获取输出结果
vector<Detector::Bbox> Detector::process_frame(Mat& inframe){

    cv::Mat showImage = inframe.clone();
    std::vector<float> pr_img = prepareImage(inframe);
    InferRequest::Ptr infer_request = _network.CreateInferRequestPtr();
    Blob::Ptr frameBlob = infer_request->GetBlob(_input_name);
    InferenceEngine::LockedMemory<void> blobMapped = InferenceEngine::as<InferenceEngine::MemoryBlob>(frameBlob)->wmap();
    float* blob_data = blobMapped.as<float*>();

    memcpy(blob_data, pr_img.data(), 3 * INPUT_H * INPUT_W * sizeof(float));

    infer_request->Infer();
    vector<Rect> origin_rect;
    vector<float> origin_rect_cof;
    int i=0;
    vector<Bbox> bboxes;
    for (auto &output : _outputinfo) {
        auto output_name = output.first;
        Blob::Ptr blob = infer_request->GetBlob(output_name);
        LockedMemory<const void> blobMapped = as<MemoryBlob>(blob)->rmap();
        float *output_blob = blobMapped.as<float *>();
        bboxes = postProcess(showImage,output_blob);
        ++i;
    }
    return bboxes;
}
//对模型输出结果进行解码及nms
std::vector<Detector::Bbox> Detector::postProcess(const cv::Mat &src_img,
                              float *output) {
    GenerateReferMatrix();
    std::vector<Detector::Bbox> result;
    float *out = output;
    float ratio = std::max(float(src_img.cols) / float(INPUT_W), float(src_img.rows) / float(INPUT_H));
    cv::Mat result_matrix = cv::Mat(refer_rows, NUM_CLASS + 4, CV_32FC1, out);
    for (int row_num = 0; row_num < refer_rows; row_num++) {
        Detector::Bbox box;
        auto *row = result_matrix.ptr<float>(row_num);
        auto max_pos = std::max_element(row + 4, row + NUM_CLASS + 4);
        box.prob = row[max_pos - row];
        if (box.prob < _cof_threshold)
            continue;
        box.classes = max_pos - row - 4;
        auto *anchor = refer_matrix.ptr<float>(row_num);
        box.x = (anchor[0] - row[0] * anchor[2] + anchor[0] + row[2] * anchor[2]) / 2 * ratio;
        box.y = (anchor[1] - row[1] * anchor[2] + anchor[1] + row[3] * anchor[2]) / 2 * ratio;
        box.w = (row[2] + row[0]) * anchor[2] * ratio;
        box.h = (row[3] + row[1]) * anchor[2] * ratio;
        result.push_back(box);
    }
    NmsDetect(result);
    return result;
}//主要对图片进行预处理，包括resize和归一化
std::vector<float> Detector::prepareImage(cv::Mat &src_img){

    std::vector<float> result(INPUT_W * INPUT_H * 3);
    float *data = result.data();
    float ratio = float(INPUT_W) / float(src_img.cols) < float(INPUT_H) / float(src_img.rows) ? float(INPUT_W) / float(src_img.cols) : float(INPUT_H) / float(src_img.rows);
    cv::Mat flt_img = cv::Mat::zeros(cv::Size(INPUT_W, INPUT_H), CV_8UC3);
    cv::Mat rsz_img = cv::Mat::zeros(cv::Size(src_img.cols*ratio, src_img.rows*ratio), CV_8UC3);
    cv::resize(src_img, rsz_img, cv::Size(), ratio, ratio);

    rsz_img.copyTo(flt_img(cv::Rect(0, 0, rsz_img.cols, rsz_img.rows)));
    flt_img.convertTo(flt_img, CV_32FC3);

    int channelLength = INPUT_W * INPUT_H;
    std::vector<cv::Mat> split_img = {
            cv::Mat(INPUT_W, INPUT_H, CV_32FC1, data + channelLength * 2),
            cv::Mat(INPUT_W, INPUT_H, CV_32FC1, data + channelLength),
            cv::Mat(INPUT_W, INPUT_H, CV_32FC1, data)
    };

    cv::split(flt_img, split_img);
    for (int i = 0; i < 3; i++) {

        split_img[i] = (split_img[i] - img_mean[i]) / img_std[i];
    }

    return result;
}

//加载IR模型，初始化网络
bool Detector::init(string xml_path,double cof_threshold,double nms_area_threshold,int input_w, int input_h, int num_class, int r_rows, int r_cols, std::vector<int> s, std::vector<float> i_mean,std::vector<float> i_std){
    _xml_path = xml_path;
    _cof_threshold = cof_threshold;
    _nms_area_threshold = nms_area_threshold;
    INPUT_W = input_w;
    INPUT_H = input_h;
    NUM_CLASS = num_class;
    refer_rows = r_rows;
    refer_cols = r_cols;
    strides = s;
    img_mean = i_mean;
    img_std = i_std;
    Core ie;
    auto cnnNetwork = ie.ReadNetwork(_xml_path); 

    InputsDataMap inputInfo(cnnNetwork.getInputsInfo());
    InputInfo::Ptr& input = inputInfo.begin()->second;
    _input_name = inputInfo.begin()->first;
    input->setPrecision(Precision::FP32);
    input->getInputData()->setLayout(Layout::NCHW);
    ICNNNetwork::InputShapes inputShapes = cnnNetwork.getInputShapes();
    SizeVector& inSizeVector = inputShapes.begin()->second;
    cnnNetwork.reshape(inputShapes);
    _outputinfo = OutputsDataMap(cnnNetwork.getOutputsInfo());
    for (auto &output : _outputinfo) {
        output.second->setPrecision(Precision::FP32);
    }
    _network =  ie.LoadNetwork(cnnNetwork, "CPU");
    return true;
}
//模型推理及获取输出结果
vector<Detector::Bbox> Detector::process_frame(Mat& inframe){

    cv::Mat showImage = inframe.clone();
    std::vector<float> pr_img = prepareImage(inframe);
    InferRequest::Ptr infer_request = _network.CreateInferRequestPtr();
    Blob::Ptr frameBlob = infer_request->GetBlob(_input_name);
    InferenceEngine::LockedMemory<void> blobMapped = InferenceEngine::as<InferenceEngine::MemoryBlob>(frameBlob)->wmap();
    float* blob_data = blobMapped.as<float*>();

    memcpy(blob_data, pr_img.data(), 3 * INPUT_H * INPUT_W * sizeof(float));

    infer_request->Infer();
    vector<Rect> origin_rect;
    vector<float> origin_rect_cof;
    int i=0;
    vector<Bbox> bboxes;
    for (auto &output : _outputinfo) {
        auto output_name = output.first;
        Blob::Ptr blob = infer_request->GetBlob(output_name);
        LockedMemory<const void> blobMapped = as<MemoryBlob>(blob)->rmap();
        float *output_blob = blobMapped.as<float *>();
        bboxes = postProcess(showImage,output_blob);
        ++i;
    }
    return bboxes;
}
//对模型输出结果进行解码及nms
std::vector<Detector::Bbox> Detector::postProcess(const cv::Mat &src_img,
                              float *output) {
    GenerateReferMatrix();
    std::vector<Detector::Bbox> result;
    float *out = output;
    float ratio = std::max(float(src_img.cols) / float(INPUT_W), float(src_img.rows) / float(INPUT_H));
    cv::Mat result_matrix = cv::Mat(refer_rows, NUM_CLASS + 4, CV_32FC1, out);
    for (int row_num = 0; row_num < refer_rows; row_num++) {
        Detector::Bbox box;
        auto *row = result_matrix.ptr<float>(row_num);
        auto max_pos = std::max_element(row + 4, row + NUM_CLASS + 4);
        box.prob = row[max_pos - row];
        if (box.prob < _cof_threshold)
            continue;
        box.classes = max_pos - row - 4;
        auto *anchor = refer_matrix.ptr<float>(row_num);
        box.x = (anchor[0] - row[0] * anchor[2] + anchor[0] + row[2] * anchor[2]) / 2 * ratio;
        box.y = (anchor[1] - row[1] * anchor[2] + anchor[1] + row[3] * anchor[2]) / 2 * ratio;
        box.w = (row[2] + row[0]) * anchor[2] * ratio;
        box.h = (row[3] + row[1]) * anchor[2] * ratio;
        result.push_back(box);
    }
    NmsDetect(result);
    return result;
}

```

# 0x5. 推理时间展示及预测结果展示

![平均推理时间6ms左右，CPU下实时推理](https://img-blog.csdnimg.cn/20210129230852752.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70#pic_center)

![安全帽检测结果](https://img-blog.csdnimg.cn/20210129231056936.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

至此完成了NanoDet在X86 CPU上的部署，希望有帮助到大家。

-----------------------------------------------------------------------------------------------
欢迎关注GiantPandaCV, 在这里你将看到独家的深度学习分享，坚持原创，每天分享我们学习到的新鲜知识。( • ̀ω•́ )✧

有对文章相关的问题，或者想要加入交流群，欢迎添加BBuf微信：

![二维码](https://img-blog.csdnimg.cn/20200110234905879.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

为了方便读者获取资料以及我们公众号的作者发布一些Github工程的更新，我们成立了一个QQ群，二维码如下，感兴趣可以加入。

![公众号QQ交流群](https://img-blog.csdnimg.cn/20200517190745584.png#pic_center)