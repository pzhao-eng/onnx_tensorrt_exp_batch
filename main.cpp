/*************************************************************************
      > File Name: main.cpp
      > Author: zhaopeng
      > Mail: zhaopeng_chem@163.com
      > Created Time: Wed 13 Apr 2022 08:20:21 PM CST
 ************************************************************************/

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <unistd.h>
#include <vector>
#include <cmath>
#include <numeric>
#include <cassert>
#include <stdexcept>
#include "print_array.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

using namespace nvinfer1;
using namespace std;

#define ck(call) check(call, __LINE__, __FILE__)
int inputChannel = 0;
int inputSize_H = 0;
int inputSize_W = 0;

const std::string onnxFile {"./model.onnx"};
const std::string trtFile {"./model.plan"};

inline bool check(cudaError_t e, int iLine, const char *szFile)
{
    if (e != cudaSuccess)
    {
        std::cout << "CUDA runtime API error " << cudaGetErrorName(e) << " at line " << iLine << " in file " << szFile << std::endl;
        return false;
    }
    return true;
}

class Logger : public ILogger
{
public:
    Severity reportableSeverity;

    Logger(Severity severity = Severity::kINFO):
        reportableSeverity(severity) {}

    void log(Severity severity, const char *msg) override
    {
        if (severity > reportableSeverity)
        {
            return;
        }
        switch (severity)
        {
        case Severity::kINTERNAL_ERROR:
            std::cerr << "INTERNAL_ERROR: ";
            break;
        case Severity::kERROR:
            std::cerr << "ERROR: ";
            break;
        case Severity::kWARNING:
            std::cerr << "WARNING: ";
            break;
        case Severity::kINFO:
            std::cerr << "INFO: ";
            break;
        default:
            std::cerr << "UNKNOWN: ";
            break;
        }
        std::cerr << msg << std::endl;
    }
};

static Logger gLogger(ILogger::Severity::kERROR);

void readFiles(const vector<string> &fileNames, uint8_t *input_data_host)
{
    int offset = 0; 
    for (const string& fileName : fileNames)
    {
        std::ifstream infile(fileName, std::ifstream::binary);
        assert(infile.is_open() && "Attempting to read from a file that is not open.");

        std::string magic;
        int h, w, max;
        infile >> magic >> h >> w >> max;

        infile.seekg(1, infile.cur);
        size_t vol = 1 * 1 * h * w;
        uint8_t *fileData = &(input_data_host[offset]);
        infile.read(reinterpret_cast<char*>(fileData), vol);
        offset += (h * w);
        cout << "Input:\n";
        for (size_t i = 0; i < vol; i++)
        {
            cout << (" .:-=+*#%@"[fileData[i] / 26]) << (((i + 1) % w) ? "" : "\n");
        }
        cout << std::endl; 
    }
}

unsigned int getElementSize(nvinfer1::DataType t)
{
    switch (t)
    {
    case nvinfer1::DataType::kINT32: return 4;
    case nvinfer1::DataType::kFLOAT: return 4;
    case nvinfer1::DataType::kHALF: return 2;
    case nvinfer1::DataType::kBOOL:
    case nvinfer1::DataType::kINT8: return 1;
    }
    throw std::runtime_error("Invalid DataType.");
    return 0;
}

int64_t volume(const nvinfer1::Dims& d)
{
    return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
}

void print_dimensione_val(const Dims &val)
{
    for (int i = 0; i < val.nbDims; i++){
        cout<<val.d[i]<<" ";
    }
    cout<<endl;
}

int main()
{
    IBuilder *builder = createInferBuilder(gLogger);
    const uint32_t explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    INetworkDefinition *network = builder->createNetworkV2(explicitBatch);
    auto config = builder->createBuilderConfig();
    config->setMaxWorkspaceSize(5<<30);
    config->setFlag(BuilderFlag::kFP16);

    auto parser = nvonnxparser::createParser(*network, gLogger);
    auto parsed = parser->parseFromFile(onnxFile.c_str(), static_cast<int>(gLogger.reportableSeverity));
    if (!parsed){
        cout<<"parser onnx file Error"<<endl;
        return 1;
    }

    auto inputTensor = network->getInput(0);
    auto profile = builder->createOptimizationProfile();
    profile->setDimensions(inputTensor->getName(), OptProfileSelector::kMIN, Dims4{1, 1, 28, 28});
    profile->setDimensions(inputTensor->getName(), OptProfileSelector::kOPT, Dims4{4, 1, 28, 28});
    profile->setDimensions(inputTensor->getName(), OptProfileSelector::kMAX, Dims4{16, 1, 28, 28});
    config->addOptimizationProfile(profile);
    
    network->unmarkOutput(*network->getOutput(0));
    auto engine = builder->buildEngineWithConfig(*network, *config);
    network->destroy();
    builder->destroy();

    vector<string> fileNames;
    fileNames.push_back("./0.jpg");
    fileNames.push_back("./1.jpg");
    fileNames.push_back("./2.jpg");
    fileNames.push_back("./3.jpg");
    fileNames.push_back("./4.jpg");
    fileNames.push_back("./5.jpg");
    fileNames.push_back("./6.jpg");
    fileNames.push_back("./7.jpg");
    fileNames.push_back("./8.png");
    fileNames.push_back("./9.jpg");
    int batchSize = fileNames.size();
    vector<unsigned char *>input_data; 
    for (int i = 0; i < batchSize; i++){
         unsigned char *data = stbi_load(fileNames[i].c_str(), &inputSize_H, &inputSize_W, &inputChannel, 0);
         input_data.emplace_back(data);
    }
    cout<<"image size = "<<inputChannel<<" "<<inputSize_H<<" "<<inputSize_W<<endl;

    auto context = engine->createExecutionContext();
    cudaStream_t stream;
    ck(cudaStreamCreate(&stream));
    context->setOptimizationProfile(0);
    //context->setOptimizationProfileAsync(0, stream);
    Dims4 inputDims2{batchSize, 1, inputSize_H, inputSize_W};
    cout<<"input dims of engine = ";
    print_dimensione_val(engine->getBindingDimensions(0));
    context->setBindingDimensions(0, inputDims2);

    // alloc device and host memory
    const auto dataTypeInput = engine->getBindingDataType(0);
    const auto dataTypeOutput = engine->getBindingDataType(1);
    Dims inputDims = context->getBindingDimensions(0);
    Dims outputDims = context->getBindingDimensions(1);

    vector<void *> buffer{nullptr, nullptr};
    int input_size = volume(inputDims); 
    int output_size = volume(outputDims); 
    cout<<"input_size = "<<inputDims.nbDims<<" "<<volume(inputDims)<<endl;
    cout<<"output_size = "<<outputDims.nbDims<<" "<<volume(outputDims)<<endl;
    
    ck(cudaMalloc(&buffer[0], input_size * getElementSize(dataTypeInput)));
    ck(cudaMalloc(&buffer[1], output_size * getElementSize(dataTypeOutput)));
    float *input_data_host = nullptr;
    int *out_data_host = nullptr;
    ck(cudaMallocHost((void **)&input_data_host, input_size * getElementSize(dataTypeInput)));
    ck(cudaMallocHost((void **)&out_data_host, output_size * getElementSize(dataTypeOutput)));
    for (int i = 0; i < batchSize; i++){
        for (int j = 0; j < inputSize_H * inputSize_W; j++){
            input_data_host[i * inputSize_H * inputSize_W + j] = static_cast<float>(input_data[i][j]);
        }
    }

    ck(cudaMemcpyAsync(buffer[0], input_data_host, input_size * getElementSize(dataTypeInput), cudaMemcpyHostToDevice, stream));
    context->enqueueV2(buffer.data(), stream, nullptr);
    ck(cudaMemcpyAsync(out_data_host, buffer[1], output_size * getElementSize(dataTypeOutput), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    cout<<"output:"<<endl;
    for (int i = 0; i < batchSize; i++){
        cout<<out_data_host[i]<<endl;
    }
    cout<<"End!"<<endl;

    context->destroy();
    engine->destroy();
    ck(cudaFree(buffer[0]));
    ck(cudaFree(buffer[1]));
    ck(cudaFreeHost(input_data_host));
    ck(cudaFreeHost(out_data_host));
    return 0;
}
