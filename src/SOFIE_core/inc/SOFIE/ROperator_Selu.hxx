#ifndef SOFIE_ROPERATOR_SELU
#define SOFIE_ROPERATOR_SELU

#include "SOFIE/SOFIE_common.hxx"
#include "SOFIE/ROperator.hxx"
#include "SOFIE/RModel.hxx"

#include <sstream>

namespace SOFIE{

template <typename T>
class ROperator_Selu final : public ROperator
{

private:

   std::string fNX;
   std::string fNY;
   std::vector<size_t> fShape;

public:
   ROperator_Selu(){}
   ROperator_Selu(std::string nameX, std::string nameY):
      fNX(UTILITY::Clean_name(nameX)), fNY(UTILITY::Clean_name(nameY)){
         fInputTensorNames = { fNX };
         fOutputTensorNames = { fNY };
      }

   std::vector<ETensorType> TypeInference(std::vector<ETensorType> input) override {
      return input;
   }

   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input) override {
      auto ret = input; //suggest copy to compiler
      return ret;
   }

   void Initialize(RModel& model) override {
      if (model.CheckIfTensorAlreadyExist(fNX) == false){   //input must be a graph input, or already initialized intermediate tensor
         throw std::runtime_error("TMVA SOFIE Selu Op Input Tensor is not found in model");
      }
      fShape = model.GetTensorShape(fNX);
      model.AddIntermediateTensor(fNY, model.GetTensorType(fNX), fShape);
   }


   std::string Generate(std::string OpName) override {
      OpName = "op_" + OpName;
      if (fShape.empty()){
         throw std::runtime_error("TMVA SOFIE Operator Selu called to Generate without being initialized first");
      }
      std::stringstream out;
      int length = 1;
      for(auto& i: fShape){
         length *= i;
      }
      out << "\t" << "for (int id = 0; id < " << length << " ; id++){\n";
      out << "\t\t" << "tensor_" << fNY << "[id] = 1.0507009873554804934193349852946 * (std::max(float(0.0), tensor_"  << fNX << "[id]) + std::min(0.0, 1.6732632423543772848170429916717 * (std::exp(" << "tensor_" << fNX << "[id]" <<")-1)));\n";
      out << "\t}\n";
      return out.str();
   }
    std::string Generate_GPU_Kernel_ALPAKA(std::string /*opName*/) override {
        std::string op;
        op = "\n//------ SELU_KERNEL_ALPAKA\n";
        op += "struct SeluKernel {\n";
        op += SP + "template<typename TAcc, typename T>\n";
        op += SP + "ALPAKA_FN_ACC void operator()(TAcc const & acc, T const* __restrict__ data, T* __restrict__ out, std::size_t numElements) const {\n";
        op += SP + SP + SP + "auto idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];\n";
        op += SP + SP + SP + "if (idx < numElements) {\n";
        op += SP + SP + SP + SP + "constexpr T lambda = T(1.0507009873554804934193349852946);\n";
        op += SP + SP + SP + SP + "constexpr T alpha = T(1.6732632423543772848170429916717);\n";
        op += SP + SP + SP + SP + "out[idx] = data[idx] > static_cast<T>(0) ? lambda * data[idx] : lambda * alpha * (std::exp(data[idx]) - static_cast<T>(1));\n";
        op += SP + SP + SP + "}\n";
        op += SP + "}\n";
        op += "};\n";
        return op;
    }

    std::string Generate_GPU_Kernel_Definitions_ALPAKA(std::string /*opName*/) override {
        return "SeluKernel seluKernel;\n";
    }

    std::string Generate_GPU_ALPAKA(std::string OpName) override {
        OpName = "op_" + OpName;
        if (fShape.empty()) {
            throw std::runtime_error("TMVA SOFIE Operator Selu called to Generate without being initialized first");
        }
        std::stringstream out;
        auto length = ConvertShapeToLength(fShape);
        out << "\n//------ SELU_GPU_ALPAKA\n";
        out << SP << "auto const elementsPerThread_" << fNX << " = Vec::all(static_cast<Idx>(1));\n";
        out << SP << "auto const elementsPerGrid_" << fNX << " = Vec::all(Idx{" << length << "});\n";
        out << SP << "alpaka::KernelCfg<Acc> const kernelCfg_" << fNX << " = {elementsPerGrid_" << fNX << ", elementsPerThread_" << fNX << "};\n";
        out << SP << "auto const workDiv_" << fNX << " = alpaka::getValidWorkDiv(kernelCfg_" << fNX << ", devAcc, seluKernel, alpaka::getPtrNative(deviceBuf_" << fNX
            << "), alpaka::getPtrNative(deviceBuf_" << fNY << "), static_cast<Idx>(" << length << "));\n";
        out << SP << "alpaka::exec<Acc>(queue, workDiv_" << fNX
            << ", seluKernel, alpaka::getPtrNative(deviceBuf_" << fNX
            << "), alpaka::getPtrNative(deviceBuf_" << fNY << "), static_cast<Idx>(" << length << "));\n";
        return out.str();
    }

   std::vector<std::string> GetStdLibs() override { return { std::string("cmath") };}
};

}//SOFIE


#endif //SOFIE_ROPERATOR_SELU
