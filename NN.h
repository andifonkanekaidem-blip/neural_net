#include "NN_Layer.h"
#include <vector>
#include <utility>
#include <map>
#include "Mat.h"
#include <cmath>
using _data = std::vector<float>;

class Mat;
class Adam{
    
    std::vector<Mat> momentum_w;
    std::vector<Mat> variance_w;
    std::vector<Mat> momentum_b;
    std::vector<Mat> variance_b;
    size_t idx{};
    size_t t{1};
    size_t size;
     float B1 = 0.9;
     float B2 = 0.999;
    public:
    Adam(size_t model_size,std::vector<std::pair<size_t,size_t>> mw_s,std::vector<std::pair<size_t,size_t>> mb_s);
    void step(Mat* weights,Mat* biases,Mat grad_w,Mat grad_b,size_t index);
};

class NN_Layer;
class Mat;
class Adam;
class NN
{
private:
    std::vector<NN_Layer> data_fp;
    std::vector<NN_Layer> data_bp;
    std::vector<float> inputs;
    std::vector<float> output;
    std::vector<std::pair<size_t,size_t>> w_s;
    std::vector<std::pair<size_t,size_t>> b_s;
    
public:
    NN(std::vector<NN_Layer>&& nn_layers,std::vector<float>&& input);
    std::vector<float> forward_pass();
    float lr = 0.001;
    float init_loss{};
    float curr_loss{100};
    void back_propagate(_data outputs,_data&& target);
    inline void print_output(){
        for (size_t i = 0; i < output.size() ;i++)
        {
            std::cout << output.at(i) << " ";
        }
    }
    static float MSELoss(std::vector<float>,std::vector<float>);
    static _data MSELoss_deriv(std::vector<float> out_,_data target);
    Adam optimizer = Adam(data_fp.size(),w_s,b_s);
    static float BCSELoss(std::vector<float>,std::vector<float>);
};

