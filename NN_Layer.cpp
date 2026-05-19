#include "NN_Layer.h"
#include "Neuron.h"
#include <iostream>
#include <utility>
#include <chrono>
#include <thread>
#include <cmath>

NN_Layer::NN_Layer(size_t num_in,size_t num_out)
    :m_num_in(num_in),m_num_out(num_out){
        std::vector<std::vector<float>> data;
        std::vector<std::vector<float>> d_bias;
        for (size_t i = 0; i < num_out; i++)
        {
            Neuron n = Neuron(num_in);
            data.push_back(n.get_weights());
            
            d_bias.push_back(std::vector<float>{n.get_bias()});
            neurons.push_back(n);   
        }
        weights_mat = new Mat(std::move(data));
        bias_mat = new Mat(std::move(d_bias));

        
    }
NN_Layer::NN_Layer(size_t num_in,size_t num_out,std::vector<float>&& inp)
:m_num_in(num_in),m_num_out(num_out){
    if(inp.size() != 0 && inp.size() != num_in){
        throw std::runtime_error("Input Given doesnt match declared size");
    }
    std::vector<float> input = std::move(inp);
    input_mat = new Mat(std::vector<std::vector<float>>{input});
    
    input_mat->transpose();
    std::vector<std::vector<float>> data;
    std::vector<std::vector<float>> d_bias;
    for (size_t i = 0; i < num_out; i++)
    {
        Neuron n = Neuron(num_in);
        data.push_back(n.get_weights());
        d_bias.push_back(std::vector<float>{n.get_bias()});
        neurons.push_back(n);
        
    }
    weights_mat = new Mat(std::move(data));
    bias_mat = new Mat(std::move(d_bias));

}

void NN_Layer::set_inputs(std::vector<float>&& inp){
    std::vector<float> input = inp;
    std::vector<std::vector<float>> data;
    data.push_back(input);


    input_mat = new Mat(std::move(data));

    input_mat->transpose();
}

NN_Layer::~NN_Layer(){
    delete weights_mat;
    delete input_mat;
    delete bias_mat;
}

void NN_Layer::print_output(){
    for (size_t i = 0; i < output.size(); i++)
    {
        std::cout << output.at(i) << " ";
    }
}

Mat NN_Layer::calculate(){
    Mat out = (*weights_mat%*input_mat) + *bias_mat;
    out.transpose();
    leaky_ReLU(out);
    output = out.to_vector_float();
    return out;
}

std::vector<float>& NN_Layer::get_outputs(){
    return output;
}

void NN_Layer::ReLU(Mat& mat){
    if(mat.get_data().size() == 1){
    for (size_t i = 0; i < mat.get_data().at(0).size(); i++)
    {
        if(mat.get_data().at(0).at(i) < 0){
            mat.get_data().at(0).at(i) = 0;
            deriv_act.push_back(0);
            continue;
        }
        deriv_act.push_back(1);
    }
}else{
    throw std::runtime_error("Make matrix a row matrix before applying ReLU");
}
    
}



void NN_Layer::leaky_ReLU(Mat& mat){
    deriv_act.clear();
    if(mat.get_data().size() == 1){
    for (size_t i = 0; i < mat.get_data().at(0).size(); i++)
    {
        if(mat.get_data().at(0).at(i) <= 0){
            float s = std::abs(mat.get_data().at(0).at(i));
            mat.get_data().at(0).at(i) = s*0.01F;
            deriv_act.push_back(0.01);
            continue;
        }
        deriv_act.push_back(1);
    }
}else{
    throw std::runtime_error("Make matrix a row matrix before applying ReLU");
}
    
}

