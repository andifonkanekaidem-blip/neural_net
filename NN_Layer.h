#pragma once 
#include <vector>
#include <iostream>
#include "Mat.h"
#include "Neuron.h"
class Neuron;
class Mat;
class NN_Layer{
    std::vector<Neuron> neurons;
    std::vector<float> output;
    std::vector<float> deriv_act;
    
    
public:

    NN_Layer(size_t num_in,size_t num_out);
    NN_Layer(size_t num_in,size_t num_out,std::vector<float>&& inp);
    ~NN_Layer();

    Mat* weights_mat{nullptr};
    Mat* input_mat{nullptr};
    Mat* bias_mat{nullptr};
    size_t m_num_in;
    size_t m_num_out;
    
    void print_output();
    Mat calculate();
    std::vector<float>& get_outputs();
    void set_inputs(std::vector<float>&&);
    inline std::vector<float>&get_deriv(){
        return deriv_act;
    }

    // Activations

    void ReLU(Mat&);
    void leaky_ReLU(Mat& mat);
    
};