#include "Neuron.h"
#include <utility>
#include <random>
#include <cmath>
#include <iostream>
Neuron::Neuron(int weight_number)
{
    m_learn_rate = 0.0001;
    m_bias = 0.1;
    float std_dev = std::sqrt(2.0/weight_number);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0,std_dev);
    for (size_t i = 0; i < weight_number; i++)
    {
        m_weights.push_back(dist(gen)); 
    }
    
}

float Neuron::pass(float x,int index){
    float init_val = m_weights.at(index) * x;

    return init_val;
}

void Neuron::edit(std::vector<float> loss,float delta_b){
    losses = std::move(loss);
    for (size_t i = 0; i < losses.size(); i++)
    {
        
        m_weights.at(i) -=  (m_learn_rate*losses.at(i));
    }
    
    m_bias = m_bias - (m_learn_rate*delta_b);

}

void Neuron::print_weigths(){
    for (auto& i : m_weights)
    {
        std::cout << i << " ";
    }
        
}

std::vector<float> Neuron::get_weights(){
    return m_weights;
}

