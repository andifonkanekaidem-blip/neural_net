#pragma once
#include <vector>
class Neuron
{
private:
    std::vector<float> m_weights;
    float m_bias;
    float m_learn_rate;
    std::vector<float> losses;
public:
    Neuron(int weight_number);
    ~Neuron()=default;

    float pass(float x,int index);
    void edit(std::vector<float> loss,float delta_b);
    inline float get_bias(){return m_bias;}
    void print_weigths();
    std::vector<float> get_weights();
    
};

