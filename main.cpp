#include <iostream>
#include "Neuron.h"
#include "NN.h"
#include "NN_Layer.h"
#include "Mat.h"
#include <utility>
#include <numeric>

using _data = std::vector<float>;
int main(){ 
    // Setting Up NN and its layers
    std::vector<NN_Layer> layers;
    layers.reserve(4);
    layers.emplace_back(3, 32);
    layers.emplace_back(32, 32);
    layers.emplace_back(32, 32);
    layers.emplace_back(32, 2);

    
    _data inp{0.3996,0.5994,9.99E-4};
   
    NN nn(std::move(layers),std::move(inp));
    for (size_t i = 0; i < 500; i++)
    {
        _data sth = nn.forward_pass();
        nn.back_propagate(sth,{0.8695,0.1304});
    }
    
    std::cout <<"Initial Loss :" << nn.init_loss << " Final loss: "<< nn.curr_loss << std::endl;

    nn.print_output();
        

    return 0;
}