#include "NN.h"
#include <cmath>

NN::NN(std::vector<NN_Layer>&& nn_layers,std::vector<float>&& input){
        data_fp = std::move(nn_layers);
        inputs = std::move(input);
        for (size_t i = 0; i < data_fp.size(); i++)
        {
            w_s.push_back(data_fp.at(i).weights_mat->size_p());
            b_s.push_back(data_fp.at(i).bias_mat->size_p());
        }
        optimizer = Adam(data_fp.size(),w_s,b_s);
        
}

Adam::Adam(size_t model_size,std::vector<std::pair<size_t,size_t>> mw_s,std::vector<std::pair<size_t,size_t>> mb_s):size(model_size){

        for (auto &i : mw_s)
        {
            momentum_w.push_back(Mat(i.first,i.second,0.0F));
            variance_w.push_back(Mat(i.first,i.second,0.0F));

        }
        for (auto &i : mb_s)
        {
            momentum_b.push_back(Mat(i.first,i.second,0.0F));
            variance_b.push_back(Mat(i.first,i.second,0.0F));
        }
        
        
    
}

std::vector<float> NN::forward_pass(){

        std::vector<float> inter_inputs = inputs;
        for (size_t i = 0; i < data_fp.size(); i++)
        {
            data_fp.at(i).set_inputs(std::move(inter_inputs));
            inter_inputs = data_fp.at(i).calculate().to_vector_float();
        }
        output = inter_inputs;
        return output;
}


float NN::MSELoss(std::vector<float> predicted,std::vector<float> target){
    if (predicted.size() == target.size())
    {
        size_t n = predicted.size();
        float sum = 0;
        for (size_t i = 0; i < n; i++)
        {
            sum += std::pow((target.at(i) - predicted.at(i)),2);
        }

        return sum/n;
    }else{
        std::string err = "Loss Function Error! Size mismatch. Predicted size: " + std::to_string(predicted.size())+", Target Size: "+std::to_string(target.size());

        throw std::runtime_error(err.c_str());
    }
    
}

float NN::BCSELoss(std::vector<float>predicted,std::vector<float>target){
    if (predicted.size() == target.size() && predicted.size() == 1)
    {
        float y = target.at(0), _y = predicted.at(0);
        float L = -(y*std::log10(_y) + (1 - y)*std::log10(1 - _y));
        return L;
    }else{
        throw std::runtime_error("Loss Function Error! Size mismatch");
    }
}

_data NN::MSELoss_deriv(std::vector<float> out_,_data target){
        _data deirvs;
        for (size_t i = 0; i < out_.size(); i++)
        {
            deirvs.push_back(((float)1/out_.size())*(out_.at(i) - target.at(i)));
        }

        return deirvs;
        
}

void NN::back_propagate(_data outputs,_data&& target)
    {

        _data t=std::move(target);
        size_t size = data_fp.size()-1;
        _data prelim;
        if(init_loss == 0){
            init_loss =  MSELoss(outputs,t);
        }else{
            curr_loss = MSELoss(outputs,t);
        }
        _data derivs_init = MSELoss_deriv(outputs,t);

            for (size_t j = 0; j < outputs.size(); j++)
            {
                prelim.push_back(data_fp.at(size).get_deriv().at(j)*derivs_init.at(j));
            }
            
            
            Mat m1(std::vector<_data>{prelim});
            m1.transpose();


            data_fp.at(size).input_mat->transpose();
            _data o_w = data_fp.at(size).input_mat->to_vector_float();
            std::vector<std::vector<float>> d_w;
            for (size_t j = 0; j < outputs.size(); j++)
            {
                d_w.push_back(o_w);
            }
            
            Mat grad_w = m1 / Mat(std::move(d_w));
            
            
            optimizer.step(data_fp.at(size).weights_mat,data_fp.at(size).bias_mat,grad_w,m1,size);
            
            Mat prev_err_signal = m1;
            if(size == 0){
                return;
            }
            for (size_t i = size-1; i >= 0; i--)
            {
                data_fp.at(i+1).weights_mat->transpose();
                _data inter_deriv = data_fp.at(i).get_deriv();
                Mat deriv_s(std::vector<_data>{inter_deriv});
                deriv_s.transpose();
                Mat err_signal_n = (*data_fp.at(i+1).weights_mat %  prev_err_signal);
                err_signal_n = err_signal_n *deriv_s;
                data_fp.at(i).input_mat->transpose();
                _data intermediary = data_fp.at(i).input_mat->to_vector_float();
                
                std::vector<_data> n_inter;
                
                for (size_t j = 0; j < data_fp.at(i).m_num_out; j++)
                {
                    n_inter.push_back(intermediary);
                }
                
                Mat grad_n = err_signal_n / Mat(std::move(n_inter));
                
                
                optimizer.step(data_fp.at(i).weights_mat,data_fp.at(i).bias_mat,grad_n,err_signal_n,i);
                
                
                prev_err_signal = err_signal_n;
                data_fp.at(i+1).weights_mat->transpose();
                data_fp.at(i).input_mat->transpose();
                
                if (i==0)
                {
                    break;
                }
                
            }
}



void Adam::step(Mat* weights,Mat* biases,Mat grad_w,Mat grad_b,size_t index){
        Mat m_w = momentum_w.at(index)*B1 + grad_w*(1-B1);
        Mat v_w = variance_w.at(index)*B2 + (grad_w*grad_w)*(1-B2);

        momentum_w.at(index) = m_w;
        variance_w.at(index) = v_w;
        Mat mb_w = m_w*((float)1/(1-std::pow(B1,t)));
        Mat vb_w = v_w*((float)1/(1-std::pow(B2,t)));
        *weights = *weights -  Mat::divide(mb_w,(vb_w.exp(0.5)+0.00000001))*0.001;
        Mat m_b = momentum_b.at(index)*B1 + grad_b*(1-B1);
        Mat v_b = variance_b.at(index)*B2 + (grad_b*grad_b)*(1-B2);

        momentum_b.at(index) = m_b;
        variance_b.at(index) = v_b;
        Mat mb_b = m_b*((float)1/(1-std::pow(B1,t)));
        Mat vb_b = v_b*((float)1/(1-std::pow(B2,t)));
        *biases = *biases -  Mat::divide(mb_b,(vb_b.exp(0.5)+0.00000001))*0.001;
        t++;
    }
