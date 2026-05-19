#include "Mat.h"
#include <utility>
#include <random>
#include <cstring>
Mat::Mat(size_t nrows,size_t ncols)
:m_nrows(nrows),m_ncols(ncols){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0,2);

    for (size_t i = 0; i < nrows; i++)
    {
        std::vector<float> row;
        for (size_t j = 0; j < ncols; j++)
        {
            row.push_back(dist(gen));
        }
        data.push_back(row);
        
    }
    
}

Mat::Mat(size_t nrows,size_t ncols,float init_value)
:m_nrows(nrows),m_ncols(ncols){
    for (size_t i = 0; i < nrows; i++)
    {
        std::vector<float> row;
        for (size_t j = 0; j < ncols; j++)
        {
            row.push_back(init_value);
        }
        data.push_back(row);
        
    }
}

Mat::Mat(std::vector<std::vector<float>>&& data){
    this->data = std::move(data);
    m_nrows = this->data.size();
    int t = this->data.at(0).size();
    for(size_t i{};i<m_nrows;i++){
        if(this->data.at(i).size() != t){
            std::string err = "Matrix of Inconsistent type! "+std::to_string(t)+", "+std::to_string(this->data.at(i).size());
            throw std::runtime_error(err.c_str());
        }
    }
    
    
    m_ncols = this->data.at(0).size();
}

std::string Mat::size(){
        std::string s_ = "("+std::to_string(m_nrows)+","+std::to_string(m_ncols)+")";
        return s_;
}


std::ostream& operator<<(std::ostream& out,const Mat& mat){
    std::cout << "[\n";
    for (auto& x : mat.data)
    {
        out << "[";
        for (size_t j = 0; j < x.size(); j++)
        {
            out << x.at(j) << " ";
        }
        out << "]\n";
    }
    out << "]";

    return out;
}


void Mat::transpose(){
   std::vector<float> data_t;
    
    for (size_t i = 0; i < m_ncols; i++)
    {
        for (size_t j = 0; j < m_nrows; j++)
        {
            
            data_t.push_back(data.at(j).at(i));
            
        }
        
    }
    float new_ncols = m_nrows;
    float new_nrows = m_ncols;
    m_ncols = new_ncols;
    m_nrows = new_nrows;
    data.clear();
    int count{};
    for (size_t i = 0; i < m_nrows; i++)
    {
        std::vector<float> row;
        for (size_t j = 0; j < m_ncols; j++)
        {
            row.push_back(data_t.at(count));
            count++;
        }
        data.push_back(row);
        
    }

    
}

Mat Mat::operator+(const Mat& other){
        if(this->m_nrows == other.m_nrows && this->m_ncols == other.m_ncols){
            std::vector<std::vector<float>> n_data;
            for (size_t i = 0; i < m_nrows; i++)
            {
                std::vector<float> row;
                for (size_t j = 0; j < m_ncols; j++)
                {
                    row.push_back(this->data.at(i).at(j) + other.data.at(i).at(j));
                }
                n_data.push_back(row);
            }
            return Mat(std::move(n_data));
            
        }else{
            throw std::runtime_error("Matrices Sizes do not support addition");
        }
}

Mat Mat::operator%(const Mat& other){
        if(this->m_ncols == other.m_nrows){
            std::vector<std::vector<float>> n_data;
            n_data.reserve(this->m_nrows);
            for (size_t i = 0; i < this->m_nrows; i++)
            {
                std::vector<float> row;
                row.reserve(other.m_ncols);
                for (size_t j = 0; j < other.m_ncols; j++)
                {
                    float sum = 0;
                    for (size_t k = 0; k < other.m_nrows; k++)
                    {
                        sum +=this->data.at(i).at(k)*other.data.at(k).at(j);
                    }
                    row.push_back(sum);
                    
                }
                n_data.push_back(row);
                
            }
            return Mat(std::move(n_data));
        }else{
            
            std::string error = "Matrices don't support matmulti. Sizes: ("+ std::to_string(this->m_nrows)+"," + std::to_string(this->m_ncols)+")\n ("+std::to_string(other.m_nrows)+","+std::to_string(other.m_ncols)+")";
            throw std::runtime_error(error.c_str());
        }
}

Mat Mat::operator-(const Mat& other){
        if(this->m_nrows == other.m_nrows && this->m_ncols == other.m_ncols){
            std::vector<std::vector<float>> n_data;
            for (size_t i = 0; i < m_nrows; i++)
            {
                std::vector<float> row;
                for (size_t j = 0; j < m_ncols; j++)
                {
                    row.push_back(this->data.at(i).at(j) - other.data.at(i).at(j));
                }
                n_data.push_back(row);
            }
            return Mat(std::move(n_data));
            
        }else{
            std::string err = "Matrices Sizes do not support Subtraction A("+std::to_string(this->m_nrows)+","+std::to_string(this->m_ncols)+") B("+std::to_string(other.m_nrows)+","+std::to_string(other.m_ncols)+")";
            throw std::runtime_error(err.c_str());
        }
}

 Mat Mat::operator*(const Mat& other){
        if(this->m_ncols == other.m_ncols && this->m_nrows == other.m_nrows){
            std::vector<std::vector<float>>n_data;
            for (size_t i = 0; i < m_nrows; i++)
            {
                std::vector<float> row;
                for (size_t j = 0; j < m_ncols; j++)
                {
                    row.push_back(this->data.at(i).at(j) * other.data.at(i).at(j));
                }
                n_data.push_back(row);
            }

            return Mat(std::move(n_data));
            
        }else{
            std::string err = "Matrices sizes doesnt support this operation. Sizes A("+std::to_string(this->m_nrows)+","+std::to_string(this->m_ncols)+") B("+std::to_string(other.m_nrows)+","+std::to_string(other.m_ncols)+")";
            throw std::runtime_error(err.c_str());
        }
    }

Mat Mat::operator/(const Mat& other){
        if(this->m_nrows == other.m_nrows){
            std::vector<std::vector<float>> n_data;
            for (size_t i = 0; i < this->m_nrows; i++)
            {
                std::vector<float> row;
                for (size_t j = 0; j < other.m_ncols; j++)
                {
                    row.push_back(this->data.at(i).at(0) * other.data.at(i).at(j));
                }
                n_data.push_back(row);
            }
            
            return Mat(std::move(n_data));
        }else{
            std::string err =  "Incompatible Sizes! A("+std::to_string(this->m_nrows)+","+std::to_string(this->m_ncols)+") B("+std::to_string(other.m_nrows)+","+std::to_string(other.m_ncols)+")";
            throw std::runtime_error(err.c_str());
        }
    }

Mat Mat::operator*(const float scalar){
        std::vector<std::vector<float>> n_data;
        for (size_t i = 0; i < m_nrows; i++)
        {
            std::vector<float> row;
            for (size_t j = 0; j < m_ncols; j++)
            {
                row.push_back(this->data.at(i).at(j)*scalar);
            }
            n_data.push_back(row);
        }
        return Mat(std::move(n_data));
}

std::vector<float> Mat::to_vector_float(){
    if (m_nrows == 1)
    {
        std::vector<float> vec_float;
        for (size_t i = 0; i < m_ncols; i++)
        {
            
            vec_float.push_back(data.at(0).at(i));
        }
        return vec_float;
    }else{
        std::string err = "Matrix must be (1 x n) dimensional. Current Dimensions: ("+std::to_string(this->m_nrows)+","+std::to_string(this->m_ncols)+")";
        throw std::runtime_error(err.c_str());
    }
    
}

Mat Mat::operator+(const float scalar){
        std::vector<std::vector<float>> n_data;
        for (size_t i = 0; i < m_nrows; i++)
        {
            std::vector<float> row;
            for (size_t j = 0; j < m_ncols; j++)
            {
                row.push_back(data.at(i).at(j)+scalar);
            }
            n_data.push_back(row);
        }
        return Mat(std::move(n_data));
    }

Mat Mat::exp(float exp){
        std::vector<std::vector<float>> n_data = data;
        for (size_t i = 0; i < m_nrows; i++)
        {
            for (size_t j = 0; j < m_ncols; j++)
            {
                n_data.at(i).at(j) = std::pow(n_data.at(i).at(j),exp);
            }
            
        }

        return Mat(std::move(n_data));
        
}

Mat Mat::divide(const Mat& a,const Mat& b){
        if(a.m_ncols == b.m_ncols && a.m_nrows == b.m_nrows){
        std::vector<std::vector<float>> n_data;
        for (size_t i = 0; i < a.m_nrows; i++)
        {
            std::vector<float> row;
            for (size_t j = 0; j < a.m_ncols; j++)
            {
                row.push_back(a.data.at(i).at(j) / b.data.at(i).at(j));
            }
            n_data.push_back(row);
        }
        return Mat(std::move(n_data));
    }else{
        std::string err = "Matrices dimensions do not support division. A("+ std::to_string(a.m_nrows)+"," + std::to_string(a.m_ncols)+")\n ("+std::to_string(b.m_nrows)+","+std::to_string(b.m_ncols)+")";
        throw std::runtime_error(err.c_str());
    }
        
    }