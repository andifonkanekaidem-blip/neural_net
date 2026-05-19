#pragma once
#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include "NN_Layer.h"
class Mat
{
private:
    std::vector<std::vector<float>> data{};
    size_t m_nrows;
    size_t m_ncols;
public:
    Mat(size_t nrows,size_t ncols);
    Mat(size_t nrows,size_t ncols,float init_value);
    Mat(std::vector<std::vector<float>>&& data);
    friend std::ostream& operator<<(std::ostream& out,const Mat& mat);
    inline std::vector<std::vector<float>>& get_data(){return data;};
    static Mat divide(const Mat& a,const Mat& b);
    Mat operator+(const Mat& other);
    Mat operator+(const float scalar);
    Mat operator-(const Mat& other);
    Mat operator%(const Mat& other);
    Mat operator*(const Mat& other);
    Mat operator*(const float scalar);
    Mat exp(float exp);
    std::vector<float> to_vector_float();
    Mat operator/(const Mat& other);
    inline std::string size();
    inline std::pair<size_t,size_t> size_p(){
        return {this->m_nrows,this->m_ncols};
    }
    void transpose();
    
};




