#include<cmath>
#include<iostream>
#include<vector>


void relu(float* arr, size_t numElements) {
    for (size_t i = 0; i < numElements; i++)
        arr[i] = (arr[i] > 0) ? arr[i] : 0;
}

void elu(std::vector<float>& arr, float a) {
    for (size_t i = 0; i < arr.size(); i++)
        arr[i] = (arr[i] > 0) ? arr[i] : a * (std::exp(arr[i]) - 1);
}

int main() {
    std::vector<float> data{-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    relu(data.data(), data.size()); // relu call
    for (auto i : data)
        std::cout << i << "\n";
}

