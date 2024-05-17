
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>


__global__ void ColorToGrayScaleConversion(unsigned char *Pout, unsigned char *Pin, int width, int height) {
	int col = threadIdx.x + blockDim.x * blockIdx.x;
	int row = threadIdx.y + blockDim.y * blockIdx.y;	

	if (col < width && row < height) {
		int grayOffset = row * width + col;		

		int rgbOffset = grayOffset * CHANNELS;
		unsigned char r = Pin[rgbOffset]; //RED VALUE
		unsigned char g = Pin[rgbOffset + 1]; //GREEN VALUE
		unsigned char b = Pin[rgbOffset + 2]; //BLUE VALUE

		Pout[grayOffset] = 0.21f * r + 0.71f * g + 0.07f * b;
	}

}

int main() {

}
