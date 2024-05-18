
#include <raylib.h>
#include <iostream>
#include "cuda_runtime.h"

const int WIDTH = 800;
const int HEIGHT = 600;
const int CHANNELS = 4; //RGBA


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

	InitWindow(WIDTH, HEIGHT, "Image To Gray");

	Image goku = LoadImage("C:/Users/ochoa/Desktop/Programing/Cuda/Learning/CudaColorToGrayScale/goku.png");

	if (goku.data == NULL) {
		std::cerr << "Failed to load image" << std::endl;
		CloseWindow();
		return -1;
	}

	// Ensure the image is in the correct format
	ImageFormat(&goku, PIXELFORMAT_UNCOMPRESSED_R8G8B8A8);

	// Get image data
	unsigned char* imgData = (unsigned char*)goku.data;
	int imgWidth = goku.width;
	int imgHeight = goku.height;
	int imgSize = imgWidth * imgHeight * CHANNELS * sizeof(unsigned char);

	// Allocate memory on the device
	unsigned char* d_Pin, * d_Pout;

	cudaMalloc(&d_Pin, imgSize);
	cudaMalloc(&d_Pout, imgWidth * imgHeight * sizeof(unsigned char));
	
	cudaMemcpy(d_Pin, imgData, imgSize, cudaMemcpyHostToDevice);

	dim3 blockSize(16, 16, 1);
	dim3 gridSize((imgWidth + blockSize.x - 1) / blockSize.x, (imgHeight + blockSize.y - 1) / blockSize.y, 1);

	ColorToGrayScaleConversion<<<gridSize, blockSize>>>(d_Pout, d_Pin, imgWidth, imgHeight);

	// Allocate memory for the result on the host
	unsigned char* grayImgData = new unsigned char[imgWidth * imgHeight];

	cudaMemcpy(grayImgData, d_Pout, imgWidth * imgHeight * sizeof(unsigned char), cudaMemcpyDeviceToHost);

	// Create a new image for the grayscale result
	Image grayImg = {
		grayImgData,
		imgWidth,
		imgHeight,
		1,
		PIXELFORMAT_UNCOMPRESSED_GRAYSCALE
	};

	// Convert the image to a texture
	Texture2D texture = LoadTextureFromImage(grayImg);

	// Unload the original image and device memory
	UnloadImage(goku);
	cudaFree(d_Pin);
	cudaFree(d_Pout);


	// Calculate the position to center the image
	int imageX = (WIDTH - texture.width) / 2;
	int imageY = (HEIGHT - texture.height) / 2;

	while (!WindowShouldClose()) {


		BeginDrawing();
			ClearBackground(WHITE);
			DrawTexture(texture, imageX, imageY, WHITE);

		EndDrawing();


	}

	// Unload the texture
	UnloadTexture(texture);

	// Clean up the grayscale image data
	delete[] grayImgData;

	CloseWindow();



	
}
