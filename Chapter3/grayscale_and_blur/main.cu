// Define and include order is important !
#include <stdio.h>
#include <stdlib.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define BLOCK_X 32
#define BLOCK_Y 32

__global__ void color_to_grayscale_kernel(
    unsigned char *Pin,
    unsigned char *Pout,
    int width,
    int height,
    int channels)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < height && col < width)
    {
        // 1D offset of the image for in grayscale
        int grayOffset = row * width + col;

        // Update the pixel offset with channels since there is
        // channels more pixel in the rgb image
        int rgbOffset = grayOffset * channels;

        // Get R, G, B
        unsigned char r = Pin[rgbOffset];
        unsigned char g = Pin[rgbOffset + 1];
        unsigned char b = Pin[rgbOffset + 2];

        // Compute grayscale value
        Pout[grayOffset] = 0.21 * r + 0.72 * g + 0.07 * b;
    }
}

__global__ void blur_image_kernel(
    unsigned char *Pin,
    unsigned char *Pout,
    int width,
    int height,
    int blur_size)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < height && col < width)
    {
        int pixVal = 0;
        int pixels = 0;

        // Aggregate value of all neighboring pixels
        for (int blurRow = -blur_size; blurRow < blur_size + 1; ++blurRow)
        {
            for (int blurCol = -blur_size; blurCol < blur_size + 1; ++blurCol)
            {
                int curRow = row + blurRow;
                int curCol = col + blurCol;

                // Verify we have a correct pixel, this disallow edge pixels
                if (curRow >= 0 && curRow < height && curCol >= 0 && curCol < width)
                {
                    ++pixels;
                    pixVal += Pin[curRow * width + curCol];
                }
            }
        }

        // Compute mean and update pixel
        Pout[row * width + col] = (unsigned char)(pixVal / pixels);
    }
}

void color_to_grayscale(
    unsigned char *Pin,
    unsigned char *Pout,
    int width,
    int height,
    int channels)
{
    // Allocate VRAM in device
    unsigned char *Pin_d, *Pout_d;
    int outSize = sizeof(unsigned char) * width * height;
    int inSize = sizeof(unsigned char) * width * height * channels;
    cudaMalloc((void **)&Pin_d, inSize);
    cudaMalloc((void **)&Pout_d, outSize);

    // Copy image to device
    cudaMemcpy(Pin_d, Pin, inSize, cudaMemcpyHostToDevice);

    // Call kernel for grayscale, don't forget to cast to float, otherwise
    // the result of the division is a int, ex: 1599/32 = 49.9 so 49 if
    // not casted to a float
    dim3 dimGrid(ceil((float)width / BLOCK_X), ceil((float)height / BLOCK_Y), 1);
    dim3 dimBlock(BLOCK_X, BLOCK_Y, 1);
    color_to_grayscale_kernel<<<dimGrid, dimBlock>>>(
        Pin_d,
        Pout_d,
        width,
        height,
        channels);

    // Copy output image to host
    cudaMemcpy(Pout, Pout_d, outSize, cudaMemcpyDeviceToHost);

    // Free VRAM
    cudaFree(Pin_d);
    cudaFree(Pout_d);
}

void blur_image(
    unsigned char *Pin,
    unsigned char *Pout,
    int width,
    int height,
    int channels,
    int blur_size)
{
    // Allocate VRAM in device
    unsigned char *Pin_d, *Pout_d;
    int outSize = sizeof(unsigned char) * width * height;
    int inSize = sizeof(unsigned char) * width * height; //* channels;
    cudaMalloc((void **)&Pin_d, inSize);
    cudaMalloc((void **)&Pout_d, outSize);

    // Copy image to device
    cudaMemcpy(Pin_d, Pin, inSize, cudaMemcpyHostToDevice);

    // Call kernel for blurring and image, don't forget to cast to float, otherwise
    // the result of the division is a int, ex: 1599/32 = 49.9 so 49 if
    // not casted to a float
    dim3 dimGrid(ceil((float)width / BLOCK_X), ceil((float)height / BLOCK_Y), 1);
    dim3 dimBlock(BLOCK_X, BLOCK_Y, 1);
    blur_image_kernel<<<dimGrid, dimBlock>>>(
        Pin_d,
        Pout_d,
        width,
        height,
        blur_size);

    // Copy output image to host
    cudaMemcpy(Pout, Pout_d, outSize, cudaMemcpyDeviceToHost);

    // Free VRAM
    cudaFree(Pin_d);
    cudaFree(Pout_d);
}

int main()
{
    // Set paths
    const char *input = "911-gt3-rs-desktop_grayscale.jpg";
    const char *output = "911-gt3-rs-desktop_grayscale_blurred.jpg";

    // Load image
    int width, height, channels;
    unsigned char *Pin = stbi_load(input, &width, &height, &channels, 1);

    // Check if image is loaded
    if (!Pin)
    {
        printf("Error while loading %s\n", input);
    }
    printf("Width: %i, Height: %i, Channels: %i", width, height, channels);

    // Convert from color to grayscale
    unsigned char *Pout = (unsigned char *)malloc(width * height);
    // color_to_grayscale(Pin, Pout, width, height, channels);
    int blur_size = 7;
    blur_image(Pin, Pout, width, height, channels, blur_size);

    // Save grayscale image
    stbi_write_jpg(output, width, height, 1, Pout, width);

    // Deallocate
    stbi_image_free(Pin);
    free(Pout);

    return 0;
}
