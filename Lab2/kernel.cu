#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "cpu_anim.h"

#define THREADS_IN_BLOCK 16
#define TIME_SPEED 4
#define MAXX 10
#define MAXY 1
#define HEIGHT 256 // must be dividable by THREADS_IN_BLOCK

#define MINY (-MAXY)
#define Y_LENGTH (2 * MAXY)
#define WIDTH (HEIGHT * MAXX / Y_LENGTH)


__device__ double formula(double x)
{
    return sin(5 * exp(cos(x / 5)));
}

__device__ double inTime(double x, int ticks, double delta)
{
    double time = ticks * TIME_SPEED;
    return x + delta + time;
}


__global__ void kernel(unsigned char* ptr, int ticks)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    if (y < HEIGHT / 2 + 1 && y > HEIGHT / 2 - 1)
    {
        ptr[offset * 4 + 0] = ptr[offset * 4 + 1] =
        ptr[offset * 4 + 2] = ptr[offset * 4 + 3] = 0;
        return;
    }

    double c0, c1, c2;

    double xInTime1 = inTime(x, ticks, 0.0);
    double xInTime2 = inTime(x, ticks, 0.5);
    double xInTime3 = inTime(x, ticks, -0.5);

    double xMapped1 = xInTime1 / WIDTH * MAXX;
    double xMapped2 = xInTime2 / WIDTH * MAXX;
    double xMapped3 = xInTime3 / WIDTH * MAXX;

    c0 = abs(HEIGHT / Y_LENGTH * (formula(xMapped1) - MINY) - y);
    c1 = abs(HEIGHT / Y_LENGTH * (formula(xMapped2) - MINY) - y);
    c2 = abs(HEIGHT / Y_LENGTH * (formula(xMapped3) - MINY) - y);

    if (c0 <= 1.2 || c1 <= 1.2 || c2 <= 1.2)
        ptr[offset * 4 + 1] = ptr[offset * 4 + 2] = 0;
    else
        ptr[offset * 4 + 1] = ptr[offset * 4 + 2] = 255;

    ptr[offset * 4 + 0] = 255;
    ptr[offset * 4 + 3] = 255;
}


struct DataBlock
{
    unsigned char* dev_bitmap;
    CPUAnimBitmap* bitmap;
};

void cleanup(DataBlock* d)
{
    cudaFree(d->dev_bitmap);
}

void generate_frame(DataBlock* d, int ticks)
{
    dim3 blocks(WIDTH / THREADS_IN_BLOCK, HEIGHT / THREADS_IN_BLOCK);
    dim3 threads(THREADS_IN_BLOCK, THREADS_IN_BLOCK);
    kernel <<<blocks, threads>>> (d->dev_bitmap, ticks);

    cudaMemcpy(d->bitmap->get_ptr(),
        d->dev_bitmap,
        d->bitmap->image_size(),
        cudaMemcpyDeviceToHost
    );
}

int main(void)
{
    DataBlock data;
    CPUAnimBitmap bitmap(WIDTH, HEIGHT, &data);
    data.bitmap = &bitmap;

    cudaMalloc((void**)& data.dev_bitmap,
        bitmap.image_size()
    );
    bitmap.anim_and_exit((void (*)(void*, int))generate_frame,
        (void (*)(void*))cleanup
    );
    return 0;
}
