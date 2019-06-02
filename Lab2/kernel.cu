#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <math.h>
#include "cpu_anim.h"

#define W 512
#define H 512
#define MAXX 3.1415628*2 // Масштаб по оси X - 2*PI
#define MAXY 1
#define MINY -1
#define DT 0.1

#define DIM 512

__global__ void kernel(unsigned char* ptr, int ticks)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    float c0, c1, c2;

    c0 = abs(H / (MAXY - MINY) * (sin((x + 0.0) / W * (MAXX + ticks * DT)) - MINY) - y);
    c1 = abs(H / (MAXY - MINY) * (sin((x + 0.5) / W * (MAXX + ticks * DT)) - MINY) - y);
    c2 = abs(H / (MAXY - MINY) * (sin((x - 0.5) / W * (MAXX + ticks * DT)) - MINY) - y);

    if (c0 <= 1 || c1 <= 1 || c2 <= 1)
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

// Освободить выделенную память устройства
void cleanup(DataBlock* d)
{
    cudaFree(d->dev_bitmap);
}

void generate_frame(DataBlock* d, int ticks)
{
    dim3 blocks(DIM / 16, DIM / 16);
    dim3 threads(16, 16);
    kernel <<<blocks, threads >> > (d->dev_bitmap, ticks);

    cudaMemcpy(d->bitmap->get_ptr(),
        d->dev_bitmap,
        d->bitmap->image_size(),
        cudaMemcpyDeviceToHost
    );
}

int main(void)
{
    DataBlock data;
    CPUAnimBitmap bitmap(DIM, DIM, &data);
    data.bitmap = &bitmap;

    cudaMalloc((void**)& data.dev_bitmap,
        bitmap.image_size()
    );
    bitmap.anim_and_exit((void (*)(void*, int))generate_frame,
        (void (*)(void*))cleanup
    );
    return 0;
}
