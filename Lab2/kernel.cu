#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "cpu_anim.h"

#define W 640
#define H 480
#define MAXX 20
#define MAXY 1
#define MINY -1
#define DT 5

#define DIM_WIDTH (W)
#define DIM_HEIGHT (H)

__device__ double formula(double x)
{
    return sin(5 * exp(cos(x / 5)));
}


__global__ void kernel(unsigned char* ptr, int ticks)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    double c0, c1, c2;

    double time = ticks * DT;

    double xMappedInTime1 = (x + time) / W * MAXX;
    double xMappedInTime2 = (x + time + 0.5) / W * MAXX;
    double xMappedInTime3 = (x + time - 0.5) / W * MAXX;

    c0 = abs(H / (MAXY - MINY) * (formula(xMappedInTime1) - MINY) - y);
    c1 = abs(H / (MAXY - MINY) * (formula(xMappedInTime2) - MINY) - y);
    c2 = abs(H / (MAXY - MINY) * (formula(xMappedInTime3) - MINY) - y);

    if (c0 <= 1.5 || c1 <= 1.5 || c2 <= 1.5)
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
    dim3 blocks(DIM_WIDTH / 16, DIM_HEIGHT / 16);
    dim3 threads(16, 16);
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
    CPUAnimBitmap bitmap(DIM_WIDTH, DIM_HEIGHT, &data);
    data.bitmap = &bitmap;

    cudaMalloc((void**)& data.dev_bitmap,
        bitmap.image_size()
    );
    bitmap.anim_and_exit((void (*)(void*, int))generate_frame,
        (void (*)(void*))cleanup
    );
    return 0;
}
