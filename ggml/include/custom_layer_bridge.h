#ifndef CUSTOM_LAYER_BRIDGE_H
#define CUSTOM_LAYER_BRIDGE_H

#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>


// 定义常量
#define MAX_LAYERS 1005
#define MAX_TENSORS_PER_LAYER 64

#ifdef __cplusplus
extern "C" {
#endif

    // 1. 定义结构体 (确保 C 编译器能看懂)
    struct TensorLocation {
        //struct ggml_tensor * tensor;
        char name[256];
        uint64_t absolute_file_offset;
        uint64_t n_bytes;

        void* cached_pool_addr;     // CPU buffer address (used by CPU backend)
        void* cached_gpu_addr;      // GPU buffer address (used by CUDA backend)
    };

    struct LayerDiskInfo {
        int layer_id;
        bool is_initialized;
        uint64_t total_bytes_needed;
        int tensor_count;
        struct TensorLocation tensors[MAX_TENSORS_PER_LAYER];
    };

    // 2. 声明全局变量 (用 extern，告诉编译器实现在别处)
    extern struct LayerDiskInfo g_my_layer_table[MAX_LAYERS];
    extern int    g_current_loaded_layer;
    extern FILE * g_model_file;           // GGUF 文件句柄
    extern void * g_layer_buffer_A;         // 缓冲区指针
    extern void * g_layer_buffer_B;         // 缓冲区指针
    extern void * g_io_ptr;
    extern void * g_compute_ptr;
    extern size_t g_layer_buffer_size;    // 缓冲区大小




    // GPU buffer globals (defined in ggml-cuda.cu)
    extern void * g_gpu_layer_buffer_A;
    extern void * g_gpu_layer_buffer_B;
    extern void * g_gpu_io_ptr;
    extern void * g_gpu_compute_ptr;
    extern size_t g_gpu_layer_buffer_size;

    // 声明函数原型
    void load_layer_from_disk(int target_layer);
    void init_layer_table(void);
    void* get_tensor_memory_by_name(int target_layer, const char* tensor_name);
    void start_async_prefetch_engine(void);
    void ensure_layer_loaded(int target_layer, int next_layer_hint);
    int get_next_layer(int current_layer);

    // GPU-side layer management (defined in ggml-cuda.cu)
    void start_gpu_async_prefetch_engine(void);
    void ensure_gpu_layer_loaded(int target_layer, int next_layer_hint);
    void* get_gpu_tensor_memory_by_name(int target_layer, const char* tensor_name);


#ifdef __cplusplus
}
#endif

#endif