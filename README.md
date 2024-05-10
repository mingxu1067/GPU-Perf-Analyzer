# GPU Performace Analyser
GPU Performance Analyzer is a tool that summarizes GPU kernel execution information from Nsight Systems reports. It reads these reports and extracts key metrics such as GPU kernel execution time, memory operation time, and NVTX projection. By aggregating this information, the tool helps users assess the performance and efficiency of their GPU programs. It provides users with an easier way to understand and analyze the performance of their GPU programs during execution.

## Usage

This tool is JSON config based and supports formats of nsys reports and pre-generated `csv` files from nsys reports.

### Generate csv files from nsys-rep
```bash
nsys stats -r cuda_gpu_kern_sum,nvtx_gpu_proj_trace,cuda_gpu_mem_time_sum --format csv -o your_csv_file_name_prefix your_nsys_report.nsys-rep
```

### command to run
```bash
python3 src/main.py --config=your_config_path
```

### JSON config
```json
{
    "inputs": [ // A list of inputs' information which shared the same "kernel_to_class_map".
        {
            "format": "csv", // The format of input files, should be ["csv", "nsys-rep"]
            "title": "Llama2-70B-From-CSV", // The title of this analysis
            "nsys_csv_file_path": { // Information for analyzing nsys-rep, is only used when input_format="csv"
                "kernel_sum":"example/llama2-70B-GBS512-FSDP256_cuda_gpu_kern_sum.csv", // The file path of pre-generated cuda_gpu_kern_sum.csv from a nsys-rep. Must be provided.
                "nvtx_proj_trace":"example/llama2-70B-GBS512-FSDP256_nvtx_gpu_proj_trace.csv", // The file path of pre-generated nvtx_gpu_proj_trace.csv from a nsys-rep. Optional, could be null.
                "mem_time_sum": "example/llama2-70B-GBS512-FSDP256_cuda_gpu_mem_time_sum.csv" // The file path of pre-generated cuda_gpu_mem_time_sum.csv from a nsys-rep. Optional, could be null.
            },
            "analysis_args": { 
                "num_of_iters": 5, // Indicates how many repeat runs/interations are contained in input nsys-rep/csv files.
                "num_of_gpus": 1, // Indicates how many GPUs are contained in input nsys-rep/csv files.
                "nvtx_tag_of_iter": "TSL:XlaModule:#hlo_module=pjit__train_step,program_id=11#", // Indicates the NVTX tag of the run/iteration. Must be provided when NVTX analysis is enabled.
            } 
        },
        {
            "format": "nsys-rep", // The format of input files, should be ["csv", "nsys-rep"]
            "nsys_report": { // Information for analyzing nsys-rep, is only used when input_format="nsys-rep"
                "path": "example/llama2-70B-GBS512-FSDP256.nsys-rep", // The file path of nsys-rep.
                "with_nvtx": true, // Indicates whether to analyze NVTX projection or not.
                "with_mem": true // Indicates whether to analyze memory operation or not.
            },
            "title": "Llama2-70B-From-NSYS-REP", // The title of this analysis
            "analysis_args": { 
                "num_of_iters": 5, // Indicates how many repeat runs/interations are contained in input nsys-rep/csv files.
                "num_of_gpus": 1, // Indicates how many GPUs are contained in input nsys-rep/csv files.
                "nvtx_tag_of_iter": "TSL:XlaModule:#hlo_module=pjit__train_step,program_id=11#", // Indicates the NVTX tag of the run/iteration. Must be provided when NVTX analysis is enabled.
            } 
        }
    ],
    "kernel_to_class_map": { // A map indicating how to classify GPU kernels. Refer to the Kernel2Class Map section for details.
        "part_of_kernel_name": "corresponding_class"
    },
}
```

### Kernel2Class Map
The Kernel2Class Map indicates how to classify GPU kernels into their corresponding classes. The key is a partial kernel name, and the value is its corresponding class. This means that all kernels containing the key will be classified into the value class. </br>

For example, consider the following map: </br>
```json
{
    "gemm":"gemm",
    "fusion":"fusion",
    "cast_transpose":"cast_transpose",
    "transpose_optimized_kernel":"cast_transpose",
    "transpose_kernel":"cast_transpose",
}
```
This map indicates that kernels with names containing `gemm` should be classified into the `gemm` class, such as `xmma_gemm_f32f32`, `s16816gemm`, or `s161616gemm_bf16`. Similarly, kernels with names containing `cast_transpose`, `transpose_optimized_kernel`, or `transpose_kernel` should be classified into the `cast_transpose` class. </br>

Notes: </br>
1. Kernels whose names do not contain any one of keys in this map will be classified into the `others` class. </br>
2. All NCCL kernels will be automatically classified into the `nccl` class in the backend. </br>


## Example
```bash
$> python3 analyzer/main.py --config='example/config.json'

===== Llama2-70B-From-CSV =====
Kernel time per iteration: 7947.172 (ms)
Kernel instances per iteration: 7877
Kernel total time: 39735.860 (ms)
Kernel total instances: 39385
Kernel Statistic:
Class            TimePerIter(ms)  InstancePerIter  TimePerIter(%)   TotalTime(ms)    TotalInstance  TotalTime(%)
others                     79.53             1518            1.00          397.67             7590          1.00
gemm                     6830.11             1363           85.94        34150.57             6815         85.94
fmha                      555.79              480            6.99         2778.96             2400          6.99
fusion                    481.73             4516            6.06         2408.65            22580          6.06

NCCL time per iteration: 3275.659 (ms)
NCCL instances per iteration: 244
NCCL total time: 16378.295 (ms)
NCCL total instances: 1220
NCCL Statistic:
Class            TimePerIter(ms)  InstancePerIter  TotalTime(ms)    TotalInstance
AllGather                2343.77              161       11718.85              805
ReduceScatter             924.66               79        4623.31              395
AllReduce                   7.23                4          36.14               20

Mem Ops Statistic:
Class                  TimePerIter(ms)  InstancePerIter  TotalTime(ms)    TotalInstance
[CUDA memcpy D2D]                30.66              181         153.28              905
[CUDA memset]                     8.57             1527          42.85             7635
[CUDA memcpy H2D]                 0.01                3           0.05               15

NVTX range count: 5
NVTX average range time: 8578.017 (ms)
NVTX total range time: 42890.084 (ms)

Estimated non-hidden NCCL, mem ops and bubble time per range: 630.845 (ms)
Estimated non-hidden NCCL, mem ops and bubble time per range: 7.35 (%)
Estimated non-hidden NCCL, mem ops and bubble total time: 3154.224 (ms)
Estimated non-hidden NCCL, mem ops and bubble total time: 7.35 (%)


===== Llama2-70B-From-NSYS-REP =====
Kernel time per iteration: 7947.172 (ms)
Kernel instances per iteration: 7877
Kernel total time: 39735.860 (ms)
Kernel total instances: 39385
Kernel Statistic:
Class            TimePerIter(ms)  InstancePerIter  TimePerIter(%)   TotalTime(ms)    TotalInstance  TotalTime(%)
others                     79.53             1518            1.00          397.67             7590          1.00
gemm                     6830.11             1363           85.94        34150.57             6815         85.94
fmha                      555.79              480            6.99         2778.96             2400          6.99
fusion                    481.73             4516            6.06         2408.65            22580          6.06

NCCL time per iteration: 3275.659 (ms)
NCCL instances per iteration: 244
NCCL total time: 16378.295 (ms)
NCCL total instances: 1220
NCCL Statistic:
Class            TimePerIter(ms)  InstancePerIter  TotalTime(ms)    TotalInstance
AllGather                2343.77              161       11718.85              805
ReduceScatter             924.66               79        4623.31              395
AllReduce                   7.23                4          36.14               20

Mem Ops Statistic:
Class                  TimePerIter(ms)  InstancePerIter  TotalTime(ms)    TotalInstance
[CUDA memcpy D2D]                30.66              181         153.28              905
[CUDA memset]                     8.57             1527          42.85             7635
[CUDA memcpy H2D]                 0.01                3           0.05               15

NVTX range count: 5
NVTX average range time: 8578.017 (ms)
NVTX total range time: 42890.084 (ms)

Estimated non-hidden NCCL, mem ops and bubble time per range: 630.845 (ms)
Estimated non-hidden NCCL, mem ops and bubble time per range: 7.35 (%)
Estimated non-hidden NCCL, mem ops and bubble total time: 3154.224 (ms)
Estimated non-hidden NCCL, mem ops and bubble total time: 7.35 (%)
```
