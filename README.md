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
    "input_format": "csv", // The format of input files, should be ["csv", "nsys-rep"]
    "nsys_report": { // Information for analyzing nsys-rep, is only used when input_format="nsys-rep"
        "path": null, // The file path of nsys-rep.
        "with_nvtx": false, // Indicates whether to analyze NVTX projection or not.
        "with_mem": false  // Indicates whether to analyze memory operation or not.
    },
    "nsys_csv_file_path": { // Information for analyzing nsys-rep, is only used when input_format="csv"
        "kernel_sum":"example/demo_cuda_gpu_kern_sum.csv", // The file path of pre-generated cuda_gpu_kern_sum.csv from a nsys-rep. Must be provided.
        "nvtx_proj_trace":"example/demo_nvtx_gpu_proj_trace.csv", // The file path of pre-generated nvtx_gpu_proj_trace.csv from a nsys-rep. Optional, could be null.
        "mem_time_sum": "example/demo_cuda_gpu_mem_time_sum.csv" // The file path of pre-generated cuda_gpu_mem_time_sum.csv from a nsys-rep. Optional, could be null.
    },
    "kernel_to_class_map": { // A map indicating how to classify GPU kernels. Refer to the Kernel2Class Map section for details.
        "part_of_kernel_name": "corresponding_class"
    },
    "title": "Paxml/GPT5B/FP8/Repeat/4FSDP_2TP", // The title of this analysis
    "analysis_args": { 
        "num_of_iters": 5, // Indicates how many repeat runs/interations are contained in input nsys-rep/csv files.
        "num_of_gpus": 1, // Indicates how many GPUs are contained in input nsys-rep/csv files.
        "nvtx_tag_of_iter": "Train_step", // Indicates the NVTX tag of the run/iteration. Must be provided when NVTX analysis is enabled.
    } 
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


# Example
```bash
$> python3 analyzer/main.py --config='example/config.json'

===== Paxml/GPT5B/FP8/R/142 =====
Kernel time per iteration: 307.620 (ms)
Kernel instances per iteration: 3355
Kernel total time: 1538.100 (ms)
Kernel total instances: 16775
Kernel Statistic:
Class            TimePerIter(ms)  InstancePerIter  TimePerIter(%)   TotalTime(ms)    TotalInstance  TotalTime(%)   
others                      1.30               84            0.42            6.52              420          0.42
fp8_gemm                  109.21              360           35.50          546.04             1800         35.50
other_gemm                 27.49              199            8.94          137.45              995          8.94
layernorm                  11.11              192            3.61           55.53              960          3.61
softmax                     7.91               72            2.57           39.57              360          2.57
gelu                        0.00                0            0.00            0.00                0          0.00
fusion                    125.56             1899           40.82          627.82             9495         40.82
convert                     0.62               55            0.20            3.09              275          0.20
cast_transpose             24.26              432            7.89          121.30             2160          7.89
fmha                        0.00                0            0.00            0.00                0          0.00
reduce                      0.16               62            0.05            0.78              310          0.05

NCCL time per iteration: 117.652 (ms)
NCCL instances per iteration: 533
NCCL total time: 588.258 (ms)
NCCL total instances: 2665
NCCL Statistic:
Class            TimePerIter(ms)  InstancePerIter  TotalTime(ms)    TotalInstance  
AllGather                  63.38              248         316.90             1240
ReduceScatter              35.34              172         176.70              860
AllReduce                  18.40              111          92.00              555
SendRecv                    0.53                2           2.66               10

Mem Ops Statistic:
Class                  TimePerIter(ms)  InstancePerIter  TotalTime(ms)    TotalInstance  
[CUDA memcpy DtoD]               42.11              163         210.56              815
[CUDA memset]                     0.69              530           3.45             2650
[CUDA memcpy DtoH]                0.12               53           0.60              265
[CUDA memcpy HtoD]                0.01               10           0.06               50

NVTX range count: 5
NVTX average range time: 404.640 (ms)
NVTX total range time: 2023.200 (ms)

Estimated non-hidden NCCL, mem ops and bubble time per range: 97.020 (ms)
Estimated non-hidden NCCL, mem ops and bubble time per range: 23.98 (%)
Estimated non-hidden NCCL, mem ops and bubble time: 485.099 (ms)
Estimated non-hidden NCCL, mem ops and bubble time: 23.98 (%)
```
