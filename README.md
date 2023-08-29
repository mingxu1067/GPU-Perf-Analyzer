# GPU Kernel Execution Information Statistics
A tool to summarize GPU kernel execution from Nsight-systems reports, which in CSV .

## How to get CSV reports from Qdrep records.
We could convert `nsys-rep` files to their own CSV statistics report by `nsys`.</br>
`nsys stats -r cuda_gpu_kern_sum,nvtx_gpu_proj_trace --format csv -o your_csv_file_name your_nsys_report.nsys-rep`

## Usage
1. Setup kernel names and its class mapping </br>
    1.1 Here we would like to classify our GPU kernels to 6 classes, Gemm, Elementwise, LayerNorm, Dropout, Softmax and Optimizer, which are the values of given Dictionary to StatisticClassifier.</br>
    1.2 We let Kernel names which contains `GEMM`, `Gemm` and so on (whatever upper or lower case) map to `Gemm` class. For example, `ampere_sgemm_128x32_tn` would be count as a member in `Gemm`.</br>
    1.3 The keys cannot be a substring of each other. For instance, `ForRangeElemwise:elementwise, Elemwise:elementwise` is not allowed, since `Elemwise` is a substring of `ForRangeElemwise`. </br>
    1.4 Lastly, all strings would be cast to lower case for further matching.
```python
sc = StatisticClassifier({
        "gemm_e4m3":"FP8_Gemm",
        "gemm_e5m2":"FP8_Gemm",
        "gemm_bf16bf16_bf16f32_f32":"Other_Gemm",
        "xmma_gemm_f32f32":"Other_Gemm",
        "s16816gemm":"Other_Gemm",
        "s161616gemm_bf16":"Other_Gemm",
        "ln_":"LayerNorm",
        "softmax":"Softmax",
        "gelu":"Gelu",
        "fusion":"fusion",
        "convert":"convert",
        "cast_transpose":"cast_transpose",
        "transpose_optimized_kernel":"cast_transpose",
        "transpose_kernel":"cast_transpose",
        "softmax":"softmax",
        "_fmha_": "fmha",
        "reduce": "reduce"
    })
```
2. Call `statistic` with profiled iterations.
```python
# We profiled our GPU program for total 5 iterations, so we have to let statisticer know this information.
report = sc.statistic(5, cuda_kernel_sum_table, nvtx_gpu_proj_table, cuda_mem_sum_table,
                      num_processes=1, nvtx_iter_name="Train_step")
```
3. Show summarized results
```python
report.show("Paxml/GPT5B/FP8/Repeat/4FSDP_2TP")
```

# Example
```bash
$> git clone https://github.com/mingxu1067/GPU_kernel_info_statistic.git
$> cd GPU_kernel_info_statistic
$> python src/kernel_stats.py

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
