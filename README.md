# GPU Kernel Execution Information Statistics
A tool to summarize GPU kernel execution from Nsight-systems reports, which in CSV .

## How to get CSV reports from Qdrep records.
We could convert `qdrep` files to their own CSV statistics report by `nsys`.</br>
`nsys stats --format csv -o your_csv_file_name your_nsys_report.qdrep`

## Usage
1. Setup kernel names and its class mapping </br>
    1.1 Here we would like to classify our GPU kernels to 6 classes, Gemm, Elementwise, LayerNorm, Dropout, Softmax and Optimizer, which are the values of given Dictionary to StatisticClassifier.</br>
    1.2 We let Kernel names which contains `GEMM`, `Gemm` and so on (whatever upper or lower case) map to `Gemm` class. For example, `ampere_sgemm_128x32_tn` would be count as a member in `Gemm`.</br>
    1.3 The keys cannot be a substring of each other. For instance, `ForRangeElemwise:elementwise, Elemwise:elementwise` is not allowed, since `Elemwise` is a substring of `ForRangeElemwise`. </br>
    1.4 Lastly, all strings would be cast to lower case for further matching.
```python
sc = StatisticClassifier({
        "GEMM":"Gemm",
        "Elemwise":"elementwise",
        "Elementwise":"elementwise",
        "TensorCWise":"elementwise",
        "VectorizedKernel":"elementwise",
        "layernorm":"layernorm",
        "VectorizedRandom":"dropout",
        "Dropout":"dropout",
        "softmax":"softmax",
        "Adam":"optimizer"
    })
```
2. Call `statistic` with profiled iterations.
```python
# We profiled our GPU program for total 10 iterations, so we have to let statisticer know this information.
class_statistic, total_time, total_instance = sc.statistic(kernel_info_table, iter_times=10)
```
3. Show summarized results
```python
show("PaddleNLP_BERT-large_Bat32_static", class_statistic, total_time, total_instance)
'''
===== PaddleNLP_BERT-large_Bat32_static =====
Time of a iteration: 91.442 (ms)
Instance of a iteration: 2372
Total Time: 916.000
Total Instance: 23746

Class            Time (ms)   Instance    Time (%)    TotalTime (ms)   TotalInstance  TotalTime (%)
others              15.81         407      17.29            158.97             4085          17.34
gemm                42.79         587      46.79            429.36             5880          46.83
elementwise         14.28         586      15.62            142.84             5861          15.58
layernorm            4.77         200       5.21             47.66             2000           5.20
dropout              3.17         146       3.47             31.71             1460           3.46
softmax              1.83          48       2.00             18.28              480           1.99
optimizer            8.80         398       9.62             88.00             3980           9.60
'''
```

# Example
We use [PaddleNLP BERT-large pretraining](https://github.com/PaddlePaddle/PaddleNLP) as examples.</br>
```bash
$> git clone https://github.com/mingxu1067/GPU_kernel_info_statistic.git
$> cd GPU_kernel_info_statistic/src
$> python kernel_stats.py

===== PaddleNLP_BERT-large_Bat32_static =====
Time of a iteration: 91.442 (ms)
Instance of a iteration: 2372
Total Time: 916.000
Total Instance: 23746

Class            Time (ms)   Instance    Time (%)    TotalTime (ms)   TotalInstance  TotalTime (%)
others              15.81         407      17.29            158.97             4085          17.34
gemm                42.79         587      46.79            429.36             5880          46.83
elementwise         14.28         586      15.62            142.84             5861          15.58
layernorm            4.77         200       5.21             47.66             2000           5.20
dropout              3.17         146       3.47             31.71             1460           3.46
softmax              1.83          48       2.00             18.28              480           1.99
optimizer            8.80         398       9.62             88.00             3980           9.60

===== PaddleNLP_BERT-large_Bat32_dynamic =====
Time of a iteration: 121.296 (ms)
Instance of a iteration: 4021
Total Time: 1215.000
Total Instance: 40220

Class            Time (ms)   Instance    Time (%)    TotalTime (ms)   TotalInstance  TotalTime (%)
others              17.86         764      14.72            178.58             7640          14.69
gemm                42.75         587      35.24            429.92             5880          35.37
elementwise         19.88         988      16.39            198.78             9880          16.35
layernorm            4.60         200       3.79             45.96             2000           3.78
dropout              4.13         146       3.40             41.29             1460           3.40
softmax              1.94          48       1.60             19.43              480           1.60
optimizer            8.61         398       7.10             86.10             3980           7.08
cast                21.54         890      17.75            215.35             8900          17.72
```
