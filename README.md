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
Time of a iteration: 153.897 (ms)
Instance of a iteration: 2234

Class            Time (ms)   Instance    Time (%)
others              15.09         390        9.80
gemm                95.23         588       61.88
elementwise         19.37         464       12.58
layernorm            7.41         200        4.81
dropout              5.41         146        3.52
softmax              3.11          48        2.02
optimizer            8.28         398        5.38
'''
```

# Example
We use [PaddleNLP BERT-large pretraining](https://github.com/PaddlePaddle/PaddleNLP) as examples.</br>
```bash
$> git clone https://github.com/mingxu1067/GPU_kernel_info_statistic.git
$> cd GPU_kernel_info_statistic/src
$> python kernel_stats.py

===== PaddleNLP_BERT-large_Bat32_static =====
Time of a iteration: 153.897 (ms)
Instance of a iteration: 2234

Class            Time (ms)   Instance    Time (%)
others              15.09         390        9.80
gemm                95.23         588       61.88
elementwise         19.37         464       12.58
layernorm            7.41         200        4.81
dropout              5.41         146        3.52
softmax              3.11          48        2.02
optimizer            8.28         398        5.38



===== PaddleNLP_BERT-large_Bat32_dynamic =====
Time of a iteration: 157.332 (ms)
Instance of a iteration: 3131

Class            Time (ms)   Instance    Time (%)
others              18.61         862       11.83
gemm                92.47         588       58.77
elementwise         21.63         889       13.75
layernorm            6.98         200        4.44
dropout              5.64         146        3.58
softmax              3.16          48        2.01
optimizer            8.84         398        5.62
```
