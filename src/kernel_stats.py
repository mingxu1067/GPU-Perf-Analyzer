import csv, re

class StatisticClassifier(object):
    OTHER_CLASS_NAME = "others"

    def __init__(self, name_to_class_map):
        super().__init__()
        self._name_to_class_map = self._convert_to_lower(name_to_class_map)
        self._check_conflict(list(self._name_to_class_map.keys()))

    def statistic(self, kernels, iter_times, num_processes=1, with_header=True,
                  total_time_idx=1, instance_idx=2, name_idx=7, checking_fp16=True):

        statistic_table = {StatisticClassifier.OTHER_CLASS_NAME:[0.0, 0, 0.0, 0]}
        for key in self.name_to_class_map:
            if self._name_to_class_map[key] not in statistic_table:
                statistic_table[self._name_to_class_map[key]] = [0.0, 0, 0.0, 0]
        start_idx = 0
        if with_header :
            header = kernels[0]
            start_idx = 1
        enable_fp16 = False
        for i in range(start_idx, len(kernels)):
            kernel_name = kernels[i][name_idx]
            class_name = self._get_class(kernel_name.lower())
            statistic_table[class_name][0] += (float(kernels[i][total_time_idx]) / (iter_times*num_processes)) / 1000000.0 # convert from ns to ms.
            statistic_table[class_name][1] += int(kernels[i][instance_idx]) // (iter_times*num_processes)
            statistic_table[class_name][2] += float(kernels[i][total_time_idx]) / 1000000.0 # convert from ns to ms.
            statistic_table[class_name][3] += int(kernels[i][instance_idx])
            if self.check_FP16_enabling(kernel_name):
                enable_fp16 = True

        if checking_fp16 and not enable_fp16:
            print("!! WARNING: all kernel names do not contain FP16 related pattern. !!")

        total_time_of_one_iter = 0.0
        total_kernels_of_one_iter = 0
        total_time_of_all = 0.0
        total_kernels_of_all = 0
        for key in statistic_table:
            total_time_of_one_iter += statistic_table[key][0]
            total_kernels_of_one_iter += statistic_table[key][1]
            total_time_of_all += statistic_table[key][2]
            total_kernels_of_all += statistic_table[key][3]

        return statistic_table, total_time_of_one_iter, total_kernels_of_one_iter, total_time_of_all, total_kernels_of_all


    def _get_class(self, kernel_name):
        for key in self.name_to_class_map:
            if key in kernel_name:
                return self.name_to_class_map[key]
        return StatisticClassifier.OTHER_CLASS_NAME


    def _check_conflict(self, keys):
        for i in range(len(keys)):
            for j in range(i+1, len(keys)):
                assert (keys[i] not in keys[j]) and (keys[j] not in keys[i]), \
                        "{} and {} should NOT be the substring of each other.".format(
                            keys[i], keys[j]
                        )

    def _convert_to_lower(self, map):
        result_dict = dict()
        for key in map:
            assert map[key].lower() != StatisticClassifier.OTHER_CLASS_NAME, \
                   'Class name should not be \"other\"'
            result_dict[key.lower()] = map[key].lower()
        return result_dict

    def check_FP16_enabling(self, txt):
        pattern = "_fp16|float16|h[0-9]*gemm|half"
        return re.search(pattern, txt) is not None



    @property
    def name_to_class_map(self):
        return self._name_to_class_map


def get_kernels_information(file_path):
    try:
        with open(file_path) as csvfile:
            rows = csv.reader(csvfile, delimiter=',', quotechar='"')
            return list(rows)
    except FileNotFoundError:
        print('CSV file {} does not exist'.format(file_path))

def show(title, class_statistic, total_time, total_instance, total_time_of_all, total_instance_of_all):
    print("=====", title, "=====")
    print("Time of a iteration: {:.3f} (ms)".format(total_time))
    print("Instance of a iteration: {}".format(int(total_instance)))
    print("Total Time: {:.3f}".format(int(total_time_of_all)))
    print("Total Instance: {}".format(int(total_instance_of_all)))
    print()
    print("{:17}{:12}{:12}{:12}{:17}{:15}{:12}".format("Class", 
          "Time (ms)", "Instance", "Time (%)", "TotalTime (ms)", "TotalInstance", "TotalTime (%)"))
    for key in class_statistic:
        print("{:15}{:10.2f}{:12}{:11.2f}{:18.2f}{:17}{:15.2f}".format(key, class_statistic[key][0],
                                                                int(class_statistic[key][1]), (class_statistic[key][0]/total_time)*100,
                                                                class_statistic[key][2], int(class_statistic[key][3]),
                                                                (class_statistic[key][2]/total_time_of_all)*100))
    print()


def main():
    # Paddle Statistic
    kernel_info_table = get_kernels_information("../example/paddle_static_gpukernsum.csv")
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
    results = sc.statistic(kernel_info_table, 10)
    show("PaddleNLP_BERT-large_Bat32_static", *results)

    # Paddle DyGraph
    kernel_info_table = get_kernels_information("../example/paddle_dygraph_gpukernsum.csv")
    sc = StatisticClassifier({
        "GEMM":"Gemm",
        "Elementwise":"elementwise",
        "Elemwise":"elementwise",
        "TensorCWise":"elementwise",
        "VectorizedKernel":"elementwise",
        "layernorm":"layernorm",
        "VectorizedRandom":"dropout",
        "Dropout":"dropout",
        "softmax":"softmax",
        "Adam":"optimizer",
        "cast":"cast"
    })
    results = sc.statistic(kernel_info_table, 10)
    show("PaddleNLP_BERT-large_Bat32_dynamic", *results)


if __name__ == "__main__":
    main()
