import csv

class StatisticClassifier(object):
    OTHER_CLASS_NAME = "others"

    def __init__(self, name_to_class_map):
        super().__init__()
        self._name_to_class_map = self._convert_to_lower(name_to_class_map)
        self._check_conflict(list(self._name_to_class_map.keys()))

    def statistic(self, kernels, iter_times, with_header=True,
                  total_time_idx=1, instance_idx=2, name_idx=7):

        statistic_table = {StatisticClassifier.OTHER_CLASS_NAME:[0.0, 0]}
        for key in self.name_to_class_map:
            if self._name_to_class_map[key] not in statistic_table:
                statistic_table[self._name_to_class_map[key]] = [0.0, 0]
        start_idx = 0
        if with_header :
            header = kernels[0]
            start_idx = 1
        for i in range(start_idx, len(kernels)):
            kernel_name = kernels[i][name_idx]
            class_name = self._get_class(kernel_name.lower())
            statistic_table[class_name][0] += (float(kernels[i][total_time_idx]) / int(kernels[i][instance_idx])) * \
                                              (int(kernels[i][instance_idx]) // iter_times) / 1000000.0 # convert from ns to ms.
            statistic_table[class_name][1] += int(kernels[i][instance_idx]) / iter_times

        total_time_of_one_iter = 0.0
        total_kernels_of_one_iter = 0
        for key in statistic_table:
            total_time_of_one_iter += statistic_table[key][0]
            total_kernels_of_one_iter += statistic_table[key][1]

        return statistic_table, total_time_of_one_iter, total_kernels_of_one_iter


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

def show(title, class_statistic, total_time, total_instance):
    print("=====", title, "=====")
    print("Time of a iteration: {:.3f} (ms)".format(total_time))
    print("Instance of a iteration: {}".format(int(total_instance)))
    print()
    print("{:17}{:12}{:12}{:12}".format("Class", "Time (ms)", "Instance", "Time (%)"))
    for key in class_statistic:
        print("{:15}{:10.2f}{:12}{:12.2f}".format(key, class_statistic[key][0],
                                                int(class_statistic[key][1]), (class_statistic[key][0]/total_time)*100))
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
    class_statistic, total_time, total_instance = sc.statistic(kernel_info_table, 10)
    show("PaddleNLP_BERT-large_Bat32_static", class_statistic, total_time, total_instance)

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
        "Adam":"optimizer"
    })
    class_statistic, total_time, total_instance = sc.statistic(kernel_info_table, 10)
    show("PaddleNLP_BERT-large_Bat32_dynamic", class_statistic, total_time, total_instance)


if __name__ == "__main__":
    main()
