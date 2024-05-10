from dataclasses import dataclass

OTHER_CLASS_NAME = "others"
NCCL_CLASS_NAME = "nccl"

@dataclass
class Report:
    kernel_statistic_table: dict = None
    kernel_total_time_per_iter: float = 0.0
    kernel_total_instancess_per_iter: int = 0
    kernel_total_time: float = 0.0
    kernel_total_instances: int = 0
    nccl_total_time_per_iter:float = 0.0
    nccl_total_kernels_per_iter: int = 0
    nccl_total_time: float = 0.0
    nccl_total_kernels: int = 0
    nccl_statistic_table: dict = None
    nvtx_avg_range_time: float = 0.0
    nvtx_total_range_time: float = 0.0
    nvtx_range_count: int = 0
    mem_statistic_table: dict = None

    def show(self, title=None):
        if title is not None:
            print("=====", title, "=====")

        assert self.kernel_statistic_table is not None

        print("Kernel time per iteration: {:.3f} (ms)".format(self.kernel_total_time_per_iter))
        print("Kernel instances per iteration: {}".format(self.kernel_total_instancess_per_iter))
        print("Kernel total time: {:.3f} (ms)".format(self.kernel_total_time))
        print("Kernel total instances: {}".format(self.kernel_total_instances))
        print("Kernel Statistic:")
        print("{:17}{:17}{:17}{:17}{:17}{:15}{:15}".format("Class", 
            "TimePerIter(ms)", "InstancePerIter", "TimePerIter(%)", "TotalTime(ms)", "TotalInstance", "TotalTime(%)"))
        for key in self.kernel_statistic_table:
            if key != NCCL_CLASS_NAME:
                print("{:17}{:15.2f}{:17}{:16.2f}{:16.2f}{:17}{:14.2f}".format(
                    key, self.kernel_statistic_table[key][0], self.kernel_statistic_table[key][1],
                    (self.kernel_statistic_table[key][0]/self.kernel_total_time_per_iter)*100,
                    self.kernel_statistic_table[key][2], self.kernel_statistic_table[key][3],
                    (self.kernel_statistic_table[key][2]/self.kernel_total_time)*100))
        print("")
        print("NCCL time per iteration: {:.3f} (ms)".format(self.nccl_total_time_per_iter))
        print("NCCL instances per iteration: {}".format(self.nccl_total_kernels_per_iter))
        print("NCCL total time: {:.3f} (ms)".format(self.nccl_total_time))
        print("NCCL total instances: {}".format(self.nccl_total_kernels))
        print("NCCL Statistic:")
        print("{:17}{:17}{:17}{:17}{:15}".format("Class", 
            "TimePerIter(ms)", "InstancePerIter", "TotalTime(ms)", "TotalInstance"))
        for key in self.nccl_statistic_table:
            print("{:17}{:15.2f}{:17}{:15.2f}{:17}".format(
                    key, self.nccl_statistic_table[key][0], int(self.nccl_statistic_table[key][1]),
                    self.nccl_statistic_table[key][2], int(self.nccl_statistic_table[key][3])))

        if self.mem_statistic_table is not None:
            name_map = {
                "[CUDA memcpy Device-to-Device]": "[CUDA memcpy D2D]",
                "[CUDA memcpy Device-to-Host]": "[CUDA memcpy D2H]",
                "[CUDA memcpy Host-to-Device]": "[CUDA memcpy H2D]",
                "[CUDA memcpy Host-to-Host]": "[CUDA memcpy H2H]",
            }

            print("")
            print("Mem Ops Statistic:")
            print("{:23}{:17}{:17}{:17}{:15}".format("Class",
                "TimePerIter(ms)", "InstancePerIter", "TotalTime(ms)", "TotalInstance"))
            for key in self.mem_statistic_table:
                display_key = name_map.get(key, key)
                print("{:23}{:15.2f}{:17}{:15.2f}{:17}".format(
                    display_key, self.mem_statistic_table[key][0], self.mem_statistic_table[key][1],
                    self.mem_statistic_table[key][2], self.mem_statistic_table[key][3]))

        print("")
        if self.nvtx_range_count > 0:
            print("NVTX range count: {}".format(self.nvtx_range_count))
            print("NVTX average range time: {:.3f} (ms)".format(self.nvtx_avg_range_time))
            print("NVTX total range time: {:.3f} (ms)".format(self.nvtx_total_range_time))
            print("")
            print("Estimated non-hidden NCCL, mem ops and bubble time per range: {:.3f} (ms)".format(
                (self.nvtx_avg_range_time - self.kernel_total_time_per_iter)))
            print("Estimated non-hidden NCCL, mem ops and bubble time per range: {:.2f} (%)".format(
                (self.nvtx_avg_range_time - self.kernel_total_time_per_iter) / self.nvtx_avg_range_time * 100))
            print("Estimated non-hidden NCCL, mem ops and bubble total time: {:.3f} (ms)".format(
                (self.nvtx_total_range_time - self.kernel_total_time)))
            print("Estimated non-hidden NCCL, mem ops and bubble total time: {:.2f} (%)".format(
                (self.nvtx_total_range_time - self.kernel_total_time) / self.nvtx_total_range_time * 100))


class Analyzer(object):

    def __init__(self, kernel2class_map):
        super().__init__()
        self._kernel2class_map = self._convert_to_lower(kernel2class_map)
        self._check_conflict(list(self._kernel2class_map.keys()))

    def analyze(self, num_of_iters, kernels, nvtxes=None, mems=None,
                num_of_gpus=1, with_header=True, time_scale_to_ms=(1/1000000.0),
                kernel_total_time_idx=1, kernel_instance_idx=2, kernel_name_idx=8,
                nvtx_tag_of_iter=None, nvtx_tag_idx=0, nvtx_proj_duration_idx=2,
                mem_name_idx=8, mem_time_idx=1, mem_count_idx=2):
        reports = {}


        kernel_statistic_report = self.statistic_kernels(kernels, num_of_iters, num_of_gpus, with_header,
                                                         time_scale_to_ms, kernel_total_time_idx, kernel_instance_idx,
                                                         kernel_name_idx)
        reports["kernel_statistic_table"] = kernel_statistic_report.kernel_statistic_table
        reports["kernel_total_time_per_iter"] = kernel_statistic_report.kernel_total_time_per_iter
        reports["kernel_total_instancess_per_iter"] = kernel_statistic_report.kernel_total_instancess_per_iter
        reports["kernel_total_time"] = kernel_statistic_report.kernel_total_time
        reports["kernel_total_instances"] = kernel_statistic_report.kernel_total_instances
        reports["nccl_total_time_per_iter"]=kernel_statistic_report.nccl_total_time_per_iter
        reports["nccl_total_kernels_per_iter"]=kernel_statistic_report.nccl_total_kernels_per_iter
        reports["nccl_total_time"]=kernel_statistic_report.nccl_total_time
        reports["nccl_total_kernels"]=kernel_statistic_report.nccl_total_kernels
        reports["nccl_statistic_table"]=kernel_statistic_report.nccl_statistic_table

        if nvtxes is not None:
            nvtx_statistic_report = self.statistic_nvtx(nvtxes, nvtx_tag_of_iter, num_of_iters,
                                                        with_header, time_scale_to_ms, nvtx_tag_idx,
                                                        nvtx_proj_duration_idx)
            reports["nvtx_avg_range_time"] = nvtx_statistic_report.nvtx_avg_range_time
            reports["nvtx_total_range_time"] = nvtx_statistic_report.nvtx_total_range_time
            reports["nvtx_range_count"] = nvtx_statistic_report.nvtx_range_count

        mem_statistic_report = None
        if mems is not None:
            mem_statistic_report = self.statistic_mem(mems, num_of_iters, num_of_gpus, with_header,
                                                      time_scale_to_ms, mem_name_idx, mem_time_idx, mem_count_idx)
            reports["mem_statistic_table"] = mem_statistic_report.mem_statistic_table

        return Report(**reports)

    def statistic_kernels(self, kernels, num_of_iters, num_of_gpus=1, with_header=True, time_scale_to_ms=(1/1000000.0),
                          kernel_total_time_idx=1, kernel_instance_idx=2, kernel_name_idx=8):
        statistic_table = {OTHER_CLASS_NAME:[0.0, 0, 0.0, 0],
                           NCCL_CLASS_NAME:[0.0, 0, 0.0, 0]}
        nccl_statistic_table = {}
        for key in self._kernel2class_map:
            if self._kernel2class_map[key] not in statistic_table:
                statistic_table[self._kernel2class_map[key]] = [0.0, 0, 0.0, 0]
        start_idx = 0
        if with_header :
            # header = kernels[0]
            start_idx = 1
        for i in range(start_idx, len(kernels)):
            kernel_name = kernels[i][kernel_name_idx]
            class_name = self._get_class(kernel_name.lower())

            statistic_table[class_name][0] += (float(kernels[i][kernel_total_time_idx]) / (num_of_iters*num_of_gpus)) * time_scale_to_ms
            statistic_table[class_name][1] += int(kernels[i][kernel_instance_idx]) // (num_of_iters*num_of_gpus)
            statistic_table[class_name][2] += float(kernels[i][kernel_total_time_idx]) * time_scale_to_ms
            statistic_table[class_name][3] += int(kernels[i][kernel_instance_idx])

            if class_name == NCCL_CLASS_NAME:
                nccl_name = kernel_name.split("_")[1]
                if nccl_name not in nccl_statistic_table:
                    nccl_statistic_table[nccl_name] = [0.0, 0, 0.0, 0]
                nccl_statistic_table[nccl_name][0] += (float(kernels[i][kernel_total_time_idx]) / (num_of_iters*num_of_gpus)) * time_scale_to_ms
                nccl_statistic_table[nccl_name][1] += int(kernels[i][kernel_instance_idx]) // (num_of_iters*num_of_gpus)
                nccl_statistic_table[nccl_name][2] += float(kernels[i][kernel_total_time_idx]) * time_scale_to_ms
                nccl_statistic_table[nccl_name][3] += int(kernels[i][kernel_instance_idx])

        nccl_total_time_per_iter, nccl_total_kernels_per_iter, nccl_total_time, nccl_total_kernels = \
            statistic_table.pop(NCCL_CLASS_NAME)

        total_time_of_one_iter = 0.0
        total_kernels_of_one_iter = 0
        total_time_of_all = 0.0
        total_kernels_of_all = 0
        for key in statistic_table:
            if key != NCCL_CLASS_NAME:
                total_time_of_one_iter += statistic_table[key][0]
                total_kernels_of_one_iter += statistic_table[key][1]
                total_time_of_all += statistic_table[key][2]
                total_kernels_of_all += statistic_table[key][3]

        return Report(
            kernel_statistic_table=statistic_table,
            kernel_total_time_per_iter=total_time_of_one_iter,
            kernel_total_instancess_per_iter=total_kernels_of_one_iter,
            kernel_total_time=total_time_of_all,
            kernel_total_instances=total_kernels_of_all,
            nccl_total_time_per_iter=nccl_total_time_per_iter,
            nccl_total_kernels_per_iter=nccl_total_kernels_per_iter,
            nccl_total_time=nccl_total_time,
            nccl_total_kernels=nccl_total_kernels,
            nccl_statistic_table=nccl_statistic_table
        )

    def statistic_nvtx(self, nvtxes, nvtx_tag_of_iter, num_of_iters,
                       with_header=True, time_scale_to_ms=(1/1000000.0),
                       nvtx_tag_idx=0, nvtx_proj_duration_idx=2):

        assert nvtx_tag_of_iter is not None

        nvtx_iter_total_time = 0.0
        nvtx_iter_count = 0
        start_idx = 0
        if with_header :
            # header = nvtxes[0]
            start_idx = 1
        for i in range(start_idx, len(nvtxes)):
            nvtx_name = nvtxes[i][nvtx_tag_idx]
            if nvtx_name == nvtx_tag_of_iter:
                nvtx_iter_total_time += float(nvtxes[i][nvtx_proj_duration_idx])
                nvtx_iter_count += 1

        if nvtx_iter_count != num_of_iters:
            print(f"[Warning]: The count of NVTX iterations {nvtx_iter_count} does not equal to the given num_of_iters {num_of_iters}"
                  f". It could led to mis-statistic.")

        nvtx_iter_total_time = nvtx_iter_total_time * time_scale_to_ms
        nvtx_iter_time_avg = nvtx_iter_total_time / nvtx_iter_count

        return Report(
            nvtx_avg_range_time=nvtx_iter_time_avg,
            nvtx_total_range_time=nvtx_iter_total_time,
            nvtx_range_count=nvtx_iter_count
        )

    def statistic_mem(self, mems, num_of_iters, num_of_gpus=1,
                      with_header=True, time_scale_to_ms=(1/1000000.0),
                      mem_name_idx=8, mem_time_idx=1, mem_count_idx=2):

        statistic_table = {}

        start_idx = 0
        if with_header :
            # header = kernels[0]
            start_idx = 1
        for i in range(start_idx, len(mems)):
            mem_op = mems[i][mem_name_idx]
            if mem_op not in statistic_table:
                statistic_table[mem_op] = [0.0, 0, 0.0, 0]

            statistic_table[mem_op][0] += (float(mems[i][mem_time_idx]) / (num_of_iters*num_of_gpus)) * time_scale_to_ms
            statistic_table[mem_op][1] += int(mems[i][mem_count_idx]) // (num_of_iters*num_of_gpus)
            statistic_table[mem_op][2] += float(mems[i][mem_time_idx]) * time_scale_to_ms
            statistic_table[mem_op][3] += int(mems[i][mem_count_idx])

        return Report(
            mem_statistic_table=statistic_table
        )

    def _get_class(self, kernel_name):
        if "nccl" in kernel_name:
            return NCCL_CLASS_NAME

        for key in self._kernel2class_map:
            if key in kernel_name:
                return self._kernel2class_map[key]
        return OTHER_CLASS_NAME


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
            assert map[key].lower() != OTHER_CLASS_NAME, \
                   'Class name should not be \"other\"'
            result_dict[key.lower()] = map[key].lower()
        return result_dict
