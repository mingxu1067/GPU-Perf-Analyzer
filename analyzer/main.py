import argparse, csv, json
import shutil, subprocess, sys
import uuid

from analysis import Analyzer

FORMAT_CSV = "csv"
FORMAT_NSYS_REPORT = "nsys-rep"

CUDA_MEM_SUM_POSTFIX = "cuda_gpu_mem_time_sum"
CUDA_KERNEL_SUM_POSTFIX = "cuda_gpu_kern_sum"
NVTX_GPU_PROJ_POSTFIX = "nvtx_gpu_proj_trace"

def read_csv_statistic(csv_file_path):
    try:
        with open(csv_file_path) as csvfile:
            return list(csv.reader(csvfile, delimiter=',', quotechar='"'))
    except FileNotFoundError as e:
        print(e)

def parse_args():
    parser = argparse.ArgumentParser(description="GPU Performance Analyzer",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--config',
                        type=str,
                        required=True,
                        help="the path to load a configuration.")

    args = parser.parse_args()

    return args


def nsys_rep_reader(config):
    CUDA_KERNEL_SUM_POSTFIX = "cuda_gpu_kern_sum"
    NVTX_GPU_PROJ_POSTFIX = "nvtx_gpu_proj_trace"
    CUDA_MEM_SUM_POSTFIX = "cuda_gpu_mem_time_sum"

    output_name = f"/tmp/gpu_pef_analysis_{uuid.uuid4()}"
    csv_reader_config = {}
    csv_reader_config["kernel_sum"] = f"{output_name}_{CUDA_KERNEL_SUM_POSTFIX}.{FORMAT_CSV}"
    csv_report_types = ['cuda_gpu_kern_sum']
    if config.get("with_nvtx", False):
        csv_report_types += ["nvtx_gpu_proj_trace"]
        csv_reader_config["nvtx_proj_trace"] = f"{output_name}_{NVTX_GPU_PROJ_POSTFIX}.{FORMAT_CSV}"
    if config.get("with_mem", False):
        csv_report_types += ["cuda_gpu_mem_time_sum"]
        csv_reader_config["mem_time_sum"] = f"{output_name}_{CUDA_MEM_SUM_POSTFIX}.{FORMAT_CSV}"
    csv_report_types = ','.join(csv_report_types)

    nsys_path = shutil.which("nsys")
    stats_command = [nsys_path, "stats", "-r", csv_report_types, "--format", "csv", "-o",  output_name, config["path"]]
    proc = subprocess.Popen(stats_command,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.STDOUT)
    proc.wait()

    return csv_reader(csv_reader_config)


def csv_reader(config):

    def read_csv_statistic(file_path):
        try:
            with open(file_path) as csvfile:
                return list(csv.reader(csvfile, delimiter=',', quotechar='"'))
        except:
            return None

    kernel_sum_table = read_csv_statistic(config["kernel_sum"])
    nvtx_proj_trace_table = read_csv_statistic(config["nvtx_proj_trace"])
    mem_time_sum_table = read_csv_statistic(config["mem_time_sum"])

    assert kernel_sum_table is not None
    
    return kernel_sum_table, nvtx_proj_trace_table, mem_time_sum_table

def read_statistic_tables(config):
    format = config["format"].lower()

    if format == FORMAT_CSV:
        return csv_reader(config["nsys_csv_file_path"])
    elif format == FORMAT_NSYS_REPORT:
         return nsys_rep_reader(config["nsys_report"])
    else:
        raise NotImplementedError(
            f"Not supported input format: {format}")



def main(args):

    config = json.loads(open(args.config).read())

    analyer = Analyzer(
        config["kernel_to_class_map"]
    )

    for input_config in config["inputs"]:
        kernel_sum_table, nvtx_proj_trace_table, mem_time_sum_table = \
            read_statistic_tables(input_config)

        report = analyer.analyze(
            kernels=kernel_sum_table,
            nvtxes=nvtx_proj_trace_table,
            mems=mem_time_sum_table,
            **input_config["analysis_args"])
    
        report.show(input_config["title"])
        print("\n")

if __name__ == "__main__":
    args = parse_args()
    main(args)
