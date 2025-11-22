import subprocess

# === 参数设置 ===
SELEX_path = "/home/hanlab/2022_zzm/InstructNA_new/data/function_SELEX/IFN-gama/R4/InstructNA_seq_input.txt"
seq_act_path = "/home/hanlab/2022_zzm/InstructNA_new/data/function_SELEX/IFN-gama/functional_label/label_data.csv"
encoder_model_path = "/home/hanlab/2022_zzm/InstructNA_new/output/model_save/DNABERT_3mers/function_SELEX/IFN-gama/encoder"
decoder_model_path = "/home/hanlab/2022_zzm/InstructNA_new/output/model_save/DNABERT_3mers/function_SELEX/IFN-gama/decoder"
BO_output_dir = "./"
search_r = 5

max_flag = True      # 想要开启 --max
use_filter_flag = False  # 想要关闭 --use_filter

HC_HEBO_batchsize = 2
model_down_dim = 8
f_primer = "CGGTTCAG"
r_primer = "CTGAACCG"

# === 构建命令 ===
cmd = [
    "python", "/home/hanlab/2022_zzm/InstructNA_new/examples/DNABERT_3mers/single_HC-HEBO_inference.py",
    "--seq_act_path", seq_act_path,
    "--encoder_model_path", encoder_model_path,
    "--decoder_model_path", decoder_model_path,
    "--BO_output_dir", BO_output_dir,
    "--search_r", str(search_r),
    "--HC_HEBO_batchsize", str(HC_HEBO_batchsize),
    "--model_down_dim", str(model_down_dim),
    "--f_primer", f_primer,
    "--r_primer", r_primer
]

# —— 根据布尔值动态添加 store_true 参数 ——
if max_flag:
    cmd.append("--max")

if use_filter_flag:
    cmd.append("--use_filter")

# === 执行命令 ===
print("🚀 正在运行 inference ...")
subprocess.run(cmd)
print("✅ 运行结束！")
