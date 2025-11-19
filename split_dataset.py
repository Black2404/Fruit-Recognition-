# split_data.py
import splitfolders

# Thư mục dữ liệu gốc, bên trong có subfolder theo từng class
input_folder = "dataset"

# Thư mục sau khi chia
output_folder = "data_split_2"

# Chia tỉ lệ 70% train, 15% val, 15% test
splitfolders.ratio(input_folder, output=output_folder, seed=42, ratio=(0.7, 0.15, 0.15))
