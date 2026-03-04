import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# 确保使用的路径存在并且结构清晰
image_dir = r'E:\data\predit k\only water\crop'  # 图像文件所在目录
output_dir = r'E:\data\predit k\only water\code\datasetone3'  # 分割后数据集保存目录
label_file = r'E:\data\predit k\only water\water.xlsx'  # 标签文件路径

# 加载标签数据
labels_df = pd.read_excel(label_file, engine='openpyxl')

# 设置数据集划分比例
train_ratio = 0.6
valid_ratio = 0.2
test_ratio = 0.2

# 验证比例之和为1
assert train_ratio + valid_ratio + test_ratio == 1, "划分比例之和必须为1"

train_data, test_data = train_test_split(
    labels_df, test_size=test_ratio, random_state=42
)

# 第二步：从训练集中划分出训练子集和验证子集
train_subset, val_subset = train_test_split(
    train_data, test_size=valid_ratio / (1 - test_ratio), random_state=42
)



# 对每个子集按index列排序
train_subset = train_subset.sort_values(by='index').reset_index(drop=True)
val_subset = val_subset.sort_values(by='index').reset_index(drop=True)
test_data = test_data.sort_values(by='index').reset_index(drop=True)
# 创建输出目录结构
os.makedirs(output_dir, exist_ok=True)
train_dir = os.path.join(output_dir, 'train')
valid_dir = os.path.join(output_dir, 'valid')
test_dir = os.path.join(output_dir, 'test')

for directory in [train_dir, valid_dir, test_dir]:
    os.makedirs(directory, exist_ok=True)

# 复制图像文件到对应目录
def copy_images(data, dest_dir):
    for index in data['index']:
        src_image_path = os.path.join(image_dir, f"{index}.jpg")
        dest_image_path = os.path.join(dest_dir, f"{index}.jpg")
        shutil.copy(src_image_path, dest_image_path)

copy_images(train_subset, train_dir)
copy_images(val_subset, valid_dir)
copy_images(test_data, test_dir)

# 保存标签文件
train_subset.to_excel(os.path.join(train_dir, 'trainlabels.xlsx'), index=False)
val_subset.to_excel(os.path.join(valid_dir, 'validlabels.xlsx'), index=False)
test_data.to_excel(os.path.join(test_dir, 'testlabels.xlsx'), index=False)

print("数据集划分完成，文件已保存。")
