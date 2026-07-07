# CelebA Data Instructions

Raw CelebA files are not included in this repository because the dataset is large and has its own distribution terms.

Download CelebA manually from:

- CelebA official page: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
- Google Drive mirror: https://drive.google.com/drive/folders/0B7EVK8r0v71pQ-1tLS0tNkZvcFo

Place the files like this:

```text
data/
└── archive/
    ├── list_attr_celeba.csv
    ├── list_bbox_celeba.csv
    ├── list_eval_partition.csv
    ├── list_landmarks_align_celeba.csv
    └── img_align_celeba/
```

Notes:

- The CSV annotation files are required for preprocessing, training, testing, and visualization.
- `img_align_celeba/` contains about 200k aligned face images and must stay local.
- Do not commit raw data files. Under `data/`, only `README.md` and `.gitignore` should be tracked.

## 中文说明

CelebA 原始数据不随仓库上传，原因是数据集体积较大，并且有单独的数据分发规则。

手动下载入口：

- CelebA 官方页面：https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
- Google Drive 镜像：https://drive.google.com/drive/folders/0B7EVK8r0v71pQ-1tLS0tNkZvcFo

下载后按下面的结构放置：

```text
data/
└── archive/
    ├── list_attr_celeba.csv
    ├── list_bbox_celeba.csv
    ├── list_eval_partition.csv
    ├── list_landmarks_align_celeba.csv
    └── img_align_celeba/
```

说明：

- 四个 CSV 标注文件是预处理、训练、测试和可视化的必需文件。
- `img_align_celeba/` 包含约 20 万张对齐后的人脸图像，只保留在本地。
- 禁止提交原始数据文件。`data/` 目录下只应跟踪 `README.md` 和 `.gitignore`。
