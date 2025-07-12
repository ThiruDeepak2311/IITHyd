# VirConvNet: Virtual Sparse Convolution for Multimodal 3D Object Detection 🚘🧠🔭

VirConvNet is an advanced multimodal 3D object detection framework that integrates **LiDAR** and **RGB camera** data to deliver **real-time, robust, and noise-resilient** detection performance. Designed for autonomous driving applications, it introduces **Stochastic Voxel Discard (StVD)** and **Noise-Resistant Submanifold Convolution (NRConv)** to optimize computational efficiency and improve accuracy, even in sparse or noisy environments.

### 🏆 Highlights

* **3.42% Improvement** in 3D Average Precision on the KITTI dataset (Moderate class)
* **2x Faster Inference** via voxel density reduction using StVD
* **Robust to Noise** using SE Blocks and Pointwise Spatial Attention
* Modular, flexible architecture compatible with standard LiDAR and RGB pipelines

---

## 🚀 Key Innovations

### 1. **Stochastic Voxel Discard (StVD)**

* **Input StVD**: Reduces voxel density by 90% using bin-based sampling based on distance.
* **Layer StVD**: Further regularizes training via 15% random voxel discarding.
* **Result**: Doubles the processing speed without loss in accuracy.

### 2. **Noise-Resistant Submanifold Convolution (NRConv)**

* Combines 3D submanifold convolutions with **SE Blocks** and **Dilated Convolutions**
* Addresses depth completion noise from RGB images
* Captures both **geometric** and **contextual** features

### 3. **Squeeze-and-Excitation (SE) Blocks**

* Channel-wise feature recalibration using global pooling and gating
* Learns to prioritize informative features and suppress irrelevant noise

### 4. **Pointwise Spatial Attention (PSA)**

* Applies attention at the point level, refining features based on contextual importance
* Complements SE blocks for full spatial-channel attention

---

## 🧠 Architecture Overview

```
Input: LiDAR + Virtual RGB Points
      ↓
StVD Layer (Input)
      ↓
VirConv Blocks (NRConv + SE + PSA + Dilated Conv)
      ↓
3D SparseConv Backbone
      ↓
RPN + ROI Head (Detection Head)
      ↓
Output: 3D Bounding Boxes
```

---

## 📊 Dataset & Evaluation

* **Dataset**: [KITTI 3D Object Detection Benchmark](http://www.cvlibs.net/datasets/kitti/)
* **Classes**: Car, Pedestrian, Cyclist
* **Metrics**: 3D AP (R40), BEV AP (R40), IoU thresholds: 0.7 (Car), 0.5 (Others)

| Model Variant           | 3D AP (Moderate) | Speed  | Notes           |
| ----------------------- | ---------------- | ------ | --------------- |
| Voxel-RCNN (Baseline)   | 74.25%           | 1x     | Standard LiDAR  |
| VirConv-L + SE + PSA    | **77.67%**       | **2x** | Fast & Accurate |
| VirConv-T (Late Fusion) | **79.21%**       | 1.5x   | High Precision  |

---

## 🛠️ Setup & Installation

```bash
# Clone the repo
git clone https://github.com/yourusername/VirConvNet.git
cd VirConvNet

# Install dependencies
pip install -r requirements.txt

# Download KITTI dataset and organize under `data/kitti/`
```

---

## 🧪 Training & Evaluation

```bash
# Train on KITTI
python train.py --config configs/virconv_l.yaml

# Evaluate on validation set
python eval.py --config configs/virconv_l.yaml --checkpoint path/to/model.pth
```

---

## 📈 Ablation Studies

| Experiment | Description                  | 3D AP (Moderate) |
| ---------- | ---------------------------- | ---------------- |
| Baseline   | VirConvNet without attention | 74.25%           |
| + SE Block | Channel-wise recalibration   | 75.8%            |
| + SE + PSA | Full attention stack         | **77.67%**       |

---

## 📚 References

* [Squeeze-and-Excitation Networks (Hu et al., 2018)](https://arxiv.org/abs/1709.01507)
* [PSANet: Point-wise Spatial Attention Network (Zhao et al., 2018)](https://openaccess.thecvf.com/content_ECCV_2018/papers/Hengshuang_Zhao_PSANet_Point-wise_Spatial_ECCV_2018_paper.pdf)
* [VirConvNet Original Paper (CVPR 2023)](https://arxiv.org/abs/2303.02083)

---

## 👨‍💻 Authors

* Deepak T (SNU Chennai)
* Under the guidance of **Prof. C. Krishna Mohan** (CS6140: Video Content Analysis)

---

## 📢 Citation

If you use this work, please cite:

```
@inproceedings{virconvnet2023,
  title={Virtual Sparse Convolution for Multimodal 3D Object Detection},
  author={Wu, Hai and Wen, Chenglu and Shi, Shaoshuai and Li, Xin and Wang, Cheng},
  booktitle={CVPR},
  year={2023}
}
```
