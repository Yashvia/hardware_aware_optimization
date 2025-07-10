# 🧠 Hardware-Aware Model Optimization for Edge Inference

## 📌 Overview

This project demonstrates the end-to-end development of a lightweight, hardware-efficient image classification pipeline optimized for deployment on resource-constrained edge devices. We use **PyTorch** for model development, **ONNX** for interoperability, and **OpenCV DNN** for simulating real-world edge inference.

---

## 🎯 Objectives

- Train a Convolutional Neural Network (CNN) using PyTorch on CIFAR-10.
- Convert the model to ONNX for hardware-friendly deployment.
- Apply model pruning to reduce memory and latency.
- Use OpenCV’s DNN module to simulate edge inference.
- Analyze trade-offs in accuracy, latency, and model size.
- Provide a reproducible pipeline on Google Colab using free tools only.

---

## 🛠️ Tools & Technologies

| Component       | Usage                                    |
|-----------------|------------------------------------------|
| PyTorch         | Model training and pruning                |
| ONNX            | Interoperable model format for deployment |
| OpenCV DNN      | Hardware-like inference benchmarking      |
| Google Colab    | Training + evaluation (free GPU/CPU)      |
| Matplotlib      | Visualization                             |
| NumPy, Pandas   | Data analysis                             |

---

## 🗂️ Project Structure

```
hardware_aware_optimization/
├── models/
│   ├── base_model.pt
│   ├── model.onnx
│   ├── model_pruned.pt
│   └── model_pruned.onnx
├── data/
│   └── cifar10/      # Managed by torchvision
├── results/
│   ├── accuracy_plot.png
│   ├── latency_comparison.png
│   └── size_chart.csv
├── README.md
└── requirements.txt
```

---

## 🧪 How to Run (Google Colab Recommended)

1. **Train the model:**  
   Run `1_train_model.ipynb` to train and save as `base_model.pt`.
2. **Convert to ONNX:**  
   Run `2_convert_to_onnx.ipynb` to export to `model.onnx`.
3. **Prune and export the model:**  
   Run `3_pruning_and_export.ipynb` to prune, fine-tune, and export as `model_pruned.onnx`.
4. **Benchmark and Analyze:**  
   Run `4_inference_and_analysis.ipynb` to benchmark ONNX inference (FP32 & pruned) and plot size/accuracy/latency results.

---

## 📈 Results Summary

| Model        | Accuracy (%) | Size (MB) | Inference Time (ms) |
|--------------|-------------|-----------|---------------------|
| FP32         | 81.6        | 8.369335  | 3.293362            |
| Pruned FP32  | 81.8        | 8.369335  | 3.098687            |



---

## 💡 Pruning : Impact on ASIC/FPGA

**Pruning** removes redundant weights/filters, reducing model size and compute requirements. On ASICs and FPGAs, this can:
- Lower the number of required multipliers/adders, reducing silicon area and power consumption.
- Allow for smaller on-chip memory, lowering latency and energy use.
- Enable higher throughput if parallelism is re-allocated to the remaining active weights.

**Quantization** (e.g., INT8 vs FP32) reduces bit-width, which:
- Reduces memory footprint and bandwidth, crucial for edge devices.
- Enables use of smaller, faster arithmetic units (multipliers/adders) in custom hardware.
- Allows higher parallelism within the same silicon area or power budget.

**Trade-offs:** Aggressive pruning or quantization can degrade model accuracy. Some hardware (especially FPGAs) can efficiently map sparse/pruned models, but not all toolchains support this natively as of now.

---

## 🚀 Future Work

- **Deploy to custom hardware:** Export and deploy the ONNX model to platforms such as NVIDIA Jetson, Google Coral, or FPGA/ASIC prototypes (e.g., via hls4ml or Vitis AI).

---

## 📦 Installation

You can run everything in Colab without local setup.  
If running locally:

```bash
pip install torch torchvision onnx opencv-python matplotlib seaborn pandas
```
Run the notebooks in order.
---

## 🏁 License

MIT