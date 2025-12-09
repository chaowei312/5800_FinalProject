# Text Classification Web Application

## How to Launch

```bash
python web_app.py
````

Then open the browser at: [http://localhost:5000](http://localhost:5000)

## Features

* Model switching between Standard and Recurrent Transformers
* Sentiment classification (positive / negative)
* Domain classification (movie review / online shopping / local business)
* Probability visualization with color-coded bar charts
* Responsive UI for desktop and mobile devices

## Quick Start

1. Select a model (Standard or Recurrent)
2. Enter or paste text into the input box
3. Click the "Classify Text" button
4. View the predicted labels and probability distributions

## Keyboard Shortcuts

* `Ctrl + Enter` — Run classification
* `Ctrl + K` — Focus the text input field
* `Escape` — Clear the input


## Technical Stack

* Backend: Flask + PyTorch
* Frontend: HTML5, CSS3, JavaScript
* Models: Baseline Transformer and Recurrent Transformer

## 🎨 SwiGLU Interactive Demo

Interactive visualization of Swish-Gated Linear Unit activation function.

### 启动方法

```bash
cd app/SwiGLU_demo
python -m http.server 8080
```

然后在浏览器打开: http://localhost:8080

### 功能特点

- **参数调节**: 拖动滑块实时调整 β、W₁、W₂、b₁、b₂ 参数
- **曲线可视化**: 查看 SwiGLU、Swish 和梯度曲线变化
- **对比分析**: 与 ReLU、GELU 激活函数对比
- **组件分解**: 显示 Swish 门控和线性路径的分量

### 参数说明

| 参数 | 描述 |
|------|------|
| β (Beta) | 控制 Swish 曲线的锐度 (β→0 变线性, β→∞ 接近ReLU) |
| W₁ Weight | 门控激活路径的缩放 |
| W₂ Weight | 线性投影路径的缩放 |
| b₁, b₂ Bias | 各路径的偏置项 |

