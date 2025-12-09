【文本分类Web应用】

## 🌐 启动方法

```bash
python app/web_app.py
```

然后在浏览器打开: http://localhost:5000

## ✨ 功能特点

- 🔄 **模型切换**: 一键切换Standard和Recurrent Transformer
- 💭 **情感分类**: 正面/负面
- 🏷️ **领域分类**: 电影评论/在线购物/本地商家
- 📊 **可视化**: 彩色概率条形图
- 📱 **响应式**: 支持手机和平板

## 🎯 快速使用

1. 点击选择模型（Standard 或 Recurrent）
2. 输入或粘贴文本
3. 点击"Classify Text"按钮
4. 查看分类结果和概率分布

## ⌨️ 键盘快捷键

- `Ctrl+Enter` - 执行分类
- `Ctrl+K` - 聚焦输入框
- `Escape` - 清除内容

## 📚 完整文档

详细使用指南请查看: [WEB_APP_GUIDE.md](../WEB_APP_GUIDE.md)

## 🔧 技术架构

- **后端**: Flask + PyTorch
- **前端**: HTML5 + CSS3 + JavaScript
- **模型**: Baseline & Recurrent Transformers

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

