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

Here is the English translation formatted in a Markdown code block for you to copy:

## 🎨 SwiGLU Interactive Demo

Interactive visualization of the Swish-Gated Linear Unit activation function.

### Launch Instructions

```bash
cd app/SwiGLU_demo
python -m http.server 8080
````

Then open http://localhost:8080 in your browser.

### Features

  - **Parameter Tuning**: Real-time adjustment of parameters β, W₁, W₂, b₁, and b₂ using sliders.
  - **Curve Visualization**: View changes in SwiGLU, Swish, and gradient curves.
  - **Comparative Analysis**: Compare against ReLU and GELU activation functions.
  - **Component Decomposition**: Display individual components of the Swish gating and linear paths.

### Parameter Descriptions

| Parameter | Description |
|-----------|-------------|
| β (Beta) | Controls the sharpness of the Swish curve (β→0 becomes linear, β→∞ approaches ReLU). |
| W₁ Weight | Scaling for the gated activation path. |
| W₂ Weight | Scaling for the linear projection path. |
| b₁, b₂ Bias | Bias terms for each path. |

