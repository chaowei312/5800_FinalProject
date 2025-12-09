// SwiGLU Interactive Demo
// ========================

// Chart.js configuration
Chart.defaults.color = '#8b9eb5';
Chart.defaults.borderColor = '#2a3650';
Chart.defaults.font.family = 'Outfit, sans-serif';

// Parameters
let params = {
    beta: 1.0,
    w1: 1.0,
    w2: 1.0,
    b1: 0.0,
    b2: 0.0,
    showComponents: true,
    showComparison: false
};

// Generate x values
const xMin = -5;
const xMax = 5;
const numPoints = 200;
const xValues = Array.from({ length: numPoints }, (_, i) => xMin + (i / (numPoints - 1)) * (xMax - xMin));

// Activation functions
function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
}

function swish(x, beta = 1.0) {
    return x * sigmoid(beta * x);
}

function swishDerivative(x, beta = 1.0) {
    const sig = sigmoid(beta * x);
    return sig + x * beta * sig * (1 - sig);
}

function relu(x) {
    return Math.max(0, x);
}

function gelu(x) {
    // Approximation of GELU
    return 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * Math.pow(x, 3))));
}

function swiglu(x, beta, w1, w2, b1, b2) {
    const gateInput = w1 * x + b1;
    const linearInput = w2 * x + b2;
    return swish(gateInput, beta) * linearInput;
}

function swigluDerivative(x, beta, w1, w2, b1, b2) {
    // Numerical derivative
    const h = 0.0001;
    return (swiglu(x + h, beta, w1, w2, b1, b2) - swiglu(x - h, beta, w1, w2, b1, b2)) / (2 * h);
}

// Chart instances
let swigluChart, swishChart, gradientChart;

// Color palette
const colors = {
    swiglu: '#06d6a0',
    swish: '#118ab2',
    linear: '#ffd166',
    gate: '#ef476f',
    relu: '#ff6b6b',
    gelu: '#c084fc',
    gradient: '#06d6a0',
    swishGrad: '#118ab2'
};

// Initialize charts
function initCharts() {
    // Main SwiGLU chart
    const swigluCtx = document.getElementById('swiglu-chart').getContext('2d');
    swigluChart = new Chart(swigluCtx, {
        type: 'line',
        data: {
            labels: xValues,
            datasets: []
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false,
            },
            plugins: {
                legend: {
                    position: 'top',
                    labels: {
                        usePointStyle: true,
                        padding: 20,
                        font: { size: 12 }
                    }
                },
                tooltip: {
                    backgroundColor: '#1a2234',
                    borderColor: '#2a3650',
                    borderWidth: 1,
                    titleFont: { weight: 'normal' },
                    callbacks: {
                        title: (items) => `x = ${parseFloat(items[0].label).toFixed(2)}`,
                        label: (item) => `${item.dataset.label}: ${item.parsed.y.toFixed(4)}`
                    }
                }
            },
            scales: {
                x: {
                    type: 'linear',
                    title: {
                        display: true,
                        text: 'Input (x)',
                        font: { size: 12 }
                    },
                    grid: { color: '#1a2234' },
                    ticks: {
                        callback: (val) => val.toFixed(1)
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Output',
                        font: { size: 12 }
                    },
                    grid: { color: '#1a2234' }
                }
            }
        }
    });

    // Swish detail chart
    const swishCtx = document.getElementById('swish-chart').getContext('2d');
    swishChart = new Chart(swishCtx, {
        type: 'line',
        data: {
            labels: xValues,
            datasets: []
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                    labels: {
                        usePointStyle: true,
                        padding: 15,
                        font: { size: 11 }
                    }
                },
                tooltip: {
                    backgroundColor: '#1a2234',
                    borderColor: '#2a3650',
                    borderWidth: 1
                }
            },
            scales: {
                x: {
                    type: 'linear',
                    title: { display: true, text: 'Input (x)', font: { size: 11 } },
                    grid: { color: '#1a2234' }
                },
                y: {
                    title: { display: true, text: 'Swish(x)', font: { size: 11 } },
                    grid: { color: '#1a2234' }
                }
            }
        }
    });

    // Gradient chart
    const gradientCtx = document.getElementById('gradient-chart').getContext('2d');
    gradientChart = new Chart(gradientCtx, {
        type: 'line',
        data: {
            labels: xValues,
            datasets: []
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                    labels: {
                        usePointStyle: true,
                        padding: 15,
                        font: { size: 11 }
                    }
                },
                tooltip: {
                    backgroundColor: '#1a2234',
                    borderColor: '#2a3650',
                    borderWidth: 1
                }
            },
            scales: {
                x: {
                    type: 'linear',
                    title: { display: true, text: 'Input (x)', font: { size: 11 } },
                    grid: { color: '#1a2234' }
                },
                y: {
                    title: { display: true, text: 'Gradient', font: { size: 11 } },
                    grid: { color: '#1a2234' }
                }
            }
        }
    });

    updateCharts();
}

// Update all charts
function updateCharts() {
    updateSwigluChart();
    updateSwishChart();
    updateGradientChart();
}

function updateSwigluChart() {
    const datasets = [];

    // Main SwiGLU curve
    const swigluData = xValues.map(x => ({
        x: x,
        y: swiglu(x, params.beta, params.w1, params.w2, params.b1, params.b2)
    }));
    
    datasets.push({
        label: 'SwiGLU',
        data: swigluData,
        borderColor: colors.swiglu,
        backgroundColor: colors.swiglu + '20',
        borderWidth: 3,
        fill: false,
        tension: 0.4,
        pointRadius: 0
    });

    if (params.showComponents) {
        // Swish component (gate)
        const swishData = xValues.map(x => ({
            x: x,
            y: swish(params.w1 * x + params.b1, params.beta)
        }));
        
        datasets.push({
            label: 'Swish(W₁x + b₁)',
            data: swishData,
            borderColor: colors.swish,
            borderWidth: 2,
            borderDash: [5, 5],
            fill: false,
            tension: 0.4,
            pointRadius: 0
        });

        // Linear component
        const linearData = xValues.map(x => ({
            x: x,
            y: params.w2 * x + params.b2
        }));
        
        datasets.push({
            label: 'W₂x + b₂',
            data: linearData,
            borderColor: colors.linear,
            borderWidth: 2,
            borderDash: [2, 2],
            fill: false,
            tension: 0,
            pointRadius: 0
        });
    }

    if (params.showComparison) {
        // ReLU
        const reluData = xValues.map(x => ({
            x: x,
            y: relu(x)
        }));
        
        datasets.push({
            label: 'ReLU',
            data: reluData,
            borderColor: colors.relu,
            borderWidth: 1.5,
            borderDash: [8, 4],
            fill: false,
            tension: 0,
            pointRadius: 0
        });

        // GELU
        const geluData = xValues.map(x => ({
            x: x,
            y: gelu(x)
        }));
        
        datasets.push({
            label: 'GELU',
            data: geluData,
            borderColor: colors.gelu,
            borderWidth: 1.5,
            borderDash: [8, 4],
            fill: false,
            tension: 0.4,
            pointRadius: 0
        });
    }

    // Zero line
    datasets.push({
        label: 'y = 0',
        data: xValues.map(x => ({ x: x, y: 0 })),
        borderColor: '#3a4a65',
        borderWidth: 1,
        borderDash: [2, 4],
        fill: false,
        pointRadius: 0,
        hidden: true
    });

    swigluChart.data.datasets = datasets;
    swigluChart.update('none');
}

function updateSwishChart() {
    const datasets = [];

    // Current Swish with beta
    const swishData = xValues.map(x => ({
        x: x,
        y: swish(x, params.beta)
    }));
    
    datasets.push({
        label: `Swish (β=${params.beta.toFixed(2)})`,
        data: swishData,
        borderColor: colors.swish,
        backgroundColor: colors.swish + '20',
        borderWidth: 2.5,
        fill: true,
        tension: 0.4,
        pointRadius: 0
    });

    // Reference curves
    if (params.beta !== 1.0) {
        const swish1Data = xValues.map(x => ({
            x: x,
            y: swish(x, 1.0)
        }));
        
        datasets.push({
            label: 'Swish (β=1.0)',
            data: swish1Data,
            borderColor: '#5c6d83',
            borderWidth: 1.5,
            borderDash: [4, 4],
            fill: false,
            tension: 0.4,
            pointRadius: 0
        });
    }

    // Sigmoid component (scaled)
    const sigmoidData = xValues.map(x => ({
        x: x,
        y: sigmoid(params.beta * x)
    }));
    
    datasets.push({
        label: `σ(${params.beta.toFixed(1)}x)`,
        data: sigmoidData,
        borderColor: colors.gate,
        borderWidth: 1.5,
        borderDash: [3, 3],
        fill: false,
        tension: 0.4,
        pointRadius: 0
    });

    swishChart.data.datasets = datasets;
    swishChart.update('none');
}

function updateGradientChart() {
    const datasets = [];

    // SwiGLU gradient
    const swigluGradData = xValues.map(x => ({
        x: x,
        y: swigluDerivative(x, params.beta, params.w1, params.w2, params.b1, params.b2)
    }));
    
    datasets.push({
        label: "SwiGLU'",
        data: swigluGradData,
        borderColor: colors.gradient,
        backgroundColor: colors.gradient + '15',
        borderWidth: 2.5,
        fill: true,
        tension: 0.4,
        pointRadius: 0
    });

    // Swish gradient
    const swishGradData = xValues.map(x => ({
        x: x,
        y: swishDerivative(x, params.beta)
    }));
    
    datasets.push({
        label: "Swish'",
        data: swishGradData,
        borderColor: colors.swishGrad,
        borderWidth: 2,
        borderDash: [5, 3],
        fill: false,
        tension: 0.4,
        pointRadius: 0
    });

    if (params.showComparison) {
        // ReLU gradient (step function)
        const reluGradData = xValues.map(x => ({
            x: x,
            y: x > 0 ? 1 : 0
        }));
        
        datasets.push({
            label: "ReLU'",
            data: reluGradData,
            borderColor: colors.relu,
            borderWidth: 1.5,
            borderDash: [6, 3],
            fill: false,
            tension: 0,
            pointRadius: 0
        });
    }

    gradientChart.data.datasets = datasets;
    gradientChart.update('none');
}

// Event handlers
function setupEventListeners() {
    // Slider handlers
    const sliders = {
        'beta-slider': { param: 'beta', display: 'beta-value' },
        'w1-slider': { param: 'w1', display: 'w1-value' },
        'w2-slider': { param: 'w2', display: 'w2-value' },
        'b1-slider': { param: 'b1', display: 'b1-value' },
        'b2-slider': { param: 'b2', display: 'b2-value' }
    };

    Object.entries(sliders).forEach(([sliderId, config]) => {
        const slider = document.getElementById(sliderId);
        const display = document.getElementById(config.display);

        slider.addEventListener('input', (e) => {
            const value = parseFloat(e.target.value);
            params[config.param] = value;
            display.textContent = value.toFixed(2);
            display.classList.add('updating');
            setTimeout(() => display.classList.remove('updating'), 300);
            updateCharts();
        });
    });

    // Checkbox handlers
    document.getElementById('show-components').addEventListener('change', (e) => {
        params.showComponents = e.target.checked;
        updateCharts();
    });

    document.getElementById('show-comparison').addEventListener('change', (e) => {
        params.showComparison = e.target.checked;
        updateCharts();
    });

    // Reset button
    document.getElementById('reset-btn').addEventListener('click', resetParams);
}

function resetParams() {
    params = {
        beta: 1.0,
        w1: 1.0,
        w2: 1.0,
        b1: 0.0,
        b2: 0.0,
        showComponents: true,
        showComparison: false
    };

    // Update sliders and displays
    document.getElementById('beta-slider').value = params.beta;
    document.getElementById('beta-value').textContent = params.beta.toFixed(2);
    
    document.getElementById('w1-slider').value = params.w1;
    document.getElementById('w1-value').textContent = params.w1.toFixed(2);
    
    document.getElementById('w2-slider').value = params.w2;
    document.getElementById('w2-value').textContent = params.w2.toFixed(2);
    
    document.getElementById('b1-slider').value = params.b1;
    document.getElementById('b1-value').textContent = params.b1.toFixed(2);
    
    document.getElementById('b2-slider').value = params.b2;
    document.getElementById('b2-value').textContent = params.b2.toFixed(2);

    document.getElementById('show-components').checked = params.showComponents;
    document.getElementById('show-comparison').checked = params.showComparison;

    updateCharts();
}

// Initialize on load
document.addEventListener('DOMContentLoaded', () => {
    initCharts();
    setupEventListeners();
});

