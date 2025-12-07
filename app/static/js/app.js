// Text Classification Web App - Frontend JavaScript

// Global state
let selectedModel = 'baseline';

// DOM Elements
const textInput = document.getElementById('text-input');
const classifyBtn = document.getElementById('classify-btn');
const clearBtn = document.getElementById('clear-btn');
const loading = document.getElementById('loading');
const results = document.getElementById('results');
const error = document.getElementById('error');
const errorMessage = document.getElementById('error-message');

// Model buttons
const baselineBtn = document.getElementById('btn-baseline');
const recurrentBtn = document.getElementById('btn-recurrent');

// Example buttons
const exampleButtons = document.querySelectorAll('.example-btn');

// Initialize
document.addEventListener('DOMContentLoaded', function() {
    console.log('Text Classification App initialized');
    
    // Set up event listeners
    setupEventListeners();
    
    // Focus on text input
    textInput.focus();
});

function setupEventListeners() {
    // Model selection
    baselineBtn.addEventListener('click', () => selectModel('baseline'));
    recurrentBtn.addEventListener('click', () => selectModel('recurrent'));
    
    // Classify button
    classifyBtn.addEventListener('click', classifyText);
    
    // Clear button
    clearBtn.addEventListener('click', clearAll);
    
    // Enter key in textarea
    textInput.addEventListener('keydown', function(e) {
        if (e.ctrlKey && e.key === 'Enter') {
            classifyText();
        }
    });
    
    // Example buttons
    exampleButtons.forEach(btn => {
        btn.addEventListener('click', function() {
            const exampleText = this.getAttribute('data-text');
            textInput.value = exampleText;
            textInput.focus();
        });
    });
}

function selectModel(modelType) {
    selectedModel = modelType;
    
    // Update button states
    if (modelType === 'baseline') {
        baselineBtn.classList.add('active');
        recurrentBtn.classList.remove('active');
    } else {
        recurrentBtn.classList.add('active');
        baselineBtn.classList.remove('active');
    }
    
    console.log('Selected model:', modelType);
}

function clearAll() {
    textInput.value = '';
    hideResults();
    hideError();
    textInput.focus();
}

function showLoading() {
    loading.classList.remove('hidden');
    results.classList.add('hidden');
    error.classList.add('hidden');
    classifyBtn.disabled = true;
}

function hideLoading() {
    loading.classList.add('hidden');
    classifyBtn.disabled = false;
}

function showResults() {
    results.classList.remove('hidden');
}

function hideResults() {
    results.classList.add('hidden');
}

function showError(message) {
    errorMessage.textContent = message;
    error.classList.remove('hidden');
    results.classList.add('hidden');
}

function hideError() {
    error.classList.add('hidden');
}

async function classifyText() {
    const text = textInput.value.trim();
    
    // Validation
    if (!text) {
        showError('Please enter some text to classify');
        return;
    }
    
    console.log('Starting classification...');
    console.log('Text:', text);
    console.log('Selected model:', selectedModel);
    
    // Show loading
    showLoading();
    hideError();
    
    try {
        // Make API request
        console.log('Sending request to /classify');
        const response = await fetch('/classify', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                text: text,
                model_type: selectedModel
            })
        });
        
        console.log('Response status:', response.status);
        
        if (!response.ok) {
            const errorData = await response.json();
            console.error('Error response:', errorData);
            throw new Error(errorData.error || 'Classification failed');
        }
        
        const data = await response.json();
        console.log('Classification result:', data);
        
        // Display results
        displayResults(data);
        
    } catch (err) {
        console.error('Error:', err);
        showError(err.message || 'An error occurred during classification');
    } finally {
        hideLoading();
    }
}

function displayResults(data) {
    // Update model badge
    const modelBadge = document.getElementById('model-badge');
    modelBadge.textContent = data.model_type === 'baseline' 
        ? 'Standard Transformer' 
        : 'Recurrent Transformer';
    
    // Display input text
    document.getElementById('result-text').textContent = data.text;
    
    // Display sentiment results
    displaySentimentResults(data.sentiment);
    
    // Display domain results
    displayDomainResults(data.domain);
    
    // Show results section
    showResults();
    
    // Scroll to results
    results.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function displaySentimentResults(sentiment) {
    // Label and confidence
    document.getElementById('sentiment-label').textContent = sentiment.label;
    document.getElementById('sentiment-confidence').textContent = sentiment.confidence + '%';
    
    // Probability bars
    const negProb = sentiment.probabilities['Negative'];
    const posProb = sentiment.probabilities['Positive'];
    
    document.getElementById('prob-negative').style.width = negProb + '%';
    document.getElementById('prob-positive').style.width = posProb + '%';
    
    document.getElementById('val-negative').textContent = negProb.toFixed(1) + '%';
    document.getElementById('val-positive').textContent = posProb.toFixed(1) + '%';
}

function displayDomainResults(domain) {
    // Label and confidence
    document.getElementById('domain-label').textContent = formatDomainLabel(domain.label);
    document.getElementById('domain-confidence').textContent = domain.confidence + '%';
    
    // Probability bars
    const movieProb = domain.probabilities['movie_review'];
    const shoppingProb = domain.probabilities['online_shopping'];
    const businessProb = domain.probabilities['local_business_review'];
    
    document.getElementById('prob-movie').style.width = movieProb + '%';
    document.getElementById('prob-shopping').style.width = shoppingProb + '%';
    document.getElementById('prob-business').style.width = businessProb + '%';
    
    document.getElementById('val-movie').textContent = movieProb.toFixed(1) + '%';
    document.getElementById('val-shopping').textContent = shoppingProb.toFixed(1) + '%';
    document.getElementById('val-business').textContent = businessProb.toFixed(1) + '%';
}

function formatDomainLabel(label) {
    const labelMap = {
        'movie_review': 'Movie Review',
        'online_shopping': 'Online Shopping',
        'local_business_review': 'Local Business Review'
    };
    return labelMap[label] || label;
}

// Keyboard shortcuts
document.addEventListener('keydown', function(e) {
    // Ctrl+K to focus on input
    if (e.ctrlKey && e.key === 'k') {
        e.preventDefault();
        textInput.focus();
    }
    
    // Escape to clear
    if (e.key === 'Escape') {
        clearAll();
    }
});

// Add visual feedback for loading
window.addEventListener('beforeunload', function(e) {
    if (loading.classList.contains('hidden') === false) {
        e.preventDefault();
        e.returnValue = '';
    }
});

console.log('App.js loaded successfully');
console.log('Keyboard shortcuts:');
console.log('  Ctrl+Enter: Classify text');
console.log('  Ctrl+K: Focus on input');
console.log('  Escape: Clear all');

