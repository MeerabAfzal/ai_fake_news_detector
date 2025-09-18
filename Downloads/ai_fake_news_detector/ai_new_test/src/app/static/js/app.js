// Fake News Detector - Enhanced Interactivity

document.addEventListener('DOMContentLoaded', function() {
    // Initialize the application
    initApp();
    
    // Setup event listeners
    setupEventListeners();
});

// Initialize application
function initApp() {
    // Initialize history from localStorage
    window.analysisHistory = JSON.parse(localStorage.getItem('analysisHistory')) || [];
    
    // Update history display on load
    updateHistoryDisplay();
    
    // Check if dark mode is enabled
    const darkModeEnabled = localStorage.getItem('darkMode') === 'true';
    if (darkModeEnabled) {
        document.body.classList.add('dark-mode');
        if (document.getElementById('dark-mode-toggle')) {
            document.getElementById('dark-mode-toggle').checked = true;
        }
    }
    
    // Create toast container if it doesn't exist
    if (!document.querySelector('.toast-container')) {
        const toastContainer = document.createElement('div');
        toastContainer.className = 'toast-container';
        document.body.appendChild(toastContainer);
    }
    
    // Show welcome toast
    showToast('Welcome!', 'Analyze news articles to detect fake news', 'info');
}

// Setup event listeners
function setupEventListeners() {
    // Form submission
    const newsForm = document.getElementById('news-form');
    if (newsForm) {
        newsForm.addEventListener('submit', handleFormSubmit);
    }
    
    // Sample article clicks
    document.querySelectorAll('.sample-article').forEach(article => {
        article.addEventListener('click', function() {
            const text = this.getAttribute('data-article');
            document.getElementById('news-text').value = text;
            document.getElementById('news-form').dispatchEvent(new Event('submit'));
            
            // Show toast notification
            showToast('Sample Loaded', 'Sample article has been loaded for analysis', 'info');
        });
    });
    
    // Clear history button
    const clearHistoryBtn = document.getElementById('clear-history-btn');
    if (clearHistoryBtn) {
        clearHistoryBtn.addEventListener('click', function() {
            // Clear history with animation
            const historyItems = document.querySelectorAll('.history-item');
            
            // If no history items, just return
            if (historyItems.length === 0) return;
            
            // Add fade-out animation to each item
            historyItems.forEach((item, index) => {
                setTimeout(() => {
                    item.style.opacity = '0';
                    item.style.transform = 'translateX(20px)';
                }, index * 100);
            });
            
            // Wait for animations to complete
            setTimeout(() => {
                // Clear history
                localStorage.removeItem('analysisHistory');
                window.analysisHistory = [];
                updateHistoryDisplay();
                
                // Show toast notification
                showToast('History Cleared', 'Your analysis history has been cleared', 'success');
                
                // Call API
                fetch('/clear_history', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    }
                });
            }, historyItems.length * 100 + 300);
        });
    }
    
    // Dark mode toggle
    const darkModeToggle = document.getElementById('dark-mode-toggle');
    if (darkModeToggle) {
        darkModeToggle.addEventListener('change', function() {
            if (this.checked) {
                document.body.classList.add('dark-mode');
                localStorage.setItem('darkMode', 'true');
                showToast('Dark Mode', 'Dark mode has been enabled', 'info');
            } else {
                document.body.classList.remove('dark-mode');
                localStorage.setItem('darkMode', 'false');
                showToast('Light Mode', 'Light mode has been enabled', 'info');
            }
        });
    }
}

// Handle form submission
async function handleFormSubmit(e) {
    e.preventDefault();
    
    // Get text
    const text = document.getElementById('news-text').value.trim();
    if (!text) {
        showToast('Error', 'Please enter a news article to analyze', 'error');
        return;
    }
    
    // Show loading
    document.getElementById('loading').style.display = 'block';
    document.getElementById('result-card').style.display = 'none';
    
    try {
        // Make API call
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ text })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            // Update result
            updateResultDisplay(data, text);
            
            // Add to history
            addToHistory(data, text);
            
            // Update history display
            updateHistoryDisplay();
            
            // Show result with animation
            const resultCard = document.getElementById('result-card');
            resultCard.style.display = 'block';
            
            // Remove previous animation classes
            resultCard.classList.remove('animate-fade-in');
            
            // Trigger reflow
            void resultCard.offsetWidth;
            
            // Add animation class
            resultCard.classList.add('animate-fade-in');
            
            // Show toast notification
            const toastMessage = data.prediction === 'FAKE' ? 
                'The article has been detected as potentially fake' : 
                'The article appears to be genuine';
                
            showToast('Analysis Complete', toastMessage, data.prediction === 'FAKE' ? 'error' : 'success');
        } else {
            showToast('Error', data.error || 'Something went wrong', 'error');
        }
    } catch (error) {
        showToast('Connection Error', 'Unable to connect to the server', 'error');
        console.error(error);
    } finally {
        document.getElementById('loading').style.display = 'none';
    }
}

// Update the result display
function updateResultDisplay(data, text) {
    const prediction = document.getElementById('prediction');
    const resultBadge = document.getElementById('result-badge');
    const resultIcon = document.getElementById('result-icon');
    const confidenceValue = document.getElementById('confidence-value');
    const confidenceText = document.getElementById('confidence-text');
    
    // Clear previous classes
    resultBadge.className = 'result-badge';
    confidenceValue.className = 'confidence-value';
    
    // Reset confidence width to 0 for animation
    confidenceValue.style.width = '0%';
    
    // Set new classes and values
    if (data.prediction === 'FAKE') {
        resultBadge.classList.add('fake-badge', 'animate-pulse');
        confidenceValue.classList.add('fake-meter');
        prediction.textContent = 'FAKE';
        resultIcon.className = 'fas fa-times-circle';
    } else {
        resultBadge.classList.add('real-badge');
        confidenceValue.classList.add('real-meter');
        prediction.textContent = 'REAL';
        resultIcon.className = 'fas fa-check-circle';
    }
    
    // Set confidence value with animation after a short delay
    setTimeout(() => {
        confidenceValue.style.width = `${data.confidence * 100}%`;
        confidenceText.textContent = `${Math.round(data.confidence * 100)}%`;
        
        // Remove pulse animation after a while
        if (data.prediction === 'FAKE') {
            setTimeout(() => {
                resultBadge.classList.remove('animate-pulse');
            }, 3000);
        }
    }, 300);
}

// Add an analysis to history
function addToHistory(data, text) {
    // Create history item
    const historyItem = {
        text: text,
        prediction: data.prediction,
        confidence: data.confidence,
        timestamp: new Date().toISOString()
    };
    
    // Add to history array
    window.analysisHistory.unshift(historyItem);
    
    // Keep only the latest 10 items
    if (window.analysisHistory.length > 10) {
        window.analysisHistory = window.analysisHistory.slice(0, 10);
    }
    
    // Save to localStorage
    localStorage.setItem('analysisHistory', JSON.stringify(window.analysisHistory));
}

// Update the history display
function updateHistoryDisplay() {
    const historyContainer = document.getElementById('history-container');
    const noHistory = document.getElementById('no-history');
    
    if (!historyContainer || !noHistory) return;
    
    // Clear current history items
    const existingItems = historyContainer.querySelectorAll('.history-item');
    existingItems.forEach(item => item.remove());
    
    // Show or hide the "no history" message
    if (window.analysisHistory.length === 0) {
        noHistory.style.display = 'block';
        
        // Hide clear history button if exists
        if (document.getElementById('clear-history-btn')) {
            document.getElementById('clear-history-btn').style.display = 'none';
        }
        return;
    } else {
        noHistory.style.display = 'none';
        
        // Show clear history button if exists
        if (document.getElementById('clear-history-btn')) {
            document.getElementById('clear-history-btn').style.display = 'block';
        }
    }
    
    // Add history items with staggered animation
    window.analysisHistory.forEach((item, index) => {
        const historyItem = document.createElement('div');
        historyItem.className = `history-item ${item.prediction.toLowerCase()}`;
        historyItem.style.animationDelay = `${index * 0.1}s`;
        historyItem.innerHTML = `
            <span class="history-label ${item.prediction === 'FAKE' ? 'bg-danger' : 'bg-info'} text-white">
                ${item.prediction}
            </span>
            <p class="history-text">${item.text.substring(0, 100)}${item.text.length > 100 ? '...' : ''}</p>
            <small class="text-muted">${formatTimestamp(item.timestamp)}</small>
        `;
        
        // Add click event to reload this article
        historyItem.addEventListener('click', function() {
            document.getElementById('news-text').value = item.text;
            document.getElementById('news-form').dispatchEvent(new Event('submit'));
        });
        
        historyContainer.appendChild(historyItem);
    });
}

// Format timestamp for display
function formatTimestamp(timestamp) {
    const date = new Date(timestamp);
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) + ' ' + 
           date.toLocaleDateString([], { month: 'short', day: 'numeric' });
}

// Show toast notification
function showToast(title, message, type = 'info') {
    const toastContainer = document.querySelector('.toast-container');
    if (!toastContainer) return;
    
    const toast = document.createElement('div');
    toast.className = `toast-notification ${type}`;
    
    // Set icon based on type
    let icon = 'fa-info-circle';
    if (type === 'success') icon = 'fa-check-circle';
    if (type === 'error') icon = 'fa-exclamation-circle';
    
    toast.innerHTML = `
        <div class="toast-icon">
            <i class="fas ${icon}"></i>
        </div>
        <div class="toast-content">
            <div class="toast-title">${title}</div>
            <div class="toast-message">${message}</div>
        </div>
        <div class="toast-close">
            <i class="fas fa-times"></i>
        </div>
    `;
    
    // Add to container
    toastContainer.appendChild(toast);
    
    // Add click event for close button
    toast.querySelector('.toast-close').addEventListener('click', function() {
        toast.style.opacity = '0';
        toast.style.transform = 'translateX(20px)';
        setTimeout(() => {
            toast.remove();
        }, 300);
    });
    
    // Auto remove after 5 seconds
    setTimeout(() => {
        if (toast.parentNode) {
            toast.style.opacity = '0';
            toast.style.transform = 'translateX(20px)';
            setTimeout(() => {
                toast.remove();
            }, 300);
        }
    }, 5000);
}

// Copy text to clipboard
function copyToClipboard(text) {
    const textarea = document.createElement('textarea');
    textarea.value = text;
    document.body.appendChild(textarea);
    textarea.select();
    document.execCommand('copy');
    document.body.removeChild(textarea);
    
    showToast('Copied', 'Text copied to clipboard', 'success');
}

// Export functions for external use
window.copyToClipboard = copyToClipboard; 