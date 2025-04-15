document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('prediction-form');
    const resultsDiv = document.getElementById('results');
    const errorDiv = document.getElementById('error');
    
    form.addEventListener('submit', function(e) {
        e.preventDefault();
        
        // Hide previous results/errors
        resultsDiv.classList.add('hidden');
        errorDiv.classList.add('hidden');
        
        // Get form data
        const formData = {
            N: document.getElementById('N').value,
            P: document.getElementById('P').value,
            K: document.getElementById('K').value,
            temperature: document.getElementById('temperature').value,
            humidity: document.getElementById('humidity').value,
            ph: document.getElementById('ph').value,
            rainfall: document.getElementById('rainfall').value,
            fertilizer: document.getElementById('fertilizer').value
        };
        
        // Make prediction request
        fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(formData)
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                // Display results
                document.getElementById('recommended-crop').textContent = data.recommended_crop;
                document.getElementById('predicted-yield').textContent = data.predicted_yield;
                resultsDiv.classList.remove('hidden');
            } else {
                // Display error
                errorDiv.textContent = data.message || 'An error occurred';
                errorDiv.classList.remove('hidden');
            }
        })
        .catch(error => {
            errorDiv.textContent = 'Network error: ' + error.message;
            errorDiv.classList.remove('hidden');
        });
    });
});