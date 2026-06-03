document.addEventListener('DOMContentLoaded', () => {
    // 1. Tab Navigation Logic
    const navButtons = document.querySelectorAll('.nav-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    navButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            // Remove active class from all buttons and tabs
            navButtons.forEach(b => b.classList.remove('active'));
            tabContents.forEach(tc => tc.classList.remove('active'));

            // Add active to clicked
            btn.classList.add('active');
            const targetTab = btn.getAttribute('data-tab');
            document.getElementById(targetTab).classList.add('active');
        });
    });

    // 2. Plot Controls Switching Logic
    const plotButtons = document.querySelectorAll('.plot-btn');
    const plotWrappers = document.querySelectorAll('.plot-wrapper');

    plotButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            // Remove active classes
            plotButtons.forEach(b => b.classList.remove('active'));
            plotWrappers.forEach(pw => pw.classList.remove('active-plot'));

            // Add active to clicked
            btn.classList.add('active');
            const targetPlot = btn.getAttribute('data-plot');
            document.getElementById(`plot-${targetPlot}`).classList.add('active-plot');
        });
    });

    // 3. Form Prediction Calculation API Call
    const predictionForm = document.getElementById('prediction-form');
    const resultDisplayEmpty = document.getElementById('result-display');
    const resultDataContainer = document.getElementById('result-data');
    const predictedTimeHeader = document.getElementById('predicted-time');
    const predictedMinsParagraph = document.getElementById('predicted-raw-mins');
    
    const distContribSpan = document.getElementById('dist-contrib');
    const stopsContribSpan = document.getElementById('stops-contrib');

    predictionForm.addEventListener('submit', async (e) => {
        e.preventDefault();

        const distance = parseFloat(document.getElementById('distance').value);
        const stops = parseInt(document.getElementById('stops').value);

        try {
            // Call our local FastAPI server backend
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ distance, stops })
            });

            if (!response.ok) {
                throw new Error('Prediction API call failed.');
            }

            const data = await response.json();

            // Display results
            resultDisplayEmpty.style.display = 'none';
            resultDataContainer.style.display = 'block';

            // Extract minutes
            const rawMins = data.predicted_minutes;
            predictedMinsParagraph.textContent = `${rawMins.toFixed(1)} minutes total`;

            // Format to hours & minutes
            const hours = Math.floor(rawMins / 60);
            const remainingMins = Math.round(rawMins % 60);
            
            if (hours > 0) {
                predictedTimeHeader.textContent = `${hours} hrs ${remainingMins} mins`;
            } else {
                predictedTimeHeader.textContent = `${remainingMins} mins`;
            }

            // Update parameters breakdown
            distContribSpan.textContent = `+${data.distance_contribution.toFixed(1)}`;
            stopsContribSpan.textContent = `+${data.stops_contribution.toFixed(1)}`;

        } catch (error) {
            console.error('Inference Error:', error);
            alert('Error running model inference. Make sure backend app.py is running.');
        }
    });

    // 4. LinkedIn Transcript Copy Button
    const copyBtn = document.getElementById('copy-transcript-btn');
    const transcriptText = document.getElementById('transcript-text');

    copyBtn.addEventListener('click', () => {
        transcriptText.select();
        transcriptText.setSelectionRange(0, 99999); // For mobile devices

        try {
            navigator.clipboard.writeText(transcriptText.value);
            copyBtn.textContent = 'Copied! ✓';
            copyBtn.style.background = '#00ff7f';
            copyBtn.style.color = '#000';
            
            setTimeout(() => {
                copyBtn.textContent = 'Copy to Clipboard';
                copyBtn.style.background = 'rgba(255, 255, 255, 0.05)';
                copyBtn.style.color = 'var(--text-primary)';
            }, 2500);
        } catch (err) {
            console.error('Failed to copy text: ', err);
            alert('Unable to copy automatically. Please select the text and copy manually.');
        }
    });
});
