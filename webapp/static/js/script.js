// Show the download link after the predictions are generated
document.querySelector('form').addEventListener('submit', function() {
    document.getElementById('download-link').style.display = 'none';
});

// Handle the response from the Flask server
document.querySelector('form').addEventListener('submit', function(event) {
    event.preventDefault();
    const formData = new FormData(event.target);
    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.blob())
    .then(blob => {
        const url = URL.createObjectURL(blob);
        document.getElementById('download-link').href = url;
        document.getElementById('download-link').style.display = 'block';
    });
});