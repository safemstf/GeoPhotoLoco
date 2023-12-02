const imageContainer = document.getElementById('image-container');

imageContainer.addEventListener('dragover', (event) => {
  event.preventDefault();
  imageContainer.classList.add('drag-over');
});

imageContainer.addEventListener('dragenter', (event) => {
  event.preventDefault();
  imageContainer.classList.add('drag-over');
});

imageContainer.addEventListener('dragleave', (event) => {
  event.preventDefault();
  imageContainer.classList.remove('drag-over');
});

imageContainer.addEventListener('drop', (event) => {
  event.preventDefault();
  imageContainer.classList.remove('drag-over');

  const file = event.dataTransfer.files[0];
  const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png'];

  if (allowedTypes.includes(file.type)) {
    const reader = new FileReader();

    reader.onload = function (readerEvent) {
      const imageElement = document.createElement('img');
      imageElement.src = readerEvent.target.result;

      // Adjust the maximum width and height to fit your layout
      const maxWidth = 400; // Set the maximum width
      const maxHeight = 300; // Set the maximum height

      imageElement.onload = function () {
        const width = imageElement.width;
        const height = imageElement.height;

        // Calculate the proportional resizing factor
        const widthFactor = maxWidth / width;
        const heightFactor = maxHeight / height;
        const resizeFactor = Math.min(widthFactor, heightFactor);

        // Resize the image proportionally
        imageElement.width = width * resizeFactor;
        imageElement.height = height * resizeFactor;
        imageElement.style.opacity = .7;
        imageElement.style.borderRadius = '10px';

        imageContainer.innerHTML = ''; // Clear the previous content
        imageContainer.appendChild(imageElement);

        // Send the image to the Flask API
        sendImageToAPI(file);
        // Start fetching updates
        fetchUpdates();
      };
    };

    reader.readAsDataURL(file);
  } else {
    alert('Please drop a valid image file (JPEG, JPG, or PNG).');
  }
});

// Function to fetch updates from the server
function fetchUpdates() {
  fetch('http://127.0.0.1:5000/get_data', {
    method: 'GET',
  })
    .then(response => response.json())
    .then(data => {
      console.log('New data from the server:', data);

      // Update the result display
      const resultDisplay = document.getElementById('result-display');
      resultDisplay.textContent = JSON.stringify(data, null, 2);

      // Schedule the next update
      setTimeout(fetchUpdates, 2000);  // Adjust the interval as needed
    })
    .catch(error => {
      console.error('Error fetching updates:', error);

      // Retry after a delay
      setTimeout(fetchUpdates, 5000);  // Retry after 5 seconds (adjust as needed)
    });
}




function sendImageToAPI(imageFile) {
  const formData = new FormData();
  formData.append('file', imageFile);

  fetch('http://127.0.0.1:5000/send_data', {
    method: 'POST',
    body: formData,
  })
    .then(response => response.json())
    .then(data => {
      console.log(data); // Handle the response from the server
    })
    .catch(error => {
      console.error('Error:', error);
    });
}
