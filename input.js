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

      // setting maximum size for inserted photo 
      const maxWidth = 400; 
      const maxHeight = 300; 

      imageElement.onload = function () {
        const width = imageElement.width;
        const height = imageElement.height;

        // resize scaling
        const widthFactor = maxWidth / width;
        const heightFactor = maxHeight / height;
        const resizeFactor = Math.min(widthFactor, heightFactor);

        // resizing the image proportionally according to ^ factor
        imageElement.width = width * resizeFactor;
        imageElement.height = height * resizeFactor;
        imageElement.style.opacity = .7;
        imageElement.style.borderRadius = '10px';

        imageContainer.innerHTML = ''; // Clear the previous content
        imageContainer.appendChild(imageElement);
      };
    };

    reader.readAsDataURL(file);
  } else {
    alert('Please drop a valid image file (JPEG, JPG, or PNG).');
  }
});