document.addEventListener('DOMContentLoaded', function() {
  var map = L.map('map').setView([0, 0], 2); // Default center and wider zoom

  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '© OpenStreetMap contributors'
  }).addTo(map);

  var marker1, marker2, polyline;

  function updateMap(data) {
    var userInputLat1 = data.message.processed_data.Coordinates[0];
    var userInputLng1 = data.message.processed_data.Coordinates[1];

    var userInputLat2 = data.message.processed_data.True_Coordinates[0];
    var userInputLng2 = data.message.processed_data.True_Coordinates[1];

    // Update markers or create new ones
    if (!marker1) {
        marker1 = L.marker([userInputLat1, userInputLng1]).addTo(map);
    } else {
        marker1.setLatLng([userInputLat1, userInputLng1]);
    }

    // Update markers or create new ones
    if (!marker2) {
        marker2 = L.marker([userInputLat2, userInputLng2]).addTo(map);
    } else {
        marker2.setLatLng([userInputLat2, userInputLng2]);
    }

    // Update polyline or create a new one
    if (!polyline) {
        polyline = L.polyline([
            [userInputLat1, userInputLng1],
            [userInputLat2, userInputLng2]
        ], { color: 'red' }).addTo(map);
    } else {
        polyline.setLatLngs([
            [userInputLat1, userInputLng1],
            [userInputLat2, userInputLng2]
        ]);
    }

    // Calculate the midpoint of the line
    var midpoint = polyline.getCenter();

    // Calculate the distance
    var distance = calculateDistance(userInputLat1, userInputLng1, userInputLat2, userInputLng2);

    // Calculate an appropriate zoom level based on the distance
    var zoomLevel = getZoomLevelFromDistance(distance);

    // Add a label above the center of the line
    L.popup()
        .setLatLng([midpoint.lat, midpoint.lng])
        .setContent(`Regression Predicted Distance: ${distance.toFixed(2)} kilometers`)
        .openOn(map);

    // Center the map on the midpoint of the two locations with the calculated zoom level
    map.setView([midpoint.lat, midpoint.lng], zoomLevel);
}

// Function to calculate an appropriate zoom level based on the distance
function getZoomLevelFromDistance(distance) {
    // You can customize this function based on your desired zoom level logic
    // This is just a simple example, you may need to fine-tune the values
    if (distance < 10) {
        return 8;
    } else if (distance < 1000) {
        return 6;
    } else {
        return 3;
    }
}
  // Function to calculate distance between two points using Turf.js
  function calculateDistance(lat1, lng1, lat2, lng2) {
      var point1 = turf.point([lng1, lat1]);
      var point2 = turf.point([lng2, lat2]);
      var options = { units: 'kilometers' };
      return turf.distance(point1, point2, options);
  }

  // Keep track of the previous data
  let previousData = null;

  // Function to fetch updates from the server
  function fetchUpdates() {
    fetch('http://127.0.0.1:5000/get_data', {
      method: 'GET',
    })
      .then(response => response.json())
      .then(data => {
        console.log('New data from the server:', data);

        // Check if the data has changed
        if (!isEqual(data, previousData)) {
          // Update the map with new data
          updateMap(data);
          const resultDisplay = document.getElementById('result-display');
          var userInputCountry = data.message.processed_data.Country;
          var userInputRegion = data.message.processed_data.Region;
          var trueCountry = data.message.processed_data.True_Country;
          var trueRegion = data.message.processed_data.True_Region;

          // Construct a string with the desired information
          var displayString = `Predicted Classification Data: ${userInputCountry}, ${userInputRegion}\n`;
          displayString += `True Data: ${trueCountry}, ${trueRegion}`;
          resultDisplay.textContent = displayString;

          // Update the previous data
          previousData = data;
        }

        // Schedule the next update
        setTimeout(fetchUpdates, 1000);  // Adjust the interval as needed
      })
      .catch(error => {
        console.error('Error fetching updates:', error);

        // Retry after a delay
        setTimeout(fetchUpdates, 5000);  // Retry after 5 seconds (adjust as needed)
      });
  }

  // Start fetching updates
  fetchUpdates();
});

// Utility function to deep compare two objects
function isEqual(obj1, obj2) {
  return JSON.stringify(obj1) === JSON.stringify(obj2);
}
