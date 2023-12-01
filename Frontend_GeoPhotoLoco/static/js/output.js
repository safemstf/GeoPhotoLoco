document.addEventListener('DOMContentLoaded', function() {
  var map = L.map('map').setView([0, 0], 2); // Default center and wider zoom

L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '© OpenStreetMap contributors'
}).addTo(map);

// Function to get user input for latitude and longitude
function getUserInput(promptMessage) {
    var userInput = parseFloat(prompt(promptMessage));
    // Check if the user canceled the prompt or entered an invalid value
    if (isNaN(userInput)) {
        alert("Invalid or empty input. Showing default location.");
        return 0; // Return a default value
    }
    return userInput;
}

// Get latitude and longitude for the first location
var userInputLat1 = getUserInput("Enter Latitude for Location 1:"); // change these values to the ones from the AI
var userInputLng1 = getUserInput("Enter Longitude for Location 1:"); // change these values to the ones from the AI

// Get latitude and longitude for the second location
var userInputLat2 = getUserInput("Enter Latitude for Location 2:"); // change these values to the ones from the AI
var userInputLng2 = getUserInput("Enter Longitude for Location 2:"); // change these values to the ones from the AI

// Add markers using user input coordinates
var marker1 = L.marker([userInputLat1, userInputLng1]).addTo(map)
    .bindPopup(`Location 1: ${userInputLat1}, ${userInputLng1}`)
    .openPopup();

var marker2 = L.marker([userInputLat2, userInputLng2]).addTo(map)
    .bindPopup(`Location 2: ${userInputLat2}, ${userInputLng2}`)
    .openPopup();

// Create a Polyline between the two points
var polyline = L.polyline([
    [userInputLat1, userInputLng1],
    [userInputLat2, userInputLng2]
], { color: 'red' }).addTo(map);

  // Calculate the midpoint of the line
var midpoint = polyline.getCenter();

  // Add a label above the center of the line
L.popup()
    .setLatLng([midpoint.lat, midpoint.lng])
    .setContent(`Distance: ${calculateDistance(userInputLat1, userInputLng1, userInputLat2, userInputLng2).toFixed(2)} kilometers`)
    .openOn(map);

  // Center the map on the midpoint of the two locations with a wider default zoom
  map.setView([midpoint.lat, midpoint.lng], 6); // Adjust the zoom level as needed
});

// Function to calculate distance between two points using Turf.js
function calculateDistance(lat1, lng1, lat2, lng2) {
    var point1 = turf.point([lng1, lat1]);
    var point2 = turf.point([lng2, lat2]);
    var options = { units: 'kilometers' };
    return turf.distance(point1, point2, options);
}