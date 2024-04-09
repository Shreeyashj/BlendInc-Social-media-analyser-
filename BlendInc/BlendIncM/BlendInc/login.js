// script.js

// Function to toggle the navigation menu
function myMenuFunction() {
    var menu = document.getElementById("navMenu");
    if (menu.className === "nav-menu") {
        menu.className += " responsive";
    } else {
        menu.className = "nav-menu";
    }
}

// Function to toggle between login and registration forms
function login() {
    var loginForm = document.getElementById("login");
    var registerForm = document.getElementById("register");
    loginForm.style.left = "4px";
    registerForm.style.right = "-520px";
    document.getElementById("loginBtn").className += " white-btn";
    document.getElementById("registerBtn").className = "btn";
    loginForm.style.opacity = 1;
    registerForm.style.opacity = 0;
}

function register() {
    var loginForm = document.getElementById("login");
    var registerForm = document.getElementById("register");
    loginForm.style.left = "-510px";
    registerForm.style.right = "5px";
    document.getElementById("loginBtn").className = "btn";
    document.getElementById("registerBtn").className += " white-btn";
    loginForm.style.opacity = 0;
    registerForm.style.opacity = 1;
}

// Function to handle the login process
function loginUser() {
    // Placeholder for actual login logic
    // For demonstration, let's assume login is successful
    var username = document.getElementById("username").value;
    var password = document.getElementById("password").value;

    // Check if username and password are valid (for demonstration purposes)
    if (username === "Abhishek" && password === "12345678") {
        // Redirect to the recommendation page
        window.location.href = "page2.html";
    } else {
        // Display error message (for demonstration purposes)
        alert("Invalid username or password. Please try again.");
    }
}
