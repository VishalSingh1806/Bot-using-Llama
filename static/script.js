let isFormSubmitted = false; // Default state: form not submitted

// Toggle Chat Window
function toggleChat() {
    const chatWindow = document.getElementById("chatWindow");
    if (chatWindow.style.display === "block") {
        chatWindow.style.display = "none";
    } else {
        chatWindow.style.display = "block";

        const chatContent = document.getElementById("chatContent");
        if (chatContent.children.length === 0) {
            triggerBackendForForm(); // Trigger backend for the form or welcome message
        }
    }
}

const MAX_CHAR_LIMIT = 100; // Set maximum allowed characters

// Initialize the character countdown
document.addEventListener("DOMContentLoaded", function () {
    const userMessageInput = document.getElementById("userMessage");
    const sendButton = document.querySelector(".chat-footer button"); // Get the "Send" button

    // Disable the send button initially
    sendButton.disabled = true;
    sendButton.style.cursor = "not-allowed"; // Change cursor to indicate disabled state

    // Add a character counter dynamically
    const counter = document.createElement("div");
    counter.id = "charCounter";
    counter.className = "char-counter"; // Apply styling through CSS
    counter.textContent = `Max Characters ${MAX_CHAR_LIMIT}`;
    userMessageInput.parentNode.insertAdjacentElement("afterend", counter);

    // Add event listener for input changes
    userMessageInput.addEventListener("input", function () {
        const remaining = MAX_CHAR_LIMIT - this.value.length;

        if (remaining > 0) {
            counter.textContent = `Remaining input: ${remaining}`;
            this.style.borderColor = ""; // Reset border color if valid
            sendButton.disabled = this.value.trim().length === 0; // Enable button if input is valid
            sendButton.style.cursor = sendButton.disabled ? "not-allowed" : "pointer";
        } else {
            counter.textContent = `Input limit exceeded`;
            this.style.borderColor = "red"; // Highlight input field in red
            sendButton.disabled = true; // Disable button when input exceeds limit
            sendButton.style.cursor = "not-allowed";
        }

        // Disable further input when limit is reached
        if (this.value.length >= MAX_CHAR_LIMIT) {
            this.value = this.value.substring(0, MAX_CHAR_LIMIT); // Truncate excess input
            counter.textContent = `Remaining input: 0`;
        }
    });
});



// Define the backend URLs dynamically
const BACKEND_CHAT_URL = "http://34.132.31.71:8000/chat";
const BACKEND_FORM_URL = "http://34.132.31.71:8000/collect_user_data";

// Trigger Backend for Form
async function triggerBackendForForm() {
    const chatContent = document.getElementById("chatContent");

    const loadingMessage = document.createElement("div");
    loadingMessage.className = "bot-message fade-in";
    loadingMessage.innerText = "Bot is loading...";
    chatContent.appendChild(loadingMessage);
    chatContent.scrollTop = chatContent.scrollHeight;

    try {
        const response = await fetch(BACKEND_CHAT_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ session_id: localStorage.getItem("session_id") || null }),
        });

        chatContent.removeChild(loadingMessage);

        if (response.ok) {
            const data = await response.json();
            localStorage.setItem("session_id", data.session_id); // Save session ID
            if (data.redirect_to === "/collect_user_data") {
                displayForm(); // Call function to display the form
            } else if (data.message) {
                addMessageToChat(data.message, "bot-message");
            }
        } else {
            addMessageToChat("Error fetching bot message. Please try again.", "bot-message");
        }
    } catch (error) {
        chatContent.removeChild(loadingMessage);
        addMessageToChat("Network error. Please check your connection.", "bot-message");
        console.error("Fetch error:", error);
    }
}

function displayForm() {
    const chatContent = document.getElementById("chatContent");

    const formHtml = `
        <div class="bot-message fade-in">
            <div class="form-container">
                <h3 id="formHeading">Let's get to know you!</h3>
                <p class="form-description">Let’s start by knowing a little about you. It won’t take long!</p>
                <form id="userForm">
                    <div class="form-group">
                        <label for="name">Name</label>
                        <input type="text" id="name" placeholder="Enter your full name" required>
                        <div class="error-message" id="nameError"></div>
                    </div>
                    <div class="form-group">
                        <label for="email">Email</label>
                        <input type="email" id="email" placeholder="Enter your email address" required>
                        <div class="error-message" id="emailError"></div>
                    </div>
                    <div class="form-group">
                        <label for="phone">Phone</label>
                        <input type="text" id="phone" placeholder="Enter your phone number" required>
                        <div class="error-message" id="phoneError"></div>
                    </div>
                    <div class="form-group">
                        <label for="organization">Organization</label>
                        <input type="text" id="organization" placeholder="Enter your organization name" required>
                        <div class="error-message" id="organizationError"></div>
                    </div>
                    <div class="form-actions">
                        <button type="button" class="submit-button" onclick="submitForm()" disabled>Submit</button>
                    </div>
                </form>
            </div>
        </div>`;

    chatContent.innerHTML += formHtml;
    chatContent.scrollTop = chatContent.scrollHeight;

    // Add blur validation listeners
    document.getElementById("name").addEventListener("blur", validateName);
    document.getElementById("email").addEventListener("blur", validateEmail);
    document.getElementById("phone").addEventListener("blur", validatePhone);
    document.getElementById("organization").addEventListener("blur", validateOrganization);

    // Add input event to enable the submit button after all fields are valid
    document.getElementById("name").addEventListener("input", toggleSubmitButton);
    document.getElementById("email").addEventListener("input", toggleSubmitButton);
    document.getElementById("phone").addEventListener("input", toggleSubmitButton);
    document.getElementById("organization").addEventListener("input", toggleSubmitButton);
}


// Validation Helper Functions
function isValidEmail(email) {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/; // Basic email format
    return emailRegex.test(email);
}

function isValidPhone(phone) {
    const phoneRegex = /^\d{10}$/; // 10-digit phone number
    return phoneRegex.test(phone);
}

function isValidName(name) {
    const nameRegex = /^[a-zA-Z\s]+$/; // Allow only letters and spaces
    return name.trim().length > 0 && nameRegex.test(name);
}

function isValidOrganization(organization) {
    return organization.trim().length > 0 && organization.length <= 100; // Max 100 characters
}


function validateName() {
    const name = document.getElementById("name").value.trim();
    const nameError = document.getElementById("nameError");
    if (!isValidName(name)) {
        nameError.textContent = "Please enter a valid name (letters and spaces only).";
        nameError.style.display = "block";
    } else {
        nameError.style.display = "none";
    }
}

function validateEmail() {
    const email = document.getElementById("email").value.trim();
    const emailError = document.getElementById("emailError");
    if (!isValidEmail(email)) {
        emailError.textContent = "Oops! Your email address looks incomplete. Please check again.";
        emailError.style.display = "block";
    } else {
        emailError.style.display = "none";
    }
}

function validatePhone() {
    const phone = document.getElementById("phone").value.trim();
    const phoneError = document.getElementById("phoneError");
    if (!isValidPhone(phone)) {
        phoneError.textContent = "Please enter a valid 10-digit phone number.";
        phoneError.style.display = "block";
    } else {
        phoneError.style.display = "none";
    }
}

function validateOrganization() {
    const organization = document.getElementById("organization").value.trim();
    const organizationError = document.getElementById("organizationError");
    if (!isValidOrganization(organization)) {
        organizationError.textContent = "Please enter a valid organization name (max 100 characters).";
        organizationError.style.display = "block";
    } else {
        organizationError.style.display = "none";
    }
}


// Toggle Submit Button
function toggleSubmitButton() {
    const submitButton = document.querySelector(".submit-button");
    const isFormValid =
        isValidName(document.getElementById("name").value.trim()) &&
        isValidEmail(document.getElementById("email").value.trim()) &&
        isValidPhone(document.getElementById("phone").value.trim()) &&
        isValidOrganization(document.getElementById("organization").value.trim());

    submitButton.disabled = !isFormValid;
}


// Submit Form Data
async function submitForm() {
    const name = document.getElementById("name").value.trim();
    const email = document.getElementById("email").value.trim();
    const phone = document.getElementById("phone").value.trim();
    const organization = document.getElementById("organization").value.trim();

    try {
        const response = await fetch(BACKEND_FORM_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                session_id: localStorage.getItem("session_id") || null,
                name,
                email,
                phone,
                organization,
            }),
        });

        if (response.ok) {
            const data = await response.json();

            // Update the heading dynamically
            const formHeading = document.getElementById("formHeading");
            if (formHeading) {
                formHeading.innerText = "🌟Great! Let’s get started with your EPR-related questions.";
            }

            // Remove the introductory line
            const formDescription = document.querySelector(".form-description");
            if (formDescription) {
                formDescription.style.display = "none";
            }

            // Remove the form
            document.getElementById("userForm").remove();

            // Allow user to send queries
            isFormSubmitted = true;

            addMessageToChat("You can now start sending your queries.", "bot-message");
        } else {
            addMessageToChat("Error submitting your details. Please try again.", "bot-message");
        }
    } catch (error) {
        addMessageToChat("Network error while submitting your details.", "bot-message");
        console.error("Fetch error:", error);
    }
}



// Add Message to Chat
function addMessageToChat(message, className) {
    const chatContent = document.getElementById("chatContent");
    const messageElement = document.createElement("div");
    messageElement.className = className;
    messageElement.innerText = message;
    chatContent.appendChild(messageElement);
    chatContent.scrollTop = chatContent.scrollHeight;
}

// Handle Enter Key
function checkEnter(event) {
    if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
}


// Send User Message
async function sendMessage() {
    if (!isFormSubmitted) {
        addMessageToChat("Please complete the form before sending queries.", "bot-message");
        return;
    }

    const userMessage = document.getElementById("userMessage").value.trim();
    const chatContent = document.getElementById("chatContent");

    if (userMessage) {
        addMessageToChat(userMessage, "user-message");
        document.getElementById("userMessage").value = "";

        const typingIndicator = document.createElement("div");
        typingIndicator.className = "typing-indicator";
        typingIndicator.innerText = "Bot is typing...";
        chatContent.appendChild(typingIndicator);
        chatContent.scrollTop = chatContent.scrollHeight;

        try {
            const response = await fetch(BACKEND_CHAT_URL, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    message: userMessage,
                    session_id: localStorage.getItem("session_id"),
                }),
            });

            chatContent.removeChild(typingIndicator);

            if (response.ok) {
                const data = await response.json();
                if (data && data.answer) {
                    addMessageToChat(data.answer, "bot-message");
                } else {
                    addMessageToChat("No valid response received from the bot.", "bot-message");
                }
            } else {
                addMessageToChat("Server error. Please try again later.", "bot-message");
            }
        } catch (error) {
            chatContent.removeChild(typingIndicator);
            addMessageToChat("Network error. Please check your connection.", "bot-message");
            console.error("Fetch error:", error);
        }
    }
}

