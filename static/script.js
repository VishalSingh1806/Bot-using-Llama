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

// Define the backend URLs dynamically
const BACKEND_CHAT_URL = "http://34.41.145.80:8000/chat";
const BACKEND_FORM_URL = "http://34.41.145.80:8000/collect_user_data";

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

// Display Form
function displayForm() {
    const chatContent = document.getElementById("chatContent");

    const formHtml = `
        <div class="bot-message fade-in">
            <div class="form-container">
                <h3>Please provide your details</h3>
                <form id="userForm">
                    <div class="form-group">
                        <label for="name">Name:</label>
                        <input type="text" id="name" placeholder="Your full name" required>
                    </div>
                    <div class="form-group">
                        <label for="email">Email:</label>
                        <input type="email" id="email" placeholder="Your email address" required>
                    </div>
                    <div class="form-group">
                        <label for="phone">Phone:</label>
                        <input type="text" id="phone" placeholder="Your phone number" required>
                    </div>
                    <div class="form-group">
                        <label for="organization">Organization:</label>
                        <input type="text" id="organization" placeholder="Your organization name" required>
                    </div>
                    <div class="form-actions">
                        <button type="button" class="submit-button" onclick="submitForm()">Submit</button>
                    </div>
                </form>
            </div>
        </div>`;
    chatContent.innerHTML += formHtml;
    chatContent.scrollTop = chatContent.scrollHeight;
}


// Submit Form Data
async function submitForm() {
    const name = document.getElementById("name").value;
    const email = document.getElementById("email").value;
    const phone = document.getElementById("phone").value;
    const organization = document.getElementById("organization").value;

    if (name && email && phone && organization) {
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
                addMessageToChat(data.message || "Details collected successfully!", "bot-message");
                document.getElementById("userForm").remove(); // Remove form on success
            } else {
                addMessageToChat("Error submitting your details. Please try again.", "bot-message");
            }
        } catch (error) {
            addMessageToChat("Network error while submitting your details.", "bot-message");
            console.error("Fetch error:", error);
        }
    } else {
        addMessageToChat("Please fill out all fields before submitting.", "bot-message");
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
    if (event.key === "Enter") {
        sendMessage();
    }
}

// Send User Message
async function sendMessage() {
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
