// Toggle Chat Window
function toggleChat() {
    const chatWindow = document.getElementById("chatWindow");
    if (chatWindow.style.display === "block") {
        chatWindow.style.display = "none";
    } else {
        chatWindow.style.display = "block";

        const chatContent = document.getElementById("chatContent");
        if (chatContent.children.length === 0) {
            // Trigger backend to fetch form or welcome message
            triggerBackendForForm();
        }
    }
}

// Define the backend URLs dynamically
const BACKEND_CHAT_URL = "http://34.41.145.80:8000/chat";
const BACKEND_FORM_URL = "http://34.41.145.80:8000/collect_user_data";

// Store session ID globally
let sessionId = null;

// Trigger backend when chat widget opens
async function initializeChat() {
    const chatWindow = document.getElementById("chatWindow");
    chatWindow.style.display = "block";

    const chatContent = document.getElementById("chatContent");
    const typingIndicator = document.createElement("div");
    typingIndicator.className = "typing-indicator";
    typingIndicator.innerText = "Bot is typing...";
    chatContent.appendChild(typingIndicator);

    try {
        const response = await fetch("http://34.41.145.80:8000/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message: "", session_id: sessionId }),
        });

        chatContent.removeChild(typingIndicator);

        if (response.ok) {
            const data = await response.json();
            sessionId = data.session_id; // Save the session ID
            if (data.redirect_to === "/collect_user_data") {
                showUserDataForm();
            } else if (data.message) {
                addMessageToChat(data.message, "bot-message");
            }
        } else {
            addMessageToChat("Error initializing chat. Please try again.", "bot-message");
        }
    } catch (error) {
        chatContent.removeChild(typingIndicator);
        addMessageToChat("Network error. Please check your connection.", "bot-message");
        console.error("Initialization error:", error);
    }
}

// Display user data form
function showUserDataForm() {
    const chatContent = document.getElementById("chatContent");
    const formContainer = document.createElement("div");
    formContainer.innerHTML = `
        <form id="userDataForm">
            <label for="name">Name:</label>
            <input type="text" id="name" name="name" required />
            <label for="email">Email:</label>
            <input type="email" id="email" name="email" required />
            <label for="phone">Phone:</label>
            <input type="text" id="phone" name="phone" required />
            <label for="organization">Organization:</label>
            <input type="text" id="organization" name="organization" required />
            <button type="submit">Submit</button>
        </form>
    `;
    chatContent.appendChild(formContainer);

    const form = document.getElementById("userDataForm");
    form.addEventListener("submit", handleUserDataSubmit);
}

// Handle user data submission
async function handleUserDataSubmit(event) {
    event.preventDefault();
    const form = event.target;
    const formData = {
        name: form.name.value,
        email: form.email.value,
        phone: form.phone.value,
        organization: form.organization.value,
        session_id: sessionId, // Include session ID in the payload
    };

    try {
        const response = await fetch("http://34.41.145.80:8000/collect_user_data", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(formData),
        });

        if (response.ok) {
            const data = await response.json();
            addMessageToChat(data.message, "bot-message");
            form.remove(); // Remove the form after successful submission
        } else {
            const errorData = await response.json();
            addMessageToChat(errorData.message || "Error submitting your details.", "bot-message");
        }
    } catch (error) {
        addMessageToChat("Network error. Please try again.", "bot-message");
        console.error("Submission error:", error);
    }
}

// Add message to chat
function addMessageToChat(message, className) {
    const chatContent = document.getElementById("chatContent");
    const messageElement = document.createElement("div");
    messageElement.className = className;
    messageElement.innerText = message;
    chatContent.appendChild(messageElement);
    chatContent.scrollTop = chatContent.scrollHeight;
}

// Initialize chat widget when it opens
document.getElementById("chatToggleButton").addEventListener("click", initializeChat);


// Handle Enter Key
function checkEnter(event) {
    if (event.key === "Enter") {
        sendMessage();
    }
}

// Define the backend URL dynamically
const BACKEND_URL = "http://34.41.145.80:8000/chat";

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
            const response = await fetch(BACKEND_URL, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ message: userMessage }),
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



// Add Message to Chat
function addMessageToChat(message, className) {
    const chatContent = document.getElementById("chatContent");
    const messageElement = document.createElement("div");
    messageElement.className = className;
    messageElement.innerText = message;
    chatContent.appendChild(messageElement);
    chatContent.scrollTop = chatContent.scrollHeight;
}
