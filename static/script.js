// Toggle Chat Window
function toggleChat() {
    const chatWindow = document.getElementById("chatWindow");
    if (chatWindow.style.display === "block") {
        chatWindow.style.display = "none";
    } else {
        chatWindow.style.display = "block";

        const chatContent = document.getElementById("chatContent");
        if (chatContent.children.length === 0) {
            const welcomeMessage = document.createElement("div");
            welcomeMessage.className = "bot-message fade-in";
            welcomeMessage.innerText = "Hello! How can I assist you today?";
            chatContent.appendChild(welcomeMessage);
            chatContent.scrollTop = chatContent.scrollHeight;
        }
    }
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
            const response = await fetch("http://34.41.145.80:8000/chat", {
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
