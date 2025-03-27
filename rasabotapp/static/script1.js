document.addEventListener("DOMContentLoaded", function () {
    const messagesContainer = document.getElementById("messages");
    const userInput = document.getElementById("user-input");

    function appendMessage(sender, message) {
        const messageElement = document.createElement("div");
        messageElement.classList.add("message", sender);
        messageElement.innerHTML = `<p>${message}</p>`;
        messagesContainer.appendChild(messageElement);

        // ✅ Auto-scroll to the latest message
        messagesContainer.scrollTop = messagesContainer.scrollHeight;

        setTimeout(() => {
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }, 1);
    }

    window.sendMessage = function () {
        const message = userInput.value.trim();
        if (!message) return;

        // Display user message on the right
        appendMessage("user", message);
        userInput.value = "";

        // Send message to Rasa bot
        fetch("/chatbot/send/", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ sender: "user", message: message }),
        })
        .then(response => response.json())
        .then(data => {
            data.forEach(reply => {
                // Display bot response on the left
                appendMessage("bot", reply.text);
            });
        })
        .catch(error => console.error("Error:", error));
    };

    // Allow sending message with Enter key
    userInput.addEventListener("keypress", function (event) {
        if (event.key === "Enter") {
            sendMessage();
        }
    });
});
