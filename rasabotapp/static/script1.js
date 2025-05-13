document.addEventListener("DOMContentLoaded", function () {
    const messagesContainer = document.getElementById("messages");
    const userInput = document.getElementById("user-input");

    function appendMessage(sender, message) {
        const messageElement = document.createElement("div");
        messageElement.classList.add("message", sender);
        messageElement.innerHTML = `<p>${message}</p>`;

        //Creating the time element
        const now = new Date();
        let hours = now.getHours();
        let minutes = now.getMinutes();
        const ampm = hours >= 12 ? 'PM' : 'AM';

        hours = hours % 12 || 12;
        minutes = minutes < 10 ? '0' + minutes : minutes;

        const timeStr = `${hours}:${minutes} ${ampm}`;

        const timeElement = document.createElement('span');
        timeElement.classList.add('time');
        timeElement.textContent = timeStr;

        messageElement.appendChild(timeElement);

        //Typing Indicator
        var typer = (sender == 'bot') ? "LAPO's AI Assistant": "You";

        const TypingIndicator = document.createElement('div');
        TypingIndicator.classList.add(`${sender}-typing`);
        TypingIndicator.textContent = `${typer}`;

        messagesContainer.appendChild(TypingIndicator);
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

        // 👉 Create and show typing indicator
        const typingIndicator = document.createElement("div");
        typingIndicator.classList.add("typing-indicator");
        typingIndicator.innerHTML = `
            <div class="bubble typing">
                <span class="dot"></span>
                <span class="dot"></span>
                <span class="dot"></span>
            </div>
        `;
        typingIndicator.id = "bot-typing";
        messagesContainer.appendChild(typingIndicator);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;

        // Send message to Rasa bot
        fetch("/chatbot/send/", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ sender: "user", message: message }),
        })
        .then(response => response.json())
        .then(data => {
            // Remove typing indicator
            const typing = document.getElementById("bot-typing");
            if (typing) typing.remove();

            data.forEach(reply => {
                // Display bot response on the left
                appendMessage("bot", reply.text);
            });
        })
        .catch(error => {
            console.error("Error:", error)
            const typing = document.getElementById("bot-typing");
            if (typing) typing.remove();
        });
    };

    // Allow sending message with Enter key
    userInput.addEventListener("keypress", function (event) {
        if (event.key === "Enter") {
            sendMessage();
        }
    });
});