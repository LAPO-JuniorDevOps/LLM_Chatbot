// === DOM Elements ===
const chatbotToggle = document.getElementById('chatbotToggle');
const chatbot = document.getElementById('chatbot');
const sendBtn = document.getElementById('sendBtn');
const userInput = document.getElementById('userInput');
const chatBody = document.getElementById('chatBody');
const expandBtn = document.getElementById('expandBtn');
const expandIcon = document.getElementById('expandIcon');
const collapseIcon = document.getElementById('collapseIcon');
const userIcon = 'static/img/green-user.png';
const botIcon = 'static/img/grey-bot.png';

let isChatOpen = false;

window.addEventListener('beforeunload', function () {
  fetch('/reset-session/default/');
});

// === Toggle Chatbot Visibility ===
chatbotToggle.addEventListener('click', () => {
  isChatOpen = !isChatOpen;
  chatbot.style.display = isChatOpen ? 'flex' : 'none';

  chatbotToggle.innerHTML = isChatOpen
    ? getCloseIcon()
    : getChatIcon();

  if (isChatOpen) {
    setTimeout(() => userInput.focus(), 100);
  }
});

// === Expand/Collapse Chatbot ===
expandBtn.addEventListener('click', () => {
  chatbot.classList.toggle('expanded');
  const isExpanded = chatbot.classList.contains('expanded');

  if (expandIcon && collapseIcon) {
    expandIcon.style.display = isExpanded ? 'none' : 'inline';
    collapseIcon.style.display = isExpanded ? 'inline' : 'none';
  }
});

// === Send Message Logic ===
sendBtn.addEventListener('click', sendMessage);
userInput.addEventListener('keypress', (event) => {
  if (event.key === 'Enter') {
    event.preventDefault();
    sendBtn.click();
  }
});

// Format Chatbot Response using marked + DOMPurify
function formatMessage(message) {
  const rawHTML = marked.parse(message);
  return DOMPurify.sanitize(rawHTML);
}

function addDate(messageElement, type){
  
//Creating the time element
const now = new Date();
let hours = now.getHours();
let minutes = now.getMinutes();
const ampm = hours >= 12 ? 'PM' : 'AM';

hours = hours % 12 || 12;
minutes = minutes < 10 ? '0' + minutes : minutes;

const timeStr = `${hours}:${minutes} ${ampm}`;

const timeElement = document.createElement('span');
// timeElement.classList.add('time');
if (type === 'user') {
  timeElement.classList.add('time-white');
} else {
  timeElement.classList.add('time');
}
timeElement.textContent = timeStr;

messageElement.appendChild(timeElement);
}

// === Send Message Handler ===
function sendMessage() {
  const message = userInput.value.trim();
  if (message === '') return;

  // Display user's message
  const userRow = createMessageRow('user', message, userIcon);
  chatBody.appendChild(userRow);

  // Show typing indicator
  const typingRow = createTypingIndicator();
  chatBody.appendChild(typingRow);
  chatBody.scrollTop = chatBody.scrollHeight;

  // Get bot reply from server
  getBotReply(message, typingRow);
  userInput.value = '';
}

// === Fetch Bot Reply from Server ===
function getBotReply(message, typingRow) {
  fetch("/chatbot/send/", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sender: "user", message }),
  })
    .then(response => response.json())
    .then(data => {
      typingRow.remove();

      // --- MODIFICATION START ---
      // Concatenate all reply texts into a single markdown string
      const fullBotReplyMarkdown = data.map(reply => reply.text).join('\n\n'); // Use '\n' to ensure new lines between parts
      // --- MODIFICATION END ---

      // Create a single message row for the concatenated reply
      const botRow = createMessageRow('bot', fullBotReplyMarkdown, botIcon, addDate);
      chatBody.appendChild(botRow);
      chatBody.scrollTop = chatBody.scrollHeight;
    })
    .catch(error => {
      typingRow.remove();
      const errorRow = createMessageRow('bot', "⚠️ Oops! Something went wrong.", botIcon);
      errorRow.querySelector('.bot-message').classList.add('error');
      chatBody.appendChild(errorRow);
      chatBody.scrollTop = chatBody.scrollHeight;
      console.error("Error fetching bot reply:", error);
    });
}


// function getBotReply(message, typingRow) {
//   fetch("/chatbot/send/", {
//     method: "POST",
//     headers: { "Content-Type": "application/json" },
//     body: JSON.stringify({ sender: "user", message }),
//   })
//     .then(response => response.json())
//     .then(data => {
//       typingRow.remove();
//       data.forEach(reply => {
//         const botRow = createMessageRow('bot', reply.text, '🤖', addDate);
//         // addDate(botRow);
//         chatBody.appendChild(botRow);
//         chatBody.scrollTop = chatBody.scrollHeight;
//       });
//     })
//     .catch(error => {
//       typingRow.remove();
//       const errorRow = createMessageRow('bot', "⚠️ Oops! Something went wrong.", '🤖');
//       errorRow.querySelector('.bot-message').classList.add('error');
//       // addDate(errorRow);
//       chatBody.appendChild(errorRow);
//       chatBody.scrollTop = chatBody.scrollHeight;
//       console.error("Error fetching bot reply:", error);
//     });
// }

// === Utility Functions ===

// Create message row
function createMessageRow(type, message, iconChar) {
  const row = document.createElement('div');
  row.className = `message-row ${type}`;

  const msg = document.createElement('div');
  msg.className = `${type}-message`;
  msg.innerHTML = formatMessage(message);
  // msg.textContent = message;
  addDate(msg, type);


  const icon = document.createElement('div');
  icon.className = `icon ${type}-icon`;
  icon.innerHTML = `<img class = 'icon' src="${iconChar}">`;
  // icon.textContent = iconChar;

  if (type === 'user') {
    row.appendChild(msg);
    row.appendChild(icon);
  } else {
    row.appendChild(icon);
    row.appendChild(msg);
  }

  return row;
}


// Create typing indicator
function createTypingIndicator() {
  const row = document.createElement('div');
  row.className = 'message-row';

  const icon = document.createElement('div');
  icon.className = 'icon';
  icon.innerHTML = `<img class = 'icon' src="${botIcon}">`;
  // icon.textContent = '🤖';

  const typing = document.createElement('div');
  typing.className = 'typing-indicator';
  typing.innerHTML = `
    <div class="typing-dot"></div>
    <div class="typing-dot"></div>
    <div class="typing-dot"></div>
  `;

  row.appendChild(icon);
  row.appendChild(typing);
  return row;
}

// Chatbot Toggle Icons
function getChatIcon() {
  return `
    <svg xmlns="http://www.w3.org/2000/svg" fill="none"
         viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="heroicon">
      <path stroke-linecap="round" stroke-linejoin="round"
            d="M8.625 12a.375.375 0 1 1-.75 0 .375.375 0 0 1 .75 0Zm0 0H8.25m4.125 0a.375.375 0 1 1-.75 0 .375.375 0 0 1 .75 0Zm0 0H12m4.125 0a.375.375 0 1 1-.75 0 .375.375 0 0 1 .75 0Zm0 0h-.375M21 12c0 4.556-4.03 8.25-9 8.25a9.764 9.764 0 0 1-2.555-.337A5.972 5.972 0 0 1 5.41 20.97a5.969 5.969 0 0 1-.474-.065 4.48 4.48 0 0 0 .978-2.025c.09-.457-.133-.901-.467-1.226C3.93 16.178 3 14.189 3 12c0-4.556 4.03-8.25 9-8.25s9 3.694 9 8.25Z" />
    </svg>
  `;
}

function getCloseIcon() {
  return `
    <svg xmlns="http://www.w3.org/2000/svg" fill="none"
         viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="size-6">
      <path stroke-linecap="round" stroke-linejoin="round"
            d="M6 18 18 6M6 6l12 12" />
    </svg>
  `;
}
