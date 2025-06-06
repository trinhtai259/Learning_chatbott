import React, { useState } from "react";
import Message from "./message";
import "./chat.css";

const Chat = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [userName, setUserName] = useState("Local"); // Thêm tên người dùng
  const [canSend, setCanSend] = useState(true);

  const sendMessage = async () => {
    if (input.trim() === "" || !canSend) return;

    // Chặn người dùng gửi thêm tin nhắn
    setCanSend(false);

    // Thêm tin nhắn của người dùng vào danh sách
    setMessages((prevMessages) => [
      { text: input, type: "sent" },
      ...prevMessages,
    ]);

    const userMessage = input;
    setInput(""); // Xóa nội dung input sau khi gửi
    
    setMessages((prevMessages) => [
      { text: "Đang chờ...", type: "waiting" },
      ...prevMessages,
    ]);

    try {
      const response = await fetch("http://127.0.0.1:8000/api/data", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: userMessage }),
      });
  
      if (response.ok) {
        const data = await response.json();
  
        // Hiển thị phản hồi của chatbot
        setMessages((prevMessages) => [
          { text: data.reply, type: "received" },
          ...prevMessages.filter((msg) => msg.type !== "waiting"),
        ]);
      }
    } catch (error) {
      console.error("Lỗi khi gọi API:", error);
        // Thêm tin nhắn báo lỗi vào giao diện
      setMessages((prevMessages) => [
        { text: "Đã xảy ra lỗi khi gửi tin nhắn. Vui lòng thử lại.", type: "error" },
        ...prevMessages.filter((msg) => msg.type !== "waiting"),
      ]);
    }
    // Kích hoạt lại input sau khi AI phản hồi xong
    setCanSend(true);
  };
  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const handleChange = (e) => {
    setInput(e.target.value);
  };

  return (
    <div className="chat-container">
      <div className="chat-header">
        <div><a href="https://ctt.hust.edu.vn" className="link-ctt">Cổng thông tin</a></div>
        <div className="user-name">{userName}</div>
        <img src="\img\logo_hust_text.png"
             alt="HUST"
             className="header-image"/>
      </div>
      
      <div className="chat-messages">
        {messages.map((msg, index) => (
          (msg.type == "waiting" || msg.type == "sent"),
            <img
            src="\img\logo_bk.png"
            alt="AI Avatar"
            className="message-avatar"
            />,
          <Message key={index} text={msg.text} type={msg.type} />
        ))}
      </div>
      <div className="chat-input">
      <textarea
          value={input}
          onChange={handleChange}
          onKeyDown={handleKeyDown}
          placeholder="Viết câu hỏi cho tôi!"
          disabled={!canSend} // Vô hiệu hóa khi không thể gửi
        ></textarea>
        <button onClick={sendMessage} disabled={!canSend}>
        <img
              src="\img\message.png"
              alt="Đại học Bách Khoa"
              className="send-image"
            />
        </button>
      </div>
    </div>
  );
};

export default Chat;