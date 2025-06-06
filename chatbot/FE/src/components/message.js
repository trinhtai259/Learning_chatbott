import React from "react";
import "./message.css";

const Message = ({ text, type, avatar }) => {
  return (
    <div className={`message-container ${type}`}>
      <div className={`message ${type}`}>
        <div
        dangerouslySetInnerHTML={{
          __html: text.replace(/\n/g, "<br>"),
        }}
      /></div>
    </div>
  );
};

export default Message;
