from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from chat_bot import reply

app = FastAPI()

# Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Origin frontend
    allow_credentials=True,
    allow_methods=["*"],  # Cho phép tất cả method: GET, POST, etc.
    allow_headers=["*"],  # Cho phép tất cả header
)

# Định nghĩa model dữ liệu gửi từ frontend
class Message(BaseModel):
    message: str

# Route API
@app.post("/api/data")
async def post_data(user_message: Message):
    # Gọi hàm reply từ chat_bot
    try:
        response = reply(str(user_message.message))
        
        return {"reply": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Route mặc định
async def root():
    return {"message": "FastAPI server is running!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app)
