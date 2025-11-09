# 🐍 Base image
FROM python:3.10-slim

# 📁 Create app directory
WORKDIR /app

# 🧠 Copy files
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 📦 Copy project files
COPY . .

# ⚙️ Expose Streamlit port
EXPOSE 8501

# 🏃 Run Streamlit app
CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
