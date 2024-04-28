apt update &&  apt install curl python-is-python3 pip -y
curl https://ollama.ai/install.sh | sh
ollama serve &
ollama pull nomic-embed-text
ollama pull mistral
ollama pull nomic-embed-text
pip install -r requirements.txt
ollama pull nomic-embed-text
python3 app.py