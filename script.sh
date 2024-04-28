apt update &&  apt install curl python-is-python3 pip -y
curl https://ollama.ai/install.sh | sh
ollama serve &
ollama pull nomic-text-embed
ollama pull mistral
pip install -r requirements.txt
python3 app.py