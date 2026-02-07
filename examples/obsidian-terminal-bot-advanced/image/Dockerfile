# MetaClaw Obsidian bot image
# Business code is baked into this image layer; agent.claw only references the image and entry command.
FROM docker.io/library/python@sha256:036871e8860c254533e1d4c2842568f19a56d1afbaed99653ee6206bf9491f6e

WORKDIR /app
COPY bot/chat_once.py /app/chat_once.py
RUN chmod 0755 /app/chat_once.py

ENTRYPOINT ["python3", "/app/chat_once.py"]
