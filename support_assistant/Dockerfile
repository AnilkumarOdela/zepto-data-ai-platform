FROM python:3.12-slim
WORKDIR /app
COPY support_assistant/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY support_assistant /app/support_assistant
ENV MOCK_LLM=1
ENV PYTHONUNBUFFERED=1
EXPOSE 7860
CMD ["uvicorn", "support_assistant.main:app", "--host", "0.0.0.0", "--port", "7860"]
