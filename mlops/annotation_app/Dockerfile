# ベースイメージとしてPythonの公式イメージを使用
FROM python:3.9-slim

# 作業ディレクトリを設定
WORKDIR /app

# 必要なファイルをコンテナ内にコピー
COPY requirements.txt requirements.txt
COPY main.py main.py
COPY app.yaml app.yaml
COPY google-storage-service-account.json google-storage-service-account.json
COPY templates/ templates/

# Pythonパッケージをインストール
RUN pip install --no-cache-dir -r requirements.txt

# 環境変数を設定
ENV GOOGLE_APPLICATION_CREDENTIALS="/app/google-storage-service-account.json"

# アプリケーションを実行
CMD ["python", "main.py"]