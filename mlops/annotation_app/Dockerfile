# ベースイメージとしてPythonの公式イメージを使用
FROM python:3.10.2-buster

USER root
EXPOSE 8080

RUN apt-get update && apt-get upgrade -y
RUN apt-get -y install locales && \
    localedef -f UTF-8 -i ja_JP ja_JP.UTF-8
ENV LANG ja_JP.UTF-8
ENV LANGUAGE ja_JP:ja
ENV LC_ALL ja_JP.UTF-8
ENV TZ JST-9
ENV TERM xterm

ARG POETRY_HOME="/opt/poetry"
ARG POETRY_VERSION="1.6.1"

RUN mkdir -p /root/app
COPY ./ /root/app
WORKDIR /root/app

RUN curl -sSL https://install.python-poetry.org/ | python3 - --version ${POETRY_VERSION} && \
    ln -s ${POETRY_HOME}/bin/poetry /usr/local/bin/poetry && \
    poetry config virtualenvs.create false

# 必要なファイルをコンテナ内にコピー
RUN mkdir -p /root/app
COPY ./ /root/app
WORKDIR /root/app

RUN poetry install --no-root --no-ansi

# アプリケーションを実行
# CMD ["python", "main.py"]