# Usa la imagen oficial de desarrollo de Microsoft para Python 3.11
FROM mcr.microsoft.com/devcontainers/python:1-3.11-bullseye

# Configura Git para evitar problemas de saltos de línea (CRLF/LF) en Windows
RUN git config --global core.autocrlf true

COPY requirements.txt /tmp/pip-tmp/

RUN pip install -r /tmp/pip-tmp/requirements.txt && rm -rf /tmp/pip-tmp