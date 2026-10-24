# The code-mode sandbox: Python and runner.py, nothing else. No recon package, no database
# driver, no keys. Built with this folder as its context: `make sandbox-image`.
FROM python:3.13.16-slim-bookworm
ENV PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1
COPY runner.py /runner.py
USER 65534:65534
ENTRYPOINT ["python", "-I", "-u", "/runner.py"]
