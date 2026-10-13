FROM python:3.12-slim-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends fonts-noto-cjk fontconfig \
    && fc-cache -fv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PYTHONPATH=/app/src \
    MPLCONFIGDIR=/tmp/matplotlib \
    PYTHONUNBUFFERED=1

RUN python -c "from matplotlib import font_manager as fm; names={f.name for f in fm.fontManager.ttflist}; assert any('Noto' in name and 'CJK' in name for name in names), sorted(name for name in names if 'Noto' in name)"

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','8501')+'/', timeout=4)"
CMD ["python", "app.py"]
