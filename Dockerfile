# Image tái tạo submission LegalQA row1 (private LB 0,6520), chạy OFFLINE hoàn toàn.
# Model/dữ liệu không nằm trong image: mount gói repro-row1 vào /repro (xem repro-row1/README.md).
#
#   docker build --platform linux/amd64 -t legalqa:row1 .     # offline được: wheel nằm ở wheels/
#   docker save legalqa:row1 | gzip > legalqa-row1.tar.gz      # mang sang máy offline
#   docker load < legalqa-row1.tar.gz
#
# Python 3.12: pyproject yêu cầu >=3.12,<3.13 và Kaggle (nơi sinh candidate) chạy 3.12;
# image pytorch/pytorch chính thức còn 3.11 nên dựng từ python:3.12-slim + wheel torch CUDA.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_RETRIES=10 PIP_DEFAULT_TIMEOUT=120 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
    NLTK_DATA=/app/nltk_data PYTHONPATH=/app/src:/app/scripts
WORKDIR /app

# Cài từ wheels/ đóng gói sẵn, --no-index: build không cần mạng. Tải torch CUDA (~2,5 GB kèm
# nvidia-*) qua mạng trong docker build từng đứt giữa chừng 2 lần. --require-hashes khoá
# đúng từng byte wheel.
# Bind-mount thay COPY: wheel không thành layer, image không phình thêm 2,5 GB.
COPY requirements-lock.txt .
RUN --mount=type=bind,source=wheels,target=/wheels \
    pip install --no-index --find-links /wheels --require-hashes -r requirements-lock.txt

COPY pyproject.toml LICENSE CONTEXT.md COMMIT.txt ./
COPY src ./src
COPY scripts ./scripts
COPY third_party ./third_party
# wordnet + omw-1.4 cho METEOR: nltk.download không chạy được khi offline.
COPY nltk_data ./nltk_data
# Không `pip install -e .`: build isolation sẽ đòi tải setuptools qua mạng; PYTHONPATH đã trỏ src.
RUN python -c "import legalqa, torch, transformers, peft; from nltk.corpus import wordnet; assert wordnet.synsets('law')"

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["entrypoint.sh"]
CMD ["mbr"]
