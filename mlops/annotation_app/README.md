### dockerに入る
```
docker compose up -d
```
```
docker exec -it annotation-app bash
```

### poetry
```
poetry install
```

### 実行
```
poetry run python3 src/main.py
```

### デプロイ
```
cd cloudbuild && sh cloudbuild.sh
```