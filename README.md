# Andersan Core

大気汚染物質（オキシダント）の予測AIシステム

## 概要

Andersan Coreは、気象データと過去の大気測定値を組み合わせて、将来の大気汚染物質濃度を予測する機械学習ベースのPythonパッケージです。

## 機能

- 過去24時間の大気測定データの収集と分析
- 気象予報データの取得と処理
- 機械学習モデルによる大気汚染物質（OX）濃度の予測
- 地理院タイルベースの地理空間データ処理

## 必要条件

- Python 3.10以上
- Poetry（依存関係管理）

## インストール

```bash
poetry install
```

## 使用例

```python
from andersan_core.predict import predict_ox

# 神奈川県の大気汚染予測を実行
prediction = predict_ox(
    prefecture="kanagawa",
    isodate="2024-04-01T12:00:00",
    model="andersan0_2",
    forecast_hours=24
)
```

## データソース

- OpenMeteo API（気象データ）
- Air Monitor（大気汚染測定データ）

## ライセンス

MIT License - 詳細は[LICENSE](LICENSE)ファイルをご覧ください。

## 作者

vitroid <vitroid@gmail.com>