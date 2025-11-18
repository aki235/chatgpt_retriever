# 初期設定

- ChatGPT から会話履歴をエクスポートして解凍
- OPENAI_API_KEY を環境変数で設定
- `python ./utils/pre_processing.py <会話履歴ディレクトリのpath>`を実行
  - API 利用料金がかかるので注意(100 会話あたり 1 円くらい)

# 検索

- `python retrieval.py <コマンド>` or `python retrieval.py --embedding-file <conversations_embeddingのpath> --textonly-file <conversations_textonly.jsonのpath> <コマンド>`
  - コマンド一覧
    - `max`: 利用可能な最大インデックス表示
    - `summary <index>`: 会話の概要表示
    - `retrieve <index>`: 会話の JSON 取得
    - `details <index>`: 詳細な会話内容表示
    - `search <query>`: 類似検索（上位 5 件）
