# 初期設定
- ChatGPTから会話履歴をエクスポートして解凍
- OPENAI_API_KEYを環境変数で設定
- ```python ./utils/pre_processing.py <会話履歴ディレクトリのpath>```を実行
    - API利用料金がかかるので注意(100会話あたり1円くらい)

# 検索
- ```python retrieval.py <コマンド>``` or ```python retrieval.py --embedding-file <conversations_embeddingのpath> <コマンド>```
    - コマンド一覧
        - max: 利用可能な最大インデックス表示
        - summary <index>: 会話の概要表示
        - retrieve <index>: 会話のJSON取得
        - details <index>: 詳細な会話内容表示
        - search "<query>": 類似検索（上位5件）