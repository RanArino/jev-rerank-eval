# rerank-eval

ベクトル検索の上位候補を判断モデルで並べ直す「リランキング」が、本当に検索の質を上げるのかを、**凍結したデータで何度でも同じ条件で**確かめるための小さなレポジトリです。

ブログ記事「ベクトル検索の並びをJevで並べ直すと本当に良くなるのか」の再現用です。判断モデル Jev 1.13.0、Cloudflare の Clef、LLM（gpt-6-luna）による Yes/No 判断の三通りの結果を同梱しています。

## 何が入っているか

| パス | 内容 |
|---|---|
| `data/corpus.json` | 架空の会社「Larkspur Analytics」の英語文書 41 件（このために書いたオリジナル文書で、第三者の著作物は含みません） |
| `data/queries.json` | 調整用（tuning）の質問 60 問。5 カテゴリ × 12 問。正解は文書中の**引用文**と等級（3/2/1）で持ちます |
| `data/pools.json` | 各質問の候補 150 件を、元のベクトル検索の順で凍結したもの（221 チャンク） |
| `results/jev-1.13.0.json` | Jev 1.13.0 が各候補に返した「役に立つ」確率（上位 150 件 × 60 問 = 9,000 件） |
| `results/clef.json` | Clef が返した「役に立つ」確率（上位 60 件 × 60 問 = 3,600 件、うち 12 件は 5 秒で時間切れ） |
| `results/gpt-6-luna-medium.json` | gpt-6-luna（reasoning effort medium）が返した Yes/No（上位 60 件 × 60 問 = 3,600 件、うち 4 件は応答なし） |
| `src/rerank_eval/metrics.py` | 指標（被覆判定・nDCG・Recall・対応のあるブートストラップ）。標準ライブラリのみ |
| `src/rerank_eval/evaluate.py` | 元の順位と並べ直した順位を同じ規則で採点し、ゲート判定まで出す |
| `src/rerank_eval/run.py` | Jev または LLM に候補を一件ずつ問い合わせる（費用上限つき、再開可能） |

> [!NOTE]
> 評価用（evaluation）の 60 問とその正解は含めていません。採用判断のために封印しているデータだからです。

## すぐに試す（API キー不要）

```sh
uv run --extra dev pytest -q                      # 指標の手計算例と、公開結果（Jev と LLM）の再現
PYTHONPATH=src python3 -m rerank_eval.evaluate results/jev-1.13.0.json
PYTHONPATH=src python3 -m rerank_eval.evaluate results/jev-1.13.0.json --top-k 60   # Clef・LLM と同じ候補で比べる
PYTHONPATH=src python3 -m rerank_eval.evaluate results/clef.json
PYTHONPATH=src python3 -m rerank_eval.evaluate results/gpt-6-luna-medium.json
```

`evaluate` は次を出力します。

- `overall` / `categories`: nDCG@10・Recall@12・文脈に選んだ候補の Recall（元の順位と並べ直し後）
- `bootstrap_95`: nDCG@10 の差の 95% 区間（10,000 回、シード 20261006）
- `gates`: 事前に決めた採用ゲート（平均改善 ≥ 0.03、区間の下限 > 0、カテゴリ劣化 ≤ 0.02、Recall が全体でもカテゴリ別でも下がらない）
- `top_k_prefixes`: 上位 20/40/60/100/150 件だけを並べ直した場合
- `relevance_floor`: 確率の下限を 0〜1 で変えたときの、文脈の Recall と精度

調整用データでの結果（三つとも同じ上位 60 件の候補で比較）:

| | 元の順位 | Jev 1.13.0 | Clef | gpt-6-luna medium |
|---|---:|---:|---:|---:|
| 返す値 | | 確率 | 確率 | Yes/No |
| nDCG@10 | 0.753 | **0.919** | 0.851 | 0.870 |
| nDCG@10 の差（95% 区間） | | **+0.165**（0.098〜0.239） | +0.098（0.022〜0.172） | +0.116（0.065〜0.172） |
| Recall@12 | 0.935 | 0.977 | 0.944 | 0.960 |
| ゲート | | すべて通過 | Recall で不合格（追質問 1.00 → 0.92、曖昧 0.54 → 0.52） | Recall で不合格（曖昧 0.54 → 0.48） |
| 1 回の呼び出し（p50 / p95） | | 171 / 230 ms | 453 / 1,109 ms | 1,590 / 3,223 ms |
| 応答なし（3,600 件中） | | 0 | 12（5 秒で時間切れ） | 4 |
| 費用（3,600 回） | | 約 0.066 ドル | 0.215 ドル | 0.165 ドル |

Jev は上位 150 件（9,000 件）で計測し、`--top-k 60` で同じ候補に切り出しています。上位 150 件のままでは nDCG@10 0.912（差 +0.159、95% 区間 0.090〜0.234）、費用は 0.164 ドルです。Jev の 3,600 回分の費用は、実測の 9,000 回分から按分した値です。

文脈に入れる下限確率（`relevance_floor`）は実装ごとに較正します。Recall を落とさない最大の下限は、Jev が 0.25（精度 0.12 → 0.52）、Clef が 0.40（精度 0.11 → 0.53）でした。Yes/No の LLM は下限 0 より大きければ「No」を除外するだけです。

LLM は確率を返さないので、Yes の候補を元の順位のまま前に出すだけです。同じ「Yes」の中では順位を付けられないことが、Jev との差の主な理由だと考えています。

## 自分で問い合わせる

```sh
uv sync --extra jev --extra llm
cp .env.example .env
# .env に、実行する実装の認証情報を設定する
uv run --env-file .env python -m rerank_eval.run jev --output results/my-jev.json --max-usd 0.5
uv run --env-file .env python -m rerank_eval.run clef --top-k 60 --output results/my-clef.json --max-usd 1
uv run --env-file .env python -m rerank_eval.run llm --effort low --top-k 60 --output results/my-llm.json --max-usd 1
uv run python -m rerank_eval.evaluate results/my-jev.json
```

- API キーの設定場所はリポジトリ直下の `.env` です。初回に `.env.example` から作成してください。`.env` は Git の管理対象外なので、実キーをコミットしません。Jev の実行には `TYPESAFE_API_KEY`、Clef の実行には `CLOUDFLARE_API_TOKEN` と `CLOUDFLARE_ACCOUNT_ID`、LLM の実行には `OPENAI_API_KEY` が必要です。Cloudflare ダッシュボードの Workers AI で REST API を選ぶと、トークンと Account ID を取得できます。
- 変更前は設定ファイルを読まず、シェルの環境変数だけを SDK に渡していました。具体的には `TYPESAFE_API_KEY=... uv run ... jev` または `OPENAI_API_KEY=... uv run ... llm` のように、コマンドの先頭で一時的にセットする方法です。`AsyncTypeSafeClient` と `AsyncOpenAI` はそれぞれこれらの標準環境変数を自動で読み取ります。Clef はこの変更で初めて直接実行に対応し、Cloudflare Workers AI REST API を使用します。
- 1 回の呼び出しごとに、最悪の費用（リクエストの UTF-8 バイト数 + 1,024 トークン、出力上限）で残額を確認してから送ります。`--max-calls` と `--max-usd` を超える前に止まります。
- 回答は `<output>.jsonl` に一件ずつ書いてから次に進むので、途中で止めても同じ候補に二重に課金しません。
- SDK の自動再試行は切っています（1 回の呼び出し = 1 HTTP リクエスト）。失敗した呼び出しは最悪の費用で計上し、結果に「回答なし」として残します。
- Clef の同梱結果（`results/clef.json`）は Scaler 本番の Clef アダプタ（Cloudflare Workers AI）経由で取得したものです。`run clef` は Cloudflare Workers AI REST API で同じ `state`・`noul` 質問を送ります。採点はほかの結果と同じく API キーなしで再現できます。
- LLM への問い合わせは、Scaler 本番の LLM フォールバックと同じ developer プロンプト・質問・厳密な JSON スキーマ（`{"useful": boolean}`）です。同梱の `gpt-6-luna-medium.json` は、本番のアダプタ経由で取得したものです。
- 単価は 2026-10-07 時点の公開価格です（Jev 入力 $0.042/100 万トークン、Clef 入力 $0.24/100 万トークン、gpt-6-luna 入力 $0.10・出力 $0.50。Jev と Clef は出力課金なし、LLM の推論トークンは出力に含まれます）。

## 採点の規則

- **被覆**: 候補が正解の引用文の文字の半分以上を含めば、その正解を覆っているとみなします（2,000 文字を超える節は 200 文字の重なりで分割されるため）。
- **nDCG@10**: 利得 `2^grade − 1`、割引 `log2(rank + 1)`。理想値は候補集合全体から作ります。
- **Recall**: 等級 2 以上を「関連あり」とし、分母は候補集合が到達できる関連箇所で固定します。正解のない質問は満点にせず、採点から外します。
- **並べ直し**: 確率の高い順（Boolean の回答は 1 か 0）、同点は元の順位。回答のない候補は最後に元の順で並べます。
- **TopK**: 結果ファイルの回答数（1 問あたり）が、その結果の候補集合の大きさです。nDCG の理想値は常に上位 150 件全体から作ります。
- **文脈の選択**: 並びに沿って、本文が重複する候補と下限未満の候補を飛ばし、最大 12 件・1 文書 4 件まで選びます。

## 候補集合の作り方と限界

本番の検索を一度だけ手元で再現し、その結果を凍結しました。

1. 見出し単位のチャンク分割（2,000 文字・重なり 200 文字）で 41 文書を 221 チャンクに分割
2. `text-embedding-3-small`（1536 次元）でチャンクと検索用の質問を一度だけ埋め込み
3. 全チャンクとのコサイン類似度で厳密に順位をつけ、上位 150 件を保存

前の会話を受けた追質問は、質問と履歴だけを見て書いた単独で意味の通る文に書き換えてあります（`search_query`）。

限界:

- 合成した英語コーパスで、実際の文書の多様さ（表、PDF 由来の崩れなど）はありません。
- 本番の近似最近傍検索とは、僅差の順位が入れ替わることがあります。検索の遅延も測れません。
- いずれも「同じ候補集合を二通りに並べて比べる」ことには影響しません。
