# movie-recommender

*[English](README.md) | 日本語*

IMDb の「Your Ratings」CSV エクスポートをもとに、TMDB 由来のジャンル＋キーワードを
特徴量として映画をレコメンドする、コンテンツベースのレコメンダーです。

自分の IMDb レーティングを起点に、その評価傾向から嗜好プロファイルを作り、TMDB API から
ライブ取得した未視聴の候補プールをランキングします。小さな CLI パイプラインとして
動作します。

## セットアップ

```bash
uv sync
# どちらか一方を設定（トークンが推奨）。いずれも以下から取得:
# https://www.themoviedb.org/settings/api
export TMDB_API_TOKEN=eyJ...     # v4「API リードアクセストークン」(Authorization: Bearer)
# export TMDB_API_KEY=xxxxxxxx   # v3 API キー (api_key クエリパラメータ)
```

IMDb からレーティングをエクスポート（Your Ratings → ⋯ → Export）し、CSV を
`data/raw/movies.csv` に配置してください。`data/` は git 管理対象外なので、あなたの
レーティングはローカルに留まり、コミットされることはありません。

> Python は uv 管理のため、必ず `uv run` 経由で実行します。

## パイプライン

```bash
uv run movie-recommender enrich                   # ① TMDB → ジャンル＋キーワード取得（認証要）
uv run movie-recommender similarity               # ② コサイン類似度行列
uv run movie-recommender neighbors --title "Up"   #    評価済み作品の近傍を表示
uv run movie-recommender candidates               # ③ TMDB から未視聴の候補プールを生成
uv run movie-recommender recommend --from-tmdb     # ④ その候補を嗜好順にランキング
```

`recommend`（`--from-tmdb` なし）と `similarity` / `neighbors` は TMDB 認証なしでも
動作し、ジャンルのみの特徴量にフォールバックします（精度は落ちますがネットワーク不要）。
`candidates` と `recommend --from-tmdb` は認証が必要です。

### 生成物（アーティファクト）

| パス | 生成コマンド |
|------|------------|
| `data/interim/enriched.parquet` | `enrich` |
| `data/interim/similarity.parquet` | `similarity` |
| `data/interim/candidates.parquet` | `candidates` |
| `recommend --output PATH` の CSV | `recommend`（既定は標準出力のみ） |

## アーキテクチャ

| モジュール | 役割 |
|--------|----------------|
| `ratings.py` | IMDb エクスポートの読み込み・正規化（Const キー） |
| `tmdb.py` | TMDB API クライアント（失敗は握り潰さず明示／リトライ／映画・TV対応） |
| `enrich.py` | TMDB のジャンル/キーワード付与・解決状況をレポート |
| `features.py` | ジャンル＋キーワード → L2正規化したスパース特徴量行列 |
| `similarity.py` | 全ペアのコサイン類似度（単一行列・Const キー） |
| `candidates.py` | TMDB のレコメンドから未視聴候補プールを構築 |
| `recommend.py` | 評価重み付きの嗜好プロファイル → 候補ランキング |
| `pipeline.py` | 中間生成物の読み書き |
| `cli.py` | `movie-recommender` エントリポイント |

## チューニング

- `--genre-weight`（既定 3.0）: キーワードに対するジャンルの重み。
- `candidates --min-rating / --max-seeds / --max-candidates`: どの高評価作品を
  シードにし、プールをどこまで広げるか。
- `candidates --media-type movie,tv`: 残すメディア種別（既定: 全種別）。
- `candidates --language ja,en`: 残す原語の ISO-639-1 コード（既定: 全言語）。
  どちらのフィルタも特徴量取得の前に適用されるため、除外された候補は API 呼び出し
  コストがかかりません。
- `recommend --output PATH`: ランキングを CSV にも書き出す（既定は標準出力のみ）。
- `recommend --top-n N`: 返す件数。

```bash
# 例: 日本語の映画のみに絞り、CSV へ保存
uv run movie-recommender candidates --media-type movie --language ja
uv run movie-recommender recommend --from-tmdb --output data/processed/recommendations.csv
```

## 仕組みのポイント

- **Const(tt-id) をキー**に全工程を貫通させ、ファイル名（タイトル＋年）依存を排除。
- 距離は**行ごとの min-max 正規化を行わず**、L2正規化済みベクトルのコサイン類似度
  [0,1] をそのまま使用（作品間でスコアが比較可能）。
- レコメンドは `Your Rating` を**平均中心化した重み**で嗜好プロファイル化。平均より
  高評価の作品は内容を引き寄せ、低評価の作品は遠ざける。スコアが負の候補は「自分の
  平均より低く付けた作品に近い」＝相対的に好みから外れる、という意味。
- IMDb スクレイピング（cinemagoer）は現在機能しないため、特徴量ソースは TMDB API。

## 今後の候補

- 候補プール内のフランチャイズ／続編の重複排除。
- TMDB の人気度／平均スコアをブレンドし、ニッチだが似ている作品が広く愛される作品を
  常に上回らないように調整。
- 嗜好プロファイルの永続化と、オフライン評価（leave-one-out）の追加。

## ライセンス

[MIT](LICENSE)。本プロダクトは TMDB API を利用していますが、TMDB によって承認・認定
されたものではありません。
