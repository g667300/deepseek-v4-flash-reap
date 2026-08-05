# deepseek-v4-flash-reap

English version: [README.md](README.md)

[DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
を REAP(Router-weighted Expert Activation Pruning)で圧縮するパイプライン。
256 experts/層 → 128 experts/層(50%プルーニング)、166.9GB → **82.4GB**。

目的は2つある。ひとつは **DGX Sparkで動かすこと**。82.4GBなら統合メモリ128GBに
配信の余裕を残して載るが、元の166.9GBでは載らない。統合メモリ128GBのPCでも
動くかもしれないが、こちらでは試していない。

もうひとつは **校正データを自前で作れること**。REAPは自分のデータが実際に使う
expertを残すので、何が生き残るかは与えた配合で決まる。ここでの配合
(`calib/mix-dsv4.json`)は日本語・英語・コード寄りで、これを差し替えることが
結果を別の用途に向ける正規の方法である。

## 一般的なREAPの回し方との違い

公式実装([llm-compressor の REAP modifier](https://arxiv.org/abs/2510.13999))は
transformers 経由でロードしたモデルに対する1パスとして動く。校正データを流し
ながらルーティングを記録し、パスが到達した時点でそのMoEブロックをその場で刈る。

**この経路はこのチェックポイントをそのままでは読めない。** テンソル名が
DeepSeekネイティブ形式(`layers.{L}.ffn.experts.{E}.w1.weight` であって
`model.layers.*.mlp.*` ではない)で、量子化もDeepSeek独自の規約
(ルーテッドexpertはFP4 = E2M1・K方向32要素ブロック・E8M0スケール、非expert部は
FP8 = E4M3・128×128ブロック)であり、compressed-tensorsが読む形式ではないためだ。

**ただしこれは変換の問題であって、行き止まりではない。** HF名+BF16へ逆量子化して
公式実装を回し、出力を再量子化すれば、プルーニング済みモデルは得られる。サイズも
障害にならない — llm-compressorはサブグラフを1つずつオンロードし、残りをディスクへ
オフロードできる。代償は**約570GBの中間生成物**と、REAPが触ってすらいない重みまで
含めて全部を再符号化する往復である。

このリポジトリはもう一方の選択肢を採り、2つのパスに分けている:

* **スコアリングパス**(`reap_saliency_dsv4.py`、GPU必要) — 層を1つずつ辿り、
  校正データを通すあいだだけその層を逆量子化して、expertごとのsaliencyを書き出す。
  モデルには一切手を加えない。
* **サージェリパス**(`reap_surgery.py`、CPUのみ) — そのスコアを読んで
  safetensorsのシャードを書き直し、残すexpertだけを取り出す。

**この分割が成立するのはREAP自体の性質による。生存expertの重みを一切変更しない**
からだ。プルーニングはexpertリストと対応するルーター行のスライスにすぎない。
だからサージェリはファイルレベルの変換で表現でき、**元のFP4/FP8の符号化をそのまま
保てる**(逆量子化と再量子化の往復がなく、GPUも要らない)。

公式実装のために書いておくと、**この分割はもともと「公式経路ではこのチェックポイントを
処理できない」と思い込んだ結果である**。それは読み違いだった — 上に書いたとおり
変換の問題であり、サイズもサブグラフ単位のオンロードで扱える。
**それでもこの思い違いには思わぬ収穫があった。** バイト列を保ったことで、
**元のチェックポイントが、出力を検証するための参照として残った**のである。
生存expertは元とビット単位で一致していなければならず、それを確かめるのが
`scripts/verify_against_source.py` である。ここで使った非ECCのマシンでは、これが
公開済みモデルにまで届いていた実際のビット化けを検出した。形式を往復する経路は
この手段を失う。往復後はバイト列が違って当然であり、比べる相手が残らないからだ。

## 必要な環境

実際に一通り動かすのに要るもの:

| | |
|---|---|
| RAM | **64GB 必須**、128GB 推奨 |
| VRAM | **11GB**(CUDA) |
| 空きディスク | **約290GB** |
| 時間 | 計算はRAM 128GBで約40分・64GBで70分以上、ほかに 166.9GB のダウンロード |

数値はデスクトップ(RTX 4090)での実測。**DGX Spark でもパイプライン自体を
実行できると思われるが、今回は試していない**(Sparkは配備先としてしか
使っていない)。

**この作業にはECCメモリを推奨する。** 出力82.4GBを公開時のSHA256と突き合わせた
ところ、**全体を通すたびに違うファイルが不一致になった**(1回目は48個中1個、
2回目は別の2個で、1回目に失敗したものは合格した)。単独で読み直せば必ず正しい値が
出る。おおよそ**30GB読むごとに1回**の割合で、カーネルログには何も出ない
(MCEもEDACもNVMeエラーもない)。ECCなしのデスクトップRAMを逼迫させた状態で、
同日に2回ハードフリーズしているマシンでの出来事である。

これが問題なのは、化けが起きうる場所のためだ。**サージェリパスがシャードを書く
最中に1ビット化ければ、それはチェックポイントに焼き込まれる**。手元のコピーと
公開物を比べても見つからない — どちらも同じバイト列に由来するからだ。
ECCなしで回すなら、**出力は元モデルと突き合わせて検証すること**。REAPは生存
expertに触れないので、そのバイト列は元のチェックポイントと完全に一致するはずである。

配備はこれとは別枠で、82.4GBのチェックポイントをメモリに載せられるホストが要る。
これらの数字の根拠と各ステージ個別の要件は[実行環境の目安](#実行環境の目安)に。

成果物は [noooop/DeepSeek-V4-Flash-REAP-noMTP](https://huggingface.co/noooop/DeepSeek-V4-Flash-REAP-noMTP)
で公開している。

この数値が何を示していて何を示していないか、手法の制約、まだ確かめていないことは
[`docs/TECHNICAL_NOTES.jp.md`](docs/TECHNICAL_NOTES.jp.md) にまとめてある。要点だけ:

| 評価 | 結果 | 基準 |
|---|---|---|
| 日本語 JCommonsenseQA(1,119件フル) | 0.9088 ± 0.0086 | ランダム0.20 |
| 英語 MMLU(診断570件) | 0.6526 ± 0.0194 | ランダム0.25 |
| 中国語 global_mmlu_zh(400件フル) | 0.4975 ± 0.0250 | ランダム0.25 |
| perplexity(held-out 262,016トークン) | 6.5891 | — |
| RULER 4K/16K/32K/64K(12タスク平均) | 98.68 / 96.10 / 96.93 / 94.51 | 実効判定閾値85.6 |

検索能力はプルーニングの影響を受けていない(niah系8タスクは32Kまで全長さで1.000)。
落ちるのは検索した箇所の読解で、`ruler_qa_squad` が4長さすべてで最下位、
4Kの0.842から64Kの0.542まで下がる。

## セットアップ

```bash
python3 -m venv .venv
.venv/bin/pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
.venv/bin/pip install -r requirements.txt
```

動作確認済み: torch 2.11.0+cu128 / transformers 5.14.1 / lm_eval 0.4.12。
`CUDA_VISIBLE_DEVICES` の指定が必要な場合がある(複数GPU構成でtorch非対応の
GPUが混在する場合)。

## 実行環境の目安

このページの所要時間を測った環境と、各ステージが実際に必要とするもの。

| ステージ | 必要なもの | 実測 |
|---|---|---|
| 校正データ生成 | CPUのみ、トークナイザ | 数分 |
| スコアリングパス | CUDA GPU、VRAMは11GB未満。**律速はホストRAM**(256×2048でキャッシュ21.5GB) | RTX 4090 + ホスト64GB: `--spill`なしで51s/層(43層で約37分)、ありで約78s/層(約55分) |
| サージェリパス | GPU不要・逆量子化不要。ディスクは元と出力の**両方**(166.9 + 82.4GB)。ワーカーあたり2シャードが同時に載る | 数分、ディスク律速 |
| 配備 | 統合メモリ121GiB(DGX Spark)、82.4GBのチェックポイントが常駐 | [配備](#sparkまたはvllm配備)を参照 |

スコアリングパスはGPUのFLOPsではなく**ホスト側の作業で律速している**。層あたり数十GBの
活性化をホストキャッシュからgatherし、約34GBをPCIe越しに動かす一方、expertの
行列積自体はカードのピークの数%しか出ていない(24GBのVRAMを11GBしか使わない)。
**したがって速いGPUに投資してもほとんど効かず、効くのはホストのメモリ容量と帯域。**

スコアリングパスを統合メモリ機(配備側)で回した場合は**未計測**。同程度か若干遅い程度と
予想している。

## モデルの取得

```bash
./scripts/dl.sh deepseek-ai/DeepSeek-V4-Flash-0731 models/DeepSeek-V4-Flash-0731 start
./scripts/dl.sh deepseek-ai/DeepSeek-V4-Flash-0731 models/DeepSeek-V4-Flash-0731 status
```

## パイプライン

```bash
# 校正データ生成(日英コード中心、DeepSeek独自のプロンプト形式でレンダリング)
.venv/bin/python scripts/build_calibration.py --mix calib/mix-dsv4.json \
    --tokenizer models/DeepSeek-V4-Flash-0731 --out artifacts/calib.pt

# スコアリングパス: 層逐次のsaliency計算(GPU必要、キャッシュが収まるホストで約35分)
.venv/bin/python scripts/reap_saliency_dsv4.py \
    --ckpt models/DeepSeek-V4-Flash-0731 --tokens artifacts/calib.pt \
    --max-samples 256 --out artifacts/saliency.json

# サージェリパス: safetensorsのファイルレベル書き換え(GPU不要、数分)
.venv/bin/python scripts/reap_surgery.py \
    --src models/DeepSeek-V4-Flash-0731 --dst artifacts/dsv4-reap50 \
    --saliency artifacts/saliency.json --sparsity 0.5 \
    --mtp drop --hash-layers prune-remap
```

### スコアリングパスのホストメモリ

スコアリングパスはGPUよりホストメモリを食う(VRAMピークは10GB程度)。校正データの
活性化キャッシュを丸ごとシステムメモリに持つため、**256×2048トークンで21.5GB**。
大半はhidden stateで、hyper-connections(`hc_mult=4`)により通常モデルの4倍になる。

128GBあれば上記コマンドをそのまま実行できる。**64GBはギリギリ**で、
キャッシュとプロセスの常駐分を足すと余裕がほとんど残らない。実際、他の作業と
並走させた状態でホストごとフリーズした(OOMキラーが動かずカーネルが
スラッシングしたため、ログに何も残らない)。64GBのホストでは次のように回す:

```bash
systemd-run --user --scope -p MemoryMax=42G -p MemorySwapMax=0 \
    .venv/bin/python scripts/reap_saliency_dsv4.py \
    --ckpt models/DeepSeek-V4-Flash-0731 --tokens artifacts/calib.pt \
    --max-samples 256 --out artifacts/saliency.json \
    --spill /var/tmp/dsv4-spill --state artifacts/saliency-state.pt --state-every 3
```

* `--spill DIR` はキャッシュの実体をファイルバックの共有mmapにする。匿名メモリ
  ではなく回収可能なページキャッシュになるため、逼迫しても回収で済む。
  コストは**最低でも45%増、しかもそこから伸びる**。序盤は層あたり77s
  (spillなしは53s)だが、層25では120sまで落ちた。hidden stateを毎層書き戻すため
  `--state`込みで**層あたり35〜40GBを書く**ことになり、コンシューマSSDは走り
  切る前にSLCキャッシュが枯れる。以降はwritebackで待たされる(実測: 0.7〜1.1GB/s、
  デバイス使用率90%、書き込みレイテンシ60〜100ms)。**律速はRAMではなく
  SSDの持続書き込み速度**。55分ではなく70分以上を見ておくこと。
* `--state FILE --state-every N` はN層ごとにhidden stateとトラッカーを保存する。
  中断してもやり直しではなく再開できる。
* スコープでプロセスに上限を掛けておくと、暴走してもマシンごとではなく
  プロセスだけが死ぬ。上限には到達し回収し続けるが、それが意図した挙動で、
  匿名メモリは1.6GB程度に留まりOOMキルは発生しない。

`--max-samples 512` にするとキャッシュは43GBになり、64GBホストではページ
キャッシュに収まらない。層あたり86GB程度の実ディスクI/Oが発生する。

**ハッシュルーティング層(先頭3層)の扱いに注意**: DeepSeek-V4は`n_routed_experts`
がモデル全体で1つの値のため、他の層を刈るならハッシュ層も同じ数まで刈らざるを
得ない。`--hash-layers prune-remap`はこの3層についても刈った上で、削除された
expertのトークンIDを「ルーター重みが最も似ている生存expert」へ再割り当てする
(balanced割り当てで負荷を均等化)。詳細は`reap_surgery.py`のdocstringと
`docs/TECHNICAL_NOTES.jp.md`を参照。

## テスト

GPU不要:

```bash
.venv/bin/python scripts/test_quant.py
.venv/bin/python scripts/test_reap_surgery_dsv4.py
.venv/bin/python scripts/test_scoring_pass_dsv4.py
```

## Spark(またはvLLM)配備

別ホストへ運ぶには `scripts/push_model.sh <src> <host:/dest>` を使う。rsyncを
リトライループで包んだもので、ここでは82GBの転送がssh越しに35〜45GBごとに
切れたため、再開できることが実質的な必須要件だった。転送先に以前のビルドが
ある場合は `--checksum` を付ける。サージェリパスは全シャードを書き直すので、
中身がほとんど同じでもmtimeは全部変わるため。

```bash
docker run -d --gpus all --ipc=host -p 8000:8000 \
    -v artifacts/dsv4-reap50:/model:ro \
    --entrypoint vllm <vllm_image> serve /model \
    --served-model-name dsv4-reap50 --gpu-memory-utilization 0.75 \
    --max-model-len 65536 --max-num-seqs 16 --kv-cache-dtype fp8
```

**`--gpu-memory-utilization` は統合メモリ機では意味が違う。** その割合がシステムRAM
から取られるため、vLLM 既定の 0.9 は 121GiB の機体で約109GiBを予約し、OSには
8GiB しか残らない。すると長文脈が破綻する: 32K・16並列での実測で、エンジンの
KVキャッシュ使用率が 4〜6% しかないのにデコードが 0.1〜2.2 tok/s まで落ちた
(KVプールは律速ではなかった)。その状態で2つ目のクライアントを向けたところ、
ホストごとハングし電源の入れ直しが必要になった。0.75 なら約28GiBが残り安定する。

`--kv-cache-dtype fp8` は必須(DeepSeek-V4のsparse MLAカーネルの制約)。
vLLM 0.25.1標準イメージは同梱FlashInfer(0.6.13)が自身のコードと噛み合わず
ロード時にクラッシュするため、`flashinfer-python==0.6.14`を追加した派生
イメージが必要(`FLASHINFER_DISABLE_VERSION_CHECK=1`でcubinの不一致を無視)。
詳細は`docs/TECHNICAL_NOTES.jp.md`を参照。

## 評価

JCommonsenseQAは原本のJSONを直接読む(HFデータセットがスクリプト形式で
`datasets>=4`では実行できないため)。初回のみ取得しておく:

```bash
./scripts/fetch_eval_data.sh
```

その後、リポジトリルートから:

```bash
MODEL=dsv4-reap50 TOKENIZER=models/DeepSeek-V4-Flash-0731 TAG=dsv4-50 \
    scripts/run_eval_suite.sh
```

各スクリプトの接続先は既定で`http://localhost:8000`。別ホストで動かしている
場合は`BASE`で指定する(例: `BASE=http://10.0.0.5:8000/v1/completions`)。

日本語常識(JCommonsenseQA)・英語知識(MMLU)・15言語(global_mmlu)・
perplexity・長文脈(RULER 4k〜64k)を安い順に流す。
`SKIP="ruler"`のようにステージ名を空白区切りで渡すと省略できる。

## ライセンス

コード: MIT([LICENSE](LICENSE))。
元モデル(DeepSeek-V4-Flash-0731)のライセンス・利用条件は
[配布元](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)を参照。
