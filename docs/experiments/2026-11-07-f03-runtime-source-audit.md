# F03：HF診断と実vLLM配信のnative weight意味・配置

**実imageのsource上、F03のpost-attention norm・post-FFW norm・layer_scalarは同じ直接係数と同じ位置へ適用される。数値同値は未証明で、正規化演算と丸め順序には具体的な実装差がある。** この照合はOpus 5.5の指摘に対する重みの意味・配置の確認であり、HF内の有限自己比較を配信能力得点へ置き換えない。

exact gemma-2 CID `fd4ec2eb…57bcf`、image `af9e746b…f7abe`、HF Transformers5.20.2/vLLM0.33.0を、`docker exec --user 1003 /usr/bin/python3 -B` の標準lib metadata/ファイル読取りだけで調べた。checkpoint payloadを読まず、Torch/model module import、モデルload、op実行、生成、GPU計算、サービス変更は0。最後もsameCID/imageでhealthy、読んだ15sourceのSHAは再照合で不変だった。全sourceの完全SHA・必要な行番号付き抜粋・実行原文を[記録](records/2026-11-07-f03-runtime-source/source-audit.json)へ残した。これはvLLM package全体の完全監査ではなく、明示した15sourceの範囲である。

| 対象 | HF | vLLM | source上の結論 |
|---|---|---|---|
| post_attention_layernorm.weight | attention出力へRMSNorm、weight.floatを直接乗算、その後residual加算 | 通常RMSNormのweightを直接乗算、その後residual加算 | 係数と位置が一致 |
| post_feedforward_layernorm.weight | MLP出力へRMSNorm、その後residual加算 | 同じbranch位置でRMSNorm、その後residual加算 | 係数と位置が一致 |
| layer_scalar | persistent Buffer、層末尾のhidden_states全体へ乗算 | persistent registered buffer、層末尾のhidden_states全体へ乗算 | skip/residualを含む層出力全体への係数で一致 |

HFのGemma4RMSNormはones初期化の`weight`を直接使う。vLLM Gemma4は`RMSNorm`をimportし、別classの`GemmaRMSNorm`（1+w）を使わない。actual AV configはhidden_size5376、60層、rms_norm_eps1e-6、PLE0、enable_moe_block=false/num_experts=null。PLE/MoE分岐はこのcheckpointで対象外である。F03 exact spec `e10018748de2795a8af03b98abb6e18362393a152c74f46b7a7892fe4975e878` の実編集は112tensors（post-attention58、post-FFW52、scalar2）。eligibleな180tensorsと実変更数を区別する。

vLLM registryはGemma4ForConditionalGenerationをgemma4_mmへ割当て、multimodal wrapperは`model.language_model.`を`language_model.model.`へ名前変換する。text-layer mapperのstack/dropはattention projections・MLP gate/up・KV shared用で、今回3familyを加算offsetや転置へ変換しない。AutoWeightsLoaderはnorm Parameterとpersistent scalar Bufferの両方を対象にし、専用weight_loaderのないこれらはdefault_weight_loaderの`copy_`（単一要素ならshape合わせ後copy_）を使う。BF16からFP32へのwideningがあれば値は保存される。読んだRMSNorm/decoder/wrapperに、これら3familyの追加weight変換・post-load hookは見つからなかった。loaderの一般quant/attention postprocess経路の存在は保持し、全モデルを独立GPU readbackしたとは称さない。

HFはFP32でRMSを計算し、`normed_output * self.weight.float()`の後に入力dtypeへcastする。vLLM IR nativeはFP32でnormを計算した後、`x.to(weight.dtype) * weight`で乗算し、入力dtypeへcastする。BF16 weightなら丸めが乗算前にも入る。HFはpow(-0.5)、vLLM IR nativeはrsqrtである。さらにvLLMにはvllm_c/oink等のIR実装登録があり、CUDA wrapperは生のweightを_C.rms_normへ渡す。今回、実選択kernel・compiler fusion・CUDA内部丸めを動かして調べていないため、native IRの差をそのまま実serving数値差の測定結果とはしない。

承認条件は「保存重みの意味と適用位置が一致」と解釈する。HF有限AV→F03→AVはHF内の反応測定として扱い、実serving効果は後続の同条件ordinary生成と全問公式WSPで別に測る。追加operator probeや新テストは作らず、source照合はここで閉じた。F03保存candidateとfinite実planはこの作業では作成していない。
