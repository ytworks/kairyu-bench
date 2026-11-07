# 行動に結び付く追加校正18件の提案（2026-11-07）

**本稿は実装前の設計。実装と著者CPU検証は[別記録](2026-11-07-behavior18-implementation.md)を参照。モデル呼出し・感度計測・solve・重み編集は本稿の作業では未実施。**
18件すべてを保持し、以下の12 train / 6 transferをモデル出力を見る前に固定する。
本稿のcorrectは実装すべき参照仕様、foilは意図した誤りであり、
「正答/foilの全数CPU検証済み」とはまだ称しない。
既存64件の高い合格率を、新しい微分目的の妥当性として扱わない。

## 実測から分かることと、まだ分からないこと

AVのcanonical dev30はコード72.9365、全90件採点。
HE152は生成途中の誤った重複定義がsanitizerに残り、
MBPP641/126は要求された関数名と実装名の不一致があった。
LCB arc190_cにはtranspose後の方向未変換、3777にはzero除去でindex依存が変わる
具体的なcode差が見える。ただしこれだけで全失敗の原因や関連する重みを断定しない。
SciCodeはshape/科学ライブラリ境界の失敗があり、失敗helperが必ず後続を壊すわけではない。

通常生成の既存48では46件合格し、失敗2件はbatched RHSの形状、
記号微分が定数化した際のscalar/vector組立てに集中した。
別のhelper16が合格しても、状態保持や順序依存が新規条件で既に十分とは限らない。
F02のproxyでは1 tokenの1件が正のmargin利得の約74%を占めた一方、
公式dev10の全26 codeは同点でcode NLLも悪化した。
したがって新稿では、正答文字列の尤度差と実行可能な行動を別々に測る。

これらは問題群を作る理由であり、特定層/familyを編集すべきという主張ではない。
公式benchmarkのgold、隠れたtest、参照実装は新問題へ転記しない。
下記は独立の式・入力・API契約による自作校正で、未見公式評価とは呼ばない。

## 共通の公開契約と分割

入力の意味・軸・dtype・空入力の可否・出力shapeをpromptに明記する。
全問は短いPython関数で、必要なNumPy/SymPy importを含め、説明や途中案を出さない。
数値出力はfloat64 ndarray、入力変更禁止。関数名を指定するC群も同じ数値出力にする。
特に「1つの最終定義のみ」を要求するC群では、Pythonの最後の定義優先だけで
合格にせず、公開契約どおりASTの定義数も検査する。

各群A/B/C/D/E/Fの番号1・2をtrain、番号3をtransferとする。
**train12=6群×2、transfer6=6群×1**。trainの1をfold0、2をfold1として、
単一のfold平均へまとめず全caseの値を保持する。
transferのprompt・correct/foil・数値seedはtrainと同時に固定するが、
gradient、係数選択、編集幅の調整に使わない。
transferを繰り返し見る場合は既見化を明記し、後から「未見」に戻さない。

全18件のprompt/answer/foilを同時にレビューし、モデルを見る前にSHAを固定する。
完全な正答frameとfoil frameを実際のtokenizer・templateで個別計数し、
**原則各1024 token以内、切り詰め禁止**。
説明・コードを短くするのは実行前だけで行い、超過が残ればそのcaseを落とさず、
全件保持の2048条件を明示再計画する。
現在はtoken数未測定であり、短い設計であることを適合証明としない。

## 18件の具体案

表のcorrect/foilはアルゴリズムと差分を一意に定める仕様。
実装時に各々を単独の完全な公開回答へし、共通fixtureの期待値とは別に検証する。
型や軸を曖昧にしてfoilを誤答にする問題は採用しない。

### A: バッチの系・右辺・出力軸

共通でAは各batchが対称正定値、batch B>=1、n>=1。入力はfloat32/64の両方、
出力float64を指定。反例が丸め誤差だけに依存しないよう良条件行列を使う。

| ID / split | 公開問題 | Correct仕様 | Foilと固定反例 |
|---|---|---|---|
| A1 train/fold0 | `solve(A,b,s)`。A=(B,n,n)、b=(B,n)、s=(B,)。各系 A[i]x[i]=s[i]b[i] を解き(B,n)を返す。 | RHS=(s[:,None]*b)[...,None]としてsolveし、最後の長さ1軸だけ除く。 | その長さ1軸を付けず2D RHSを渡す。B=3,n=2で形状の矛盾、B=nでも必ず正しくないことを試験。 |
| A2 train/fold1 | `solve(A,b,c)`。同じAで2 RHSを解き、出力[i,:,0]がb、[:,:,1]がc。 | `stack((b,c),axis=-1)`を(B,n,2)にしてsolve。 | `axis=1`で(B,2,n)にする。B=2,n=3、bとcが異なる非対角系で破綻を検出。 |
| A3 transfer | `solve(A,loads)`。loads=(B,n,q)。各解の二乗ノルムを(B,q)で返す。 | X=solve(A,loads)、n軸だけ二乗和。CPU期待値は各i,jを独立に解く。 | q軸を二乗和して(B,n)を返す。n=2,q=3、q=1、B=1を含む。 |

依存関係は「数式→バッチ/方程式/右辺軸→ライブラリの引数解釈→出力shape」。
A1だけでなくA2/A3も測り、単一の `[...,None]` 文字列への適応と区別する。

### B: 記号式の退化とscalar/vector broadcast

x,yは同長1D配列。空入力を許す場合は出力(0,k)を明記する。
SymPyで微分してから数値化する要求をpromptに置き、a=0も有効入力と明記する。

| ID / split | 公開問題 | Correct仕様 | Foilと固定反例 |
|---|---|---|---|
| B1 train/fold0 | `solve(x,a)`。f(t)=a*t²+tの導関数を各xで返す。shape=x.shape。 | diff→lambdify→評価後、結果をx.shapeへbroadcastしてcopy/float64化。 | `asarray(g(x))`だけ。a=0、x=[1,2,3]でscalar1になり、必要な[1,1,1]と異なる。 |
| B2 train/fold1 | `solve(x,y,a)`。f=a*x*y+y²の偏微分を[x偏微分,y偏微分]順に(N,2)へ。 | aを代入した式をdiff/lambdifyし、2結果をbroadcast_arraysしてcolumn_stack。 | broadcastせずcolumn_stack。a=0、N=4では0と2*yのscalar/vector結合が破綻。 |
| B3 transfer | `solve(x,a,b,c)`。f=a*t³+b*t+cの1階/2階微分を(N,2)へ。 | それぞれx.shapeへbroadcastしてから2列にする。a=0とa=b=0も対象。 | 2つの結果をそのままcolumn_stack。a=0,N=3なら両定数から(1,2)となり(N,2)を満たさない。 |

CPU期待値はそれぞれ2*a*x+1、[a*y,a*x+2*y]、[3*a*x²+b,6*a*x]を
NumPyで直接計算し、SymPy/lambdifyと独立にする。
出力shapeだけでなく、非退化aでも列順/値が一致することを要求する。
B群のゼロ係数は公開domainに含め、後付けの特殊testにしない。

### C: 実行入口と最終定義の契約

この群だけは `solve` でない具体的な関数名を指定する。
全回答でtop-level target関数は正確に1個、他のtop-level関数定義は不可、
必要なimportは許可。入力は数値列、出力はfloat64 ndarray。
名前だけのtoken marginを能力得点とせず、AST/interfaceと実行を別々に記録する。

| ID / split | 公開問題 | Correct仕様 | Foilと固定反例 |
|---|---|---|---|
| C1 train/fold0 | `positive_mass(xs)`。正の要素の総和をshape(1,)で返す。空列は[0]。 | 指定名の関数1個、[sum(max(x,0) for x in xs)]。 | 本体は同じだが関数名がsolve。xs=[-2,3,4]の期待[7]へ要求名で到達できない。 |
| C2 train/fold1 | `rotate_left(xs,k)`。整数k>=0だけ左循環。空列はshape(0,)。 | 空列を明示処理し、それ以外はk%len(xs)のslice結合、最終定義1個。 | 先に誤った同名定義、後にcorrectを重複定義。実行結果だけなら合格し得るが公開single-definition契約に反する。 |
| C3 transfer | `run_count(xs)`。隣接して同値が続くrunの個数をshape(1,)へ。空列は[0]。 | 0 if empty else 1+sum(xs[i]!=xs[i-1])、指定名1個。 | 関数名をcount_runsへ変更。本体正しくても要求APIは未提供。例[2,2,1,2]→[3]。 |

採点は「AST/入口契約」と「実行の数値結果」の2欄を保存し、混ぜない。
C2の失敗は算術能力の低下ではない。未対応のcheckerを黙ってexecution passにしない。

### D: 状態の持越し・順序・独立した状態

状態保持は現48の失敗原因として確定していない。
A/Bの局所primitiveが通っても組合せで失敗しないかを測る依存側のcontrolである。

| ID / split | 公開問題 | Correct仕様 | Foilと固定反例 |
|---|---|---|---|
| D1 train/fold0 | `solve(maps,start)`。maps=(T,n)の整数遷移表。時刻tでstate=maps[t,state]、各遷移後を(T,)へ。T=0も可。 | 1つのstateを順に更新し保存、入力不変。 | 毎回maps[t,start]へ戻す。maps=[[1,0],[0,1]],start=0でcorrect[1,1]、foil[1,0]。 |
| D2 train/fold1 | `solve(matrices,offsets,x)`。順にx←M[t]x+u[t]を適用し、最終xを返す。T=0はxのcopy。 | forward順で更新。CPU期待値は小行列のPython二重和で各stepを再構成。 | 逆順に適用。M0=[[1,1],[0,1]],M1=[[1,0],[1,1]],u=0,x=[1,0]ならcorrect[1,1]、foil[2,1]。 |
| D3 transfer | `solve(keys,deltas,initial)`。各eventは指定accountだけをdelta分更新し、その直後の残高を(N,)へ。keysは有効整数、負残高可。 | balances=initial.copy()を持ち、選ばれたkeyだけ更新。 | 各eventをinitial[key]+deltaから計算。initial=[10,20],keys=[0,1,0],delta=[1,2,3]なら[11,22,14]対[11,22,13]。 |

### E: 長方形・座標・方向の対応

最適化のためtransposeすること自体を要求せず、元座標の意味を守らせる。
どの解法を選んでも同じpublic contractを実行で確認する。

| ID / split | 公開問題 | Correct仕様 | Foilと固定反例 |
|---|---|---|---|
| E1 train/fold0 | `solve(grid,start,moves)`。N/E/S/Wのtorus移動を元grid座標で行い、開始点と各移動後の値の総和を(1,)へ。 | 元H/Wと方向dr/dcで逐次更新。 | H<W時にgrid/startをtransposeするが方向を交換しない。grid=[[1,2,3],[4,5,6]],start=(0,0),moves=ESでcorrect[8]、foil[10]。 |
| E2 train/fold1 | `solve(points,height,width)`。元HxW内の(N,2)整数座標を時計回り90度回転後の座標へ。出力(N,2)、N=0可。 | 各(r,c)を(c,H-1-r)へ。出力値はfloat64。 | HをWへ置換。H=2,W=3,point=(0,0)で(0,1)対(0,2)。 |
| E3 transfer | `solve(grid,queries)`。queriesは時計回り90度回転後の(W,H) grid上の座標。回転gridを作らず元gridから値を(N,)へ。 | query(r,c)→grid[H-1-c,r]。CPU期待値は独立に回転配列を明示構築してlookup。 | HをWへ置換。上の2x3gridでquery(0,0)の正答4に対し範囲外。正方形だけの試験では見逃す。 |

### F: ゼロ要素と元の位置・選択順序

| ID / split | 公開問題 | Correct仕様 | Foilと固定反例 |
|---|---|---|---|
| F1 train/fold0 | `solve(xs)`。元index0を+とし、交互符号和をshape(1,)へ。空列[0]。 | enumerateした元indexで符号を決める。 | ゼロ除去後にenumerate。[2,0,5]はcorrect[7]、foil[-3]。 |
| F2 train/fold1 | `solve(values,k)`。元event番号1から数え、k番ごとのvalueの和を(1,)へ。k>=1、zero eventも1件を占める。 | values[k-1::k]の和。 | ゼロ除去してからslice。[0,4,0,8],k=2でcorrect[12]、foil[8]。 |
| F3 transfer | `solve(xs,k)`。順序を保って正確にk個を選び、選択順の交互符号和を最大化。0<=k<=n<=10、k=0は[0]。 | 許された小domainでは全組合せを列挙して最大化できる。独立DPでも照合。 | ゼロを削ってから同じ処理。[2,0,5],k=3はcorrect[7]だがfoilには3要素の選択がない。 |

F3は大規模計算量の能力試験ではない。元index、選択順、zeroの役割を切り分ける
小さな合成問題で、公式LCB問題のgoldや入力を使わない。

## 実装する場合の最小CPU検証

新規の大きなharnessは作らず、既存sandboxのbounded workerとcontainer制約、
公開回答抽出・shape/dtype/入力保全比較を再利用する。
新caseのsample/referenceはモデル固有recipe内の1ファイルへ置く。
C群だけ必要なASTの関数名/定義数検査を行い、契約を満たす名前が1個ある場合だけ
隔離container内用codeへsolve=要求名のaliasを追記し、同じDocker.run/WORKERを使う。
missing/duplicateをrenameや定義選択で救済しない。
生成コードをhostでexecしない。aliasを付ける前のpublic code SHAも保持する。
この拡張はまだ実装していない。

モデルを見る前のsource-owned correct/foil検証は全18件を対象とし、
(1) 表の固定反例、(2) 境界suite、(3) 固定seedの小さな数値suiteを実施する。
correctは全suite合格、各foilは少なくとも対応する固定反例で必ず落ちる必要がある。
foilだけが有効な代替仕様になっていないか、prompt/domainを同時にレビューする。
誤りがあればモデルを見る前に新revisionへ直し、旧版と失敗を残す。

境界suiteはB/n/qの一致・不一致、B=1/n=1/q=1、指定された空列、
zero係数、float32/64、非対角/非可換行列、長方形両向き、先頭/中間/末尾zero、
繰返しkeyと負deltaを明示的に含める。
全組合せで大きくせず、各case8個の固定witness+8個のseed付き生成例を初期案とする。
train seed=202611071、transfer seed=202611072を事前固定する。
実数比較は既存rtol=2e-6/atol=2e-7、離散個数・座標はexact値も別検査する。
一致/不一致の失敗を平均で隠さず、全case×全trialの結果を保存する。

## 実行・proxyの使い方と次へ進むための条件案

実装承認時の事前修正：最初のAV通常thinking生成は**train12件だけを1回**とする。
transfer6件は候補spec/幅を凍結した後にAVと候補で各1回生成する。
transferから方向・幅を選ばない。全18件の著者正答/foilを事前CPU検証・SHA固定する点は同じ。
[修正前の提案](records/2026-11-07-behavior18-inputs-v1/proposal-before-implementation.md)を残す。
温度等は配信条件、thinkingは実server/template attestationを固定する。
この提案は実行許可ではない。全cases/token/source検証の後に具体的planを別途reviewする。
未回答/HTTP失敗/timeoutはunknownであり、0点を入れない。
全caseを残し、出力を見た後の除外・再分割・foil交換はしない。

本案の狙いは「AVで行動失敗があり、正答・foilの意味が検証済みのcaseがあるか」を確認すること。
また近天井なら、それ自体を結果として残し、平均marginを伸ばすための編集を始めない。
個別failureが見つかっても、その1件だけで候補を作るとは決めない。
6群のpass/unknown/契約違反と、状態・軸を跨ぐ依存の対応を見て初めてjoint編集目的を決める。
一般能力の保持は既存全領域WSPで測り、この18件で代用しない。

後の感度/有限probeは全caseを単独に識別し、LL・margin・長さ・元のpair rankingと
実行passを並べる。既存forced-empty-thoughtレシピは通常thinkingの生成文脈と
同一ではないため、使う場合はその条件差を残し、LL改善を行動改善に換算しない。
今回のproposalは既存F02の目的値・層/family制限・採用ゲートを再利用しない。

少なくとも「最大caseの利得割合」「case別絶対delta」「6群の内訳」
「1件を除いた場合の記述統計」を全件併記し、
平均改善だけで候補を昇格させない。除外統計は診断専用で、原集計を置き換えない。
transfer6は編集条件の調整に使わず、凍結候補ができた後の別確認とする。
具体的な重み、幅、最適化制約、採用閾値は本稿ではまだ提案しない。

証拠：[N30全件監査](2026-11-07-n30-independent-audit.md)、
[通常生成48の実行結果](2026-11-07-authored48-execution-results.md)、
[自作120の契約・集中監査](2026-11-07-authored-case-audit.md)。
