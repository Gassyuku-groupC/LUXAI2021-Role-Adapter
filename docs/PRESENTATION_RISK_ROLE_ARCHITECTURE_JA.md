# Lux AI Agent 発表用説明資料

## 1. 最終エージェント

最終版は単一モデルではなく、マップサイズごとに検証済み checkpoint を切り替える `routed_teacher_final` である。

| マップ | 使用モデル | Rot180 | Role bias |
|---|---|:---:|:---:|
| 12x12 | `er100_35072` | ON | OFF |
| 16x16 | `role_05376_nofs` | ON | ON |
| 24x24 | `log_03584` | ON | ON |
| 32x32 | `role_05376_nofs` | ON | ON |

```text
ゲーム状態
  -> マップ別 checkpoint 選択
  -> 24-block ResNet Actor
  -> Rot180 推論統合
  -> Role-City の正方向 soft bias
  -> 合法手 mask / 衝突処理
  -> 行動 command
```

12x12 は序盤の拡張テンポを守るため Role bias を無効化する。16/24/32 では Role-City を有効化する。

## 2. 中心的な提案: Plug-and-Play Role-City

既存 Actor の checkpoint 構造を変更せず、Actor が出力した action logits に小さな正方向 bias を加える。

```text
FinalLogits = LegalMask(Rot180ActorLogits + PositiveRoleDelta)
```

- `update()` は毎ターンの役割と cooldown を更新する。
- `apply()` は合法な logits だけを調整する。
- Role モジュールは action を直接強制しない。
- 無効化すれば元の Actor 経路へ戻せる。

## 3. Worker の役割

| Role | 目的 |
|---|---|
| Harvester | 資源採集と資源地点への移動 |
| Builder | 都市拡張機会の支援 |
| Firefighter | 危険都市への移動と隣接味方 unit 間の燃料 relay |
| Attacker | 敵 Worker への位置圧力。優先度は最下位 |

役割変更には原則 5 ターンの cooldown がある。Firefighter は本当に危険な都市に対してのみ cooldown を上書きできる。

## 4. City の役割

| Role | 目的 |
|---|---|
| FuelDepot | 燃料輸送の到着先 |
| ResearchStation | 研究 action の優先 |
| ManufacturingPoint | Worker 生産の優先 |
| SacrificialDecay | 厳格な条件を満たす最大 1 都市だけを損失候補とする |

FuelStation は削除した。Lux では都市 fuel は一方向の備蓄であり、都市から unit へ取り出せない。`transfer` は隣接する味方 unit 間だけである。

## 5. 拡張能力の保護

Lux は序盤の 1 回の BUILD_CITY 差が後半に大きな都市差へ拡大する。したがって最終版では次を守る。

- BUILD_CITY への固定負 bias を禁止する。
- `preserve_build_city_logit: true` を使用する。
- Role は正方向 guidance のみを適用する。
- 12x12 では Role bias 自体を bypass する。

## 6. 学習方法

1. `role_05376_nofs + Rot180` を安定した出発点とする。
2. Actor backbone を凍結し、Role bias / Local Adapter を学習する。
3. 必要な場合だけ Policy Head を低 learning rate で解凍する。
4. `best_agent` を弱い KL anchor として固定する。
5. `first` など実行可能な opponent と APPO + V-trace で学習する。
6. loss ではなく固定 seed・両 side の対戦結果で checkpoint を選ぶ。

## 7. Risk Sidecar / Gate

空間 Risk Sidecar と zero-init Intervention Gate は実装済みである。

- Actor feature は `detach()` して backbone への gradient を遮断する。
- Pooled-KV attention により 32x32 の計算量を抑える。
- Gate は additive logit delta を出力し、最後に合法手 mask を適用する。

ただし、対戦評価で性能向上を確認できなかったため、最終 `routed_teacher_final` では両方 OFF である。最終成績を Risk Gate の効果として説明してはいけない。

## 8. 最終評価

`first` に対して seeds `20260920` から `20260929`、4 マップ、両 side を評価した。

- 有効 73 局、timeout 7 局（すべて 32x32）
- 38 勝、完了局勝率 52.1%
- 平均 Score 150.49
- 平均都市差 +7.86
- 平均 unit 差 +8.26

Score の定義は次である。

```text
Score = 最終 city tiles + 最終 units
```

12x12 の勝率と 32x32 の timeout 安定性は残る課題である。

## 9. 発表時の要点

1. 中心的な新規性は説明可能で着脱可能な Role-City モジュールである。
2. 元 Actor の強い拡張能力を壊さないことを最優先した。
3. 単一 checkpoint を無理に全マップへ適用せず、paired evaluation に基づいて routing した。
4. 実装した機能と最終版で有効な機能を区別して説明する。
5. timeout は敗北へ混ぜず、完了局勝率と別に報告する。
