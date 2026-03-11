# WarZone Tower Defense — Reinforcement Learning Agent

使用強化學習（Reinforcement Learning）訓練一個能自動玩 WarZone Tower Defense Extended 的 Agent。目標是讓 Agent 在 Level 4 "Enclave" 地圖上盡可能存活最多波數。

![Map](enclave_map.png)

## 專案概述

本專案包含：

1. **遊戲模擬器** — 完整重建 WarZone Tower Defense Extended 的遊戲邏輯（Python + Pygame）
2. **RL 環境** — 基於 Gymnasium 的 RL 環境，支援 Action Masking
3. **訓練系統** — 使用 MaskablePPO（sb3-contrib）訓練 Agent
4. **模仿學習** — 支援錄製人類玩家示範，並用 Behavioral Cloning 預訓練

### 地圖參數

- 地圖：Level 4 "Enclave"（僅地面單位）
- 模式：Quick Cash
- 網格：51×31
- 敵人出生點：左側 27 個（x=0, y=2~28）
- 目標點：(45, 5)
- 障礙物：469 格
- 可建塔位置：926 格（塔佔 2×2）
- 可建牆位置：1,084 格

### 塔的種類（7 種）

| 塔 | 縮寫 | 花費 | 特性 |
|----|------|------|------|
| Machine Gun | MG | $200 | 快速射擊，單體 |
| Cannon | CN | $400 | 範圍傷害 |
| Freezer | FR | $300 | 減速敵人 |
| Sniper King | SK | $500 | 長距離高傷害 |
| Laser | LS | $600 | 光束持續傷害 |
| Anti-Tank | AT | $1,000 | 對坦克高傷害 |
| Plasma | PL | $800 | 高 DPS |

## 架構

```
├── simulator/          # 遊戲引擎
│   ├── game_engine.py  # 主引擎：tick 循環、建塔/升級/放牆
│   ├── game_config.py  # 遊戲設定：塔屬性、敵人屬性、波次資料
│   ├── game_map.py     # 地圖：障礙物、可建造位置
│   ├── towers.py       # 塔邏輯：攻擊、升級、傷害追蹤
│   ├── enemies.py      # 敵人邏輯：移動、血量、路徑跟隨
│   ├── pathfinding.py  # BFS 尋路（支援牆壁改變路徑）
│   ├── wave_controller.py # 波次生成控制
│   └── renderer.py     # Pygame 繪圖
│
├── rl/                 # 強化學習
│   ├── td_env.py       # Gymnasium 環境（觀察/動作/獎勵）
│   ├── train.py        # MaskablePPO 訓練腳本
│   ├── replay.py       # 視覺化回放工具
│   ├── pretrain.py     # Behavioral Cloning 預訓練
│   └── record_demo.py  # 錄製人類玩家示範
│
├── runs/               # 訓練紀錄（每個 run 一個資料夾）
├── main.py             # 互動式 Pygame 遊戲（人類玩家用）
└── requirements.txt
```

## RL 環境設計

### 觀察空間（3,438 維）

| 區塊 | 維度 | 內容 |
|------|------|------|
| 全域資訊 | 15 | 金錢、生命、波數、塔/牆數量、路徑長度等 |
| 塔資訊 | 200×17 = 3,400 | 每座塔的位置、類型、等級、傷害、效率等 |
| 敵人統計 | 8 | 敵人數量、平均血量、最近敵人距離等 |
| 波次資訊 | 15 | 當前波次的敵人類型與數量 |

### 動作空間（7,767 個離散動作 + Action Masking）

| 動作類型 | 數量 | 說明 |
|----------|------|------|
| NOOP | 1 | 不做任何事 |
| BUILD | 6,482 | 7 種塔 × 926 個位置 |
| UPGRADE | 200 | 升級已有的塔（最多 200 座） |
| WALL | 1,084 | 在可用位置放牆 |

使用 Action Masking 確保 Agent 只能選擇合法動作（有足夠金錢、位置未被佔用等）。

### 獎勵設計（最終版本）

經過 20 次迭代最終確定的獎勵結構：

| 獎勵 | 數值 | 說明 |
|------|------|------|
| 擊殺普通敵人 | +1.0 | 基本擊殺獎勵 |
| 擊殺 Boss | +5.0 | Boss 額外獎勵 |
| 敵人漏出 | -5.0 | 放走普通敵人 |
| Boss 漏出 | -20.0 | 放走 Boss |
| 波次完成 | +3.0 + wave × 0.1 | 越後期的波次完成獎勵越高 |
| 升級塔 | DPS 提升 × 3 | 基於實際 DPS 增量的獎勵 |
| 放牆（有效） | 路徑增量 × 0.3 | 牆壁確實延長路徑才有獎勵 |
| 放牆（無效） | -0.3 | 懲罰無意義的牆壁 |
| 升級壓力 | -0.005 × 可升級數 | Wave 20 後懲罰不升級的行為 |
| 塔效率懲罰 | -0.01 | 傷害/投資比 < 5 的塔 |
| 遊戲結束 | -50.0 | 生命歸零 |

## 訓練歷程

總共訓練了 **20 個 Run**，投入 **267.6 小時**（11.2 天），累計 **2.09 億步**。

### 訓練結果總覽

| Run | 名稱 | 時間 | 步數 | 最佳波數 | 主要改動 |
|-----|------|------|------|---------|---------|
| 1 | baseline | 24h | 5M | 101 | 初版：score-based，100 波上限 |
| 2 | survival | 8h | 5M | 117 | 改為生存目標，移除建塔固定獎勵 |
| 3 | explore | 22h | 10M | 137 | 網路加大 512×512，entropy ×3 |
| 4 | walls | 14h | 10M | ~130 | 加入牆壁動作 + 塔類型多樣性獎勵 |
| 5 | balanced | 8h | 10M | ~125 | 調整獎勵平衡 |
| 6 | survival | 3h | 4M | ~100 | 實驗性，提前終止 |
| 7 | unlimit | 11h | 10M | ~130 | 解除塔數量限制（20→200） |
| 8 | imitation | 11h | 10M | ~120 | 第一次模仿學習 |
| 9 | imitation2 | 17h | 15M | ~130 | 第二次模仿學習 |
| 10 | upgrade | 16h | 15M | ~135 | 調整升級獎勵 |
| 11 | forceupgrade | 5h | 5M | ~110 | 強制升級機制，提前終止 |
| 12 | upgrade3x | 6h | 5M | ~115 | 3 倍升級獎勵，提前終止 |
| 13 | cashpenalty | 20h | 15M | ~140 | 囤錢懲罰 |
| 14 | buildreward | 15h | 12M | ~140 | 基於路徑覆蓋率的建塔獎勵 |
| 15 | pathaware | 17h | 15M | ~142 | 路徑感知建塔 |
| **16** | **balanced** | **19h** | **15M** | **145.7** | 綜合平衡版 |
| **17** | **imitation** | **11h** | **10M** | **165.6** | Behavioral Cloning + RL fine-tune |
| 18 | quality | 6h | 6M | 151.6 | 升級壓力 + 低傷懲罰，失敗退化 |
| 19 | efficiency | 1h | 0.5M | — | 效率獎勵實驗，立即終止 |
| **20** | **fresh** | **36h** | **31M** | **155.4** | 從頭訓練，效率獎勵 |

### 波數進步曲線

```
Run  1: ████████████████████ 101 波（基線）
Run  2: ███████████████████████ 117 波
Run  3: ███████████████████████████ 137 波
Run 16: █████████████████████████████ 145.7 波
Run 20: ██████████████████████████████ 155.4 波
Run 17: ████████████████████████████████ 165.6 波 ← 最佳
人類:   ██████████████████████████████ 151 波
```

### 關鍵發現

#### 1. 模仿學習是最大的突破

Run 17 用人類玩家的 3 場示範做 Behavioral Cloning 預訓練，再用 RL fine-tune，僅花 10.8 小時就達到 165.6 波，超越人類玩家（151 波）。

相比之下，純 RL（Run 1~16）花了 200+ 小時，從 101 波慢慢爬到 145.7 波。

#### 2. 獎勵工程的教訓

經歷了 20 次迭代才學到：
- **移除錯誤的獎勵比增加新獎勵更重要**：固定建塔獎勵導致 Agent 洗塔騙分；現金懲罰導致亂花錢
- **Build-Sell 循環漏洞**：Run 1 的 Agent 發現建塔+賣塔可以刷 +0.2 獎勵，完全不打敵人
- **同質化塔問題**：Agent 只用 Machine Gun + Cannon（DPS/成本最高），即使加了多樣性獎勵也沒用
- **不升級問題**：Agent 寧願蓋新塔也不升級舊塔，因為建塔有即時獎勵但升級效益延遲

#### 3. 過度訓練會退化

Run 20 在 17.6M 步達到 155.4 波高峰後，繼續訓練反而退化。Policy 變得過於自信但脆弱，entropy 持續下降，出現多次 policy collapse（波數跌到 80 以下）。

#### 4. 改獎勵後不能接續訓練

Run 18 從 Run 17（165.6 波）接續訓練，但修改了獎勵結構，結果 Value Function 嚴重失配，從 151.6 波一路退化。改獎勵後應該從頭訓練。

### Agent 仍未學會的行為

- 持續升級塔到最高等級
- 策略性地放置牆壁形成迷宮
- 後期的資源管理（有錢不花 vs 亂花）
- 針對不同波次調整防禦策略

## 使用方式

### 安裝

```bash
pip install -r requirements.txt
```

### 人類玩家模式

```bash
python3 main.py
```

### 訓練 Agent

```bash
# 從頭訓練
python3 rl/train.py train --run-name my_run --timesteps 10000000

# 從 checkpoint 接續訓練
python3 rl/train.py train --run-name my_run --timesteps 10000000 --resume runs/my_run/best_wave_model

# 調整參數
python3 rl/train.py train --run-name my_run --timesteps 10000000 \
    --ent-coef 0.02 --net-size 512 --batch-size 512
```

### 錄製人類示範

```bash
python3 rl/record_demo.py --output demos/my_demo.npz --seed 42
```

### Behavioral Cloning 預訓練

```bash
# 從頭 BC
python3 rl/pretrain.py --demos demos/ --epochs 50

# 在既有模型上 fine-tune
python3 rl/pretrain.py --demos demos/ --epochs 50 --resume runs/my_run/best_wave_model
```

### 回放模型

```bash
python3 rl/replay.py --model runs/my_run/best_wave_model --seed 42
```

## 技術細節

- **演算法**：MaskablePPO（Proximal Policy Optimization + Action Masking）
- **框架**：Stable-Baselines3 + sb3-contrib
- **網路架構**：MLP [512, 512]（Policy 和 Value 各一個）
- **並行環境**：10 個 SubprocVecEnv
- **學習率**：3e-4
- **Discount factor**：γ = 0.995
- **Entropy coefficient**：0.02
- **每步遊戲時間**：40 ticks（2 秒）
- **最大步數/局**：2,500 步
- **尋路**：BFS（比 A* 快 15 倍，支援牆壁動態改變路徑）

## License

本專案僅供學習與研究用途。WarZone Tower Defense Extended 的原始遊戲版權歸原作者所有。
