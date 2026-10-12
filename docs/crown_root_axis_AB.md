# 以 PCA 主軸求牙冠端點 A 與牙根端點 B

本文件說明 `src/segmentation/alveolar.py` 的 `tooth_axis` 如何由單顆牙齒的二元 mask 求出質心 `C′`、PCA 主軸 `L` 與兩個端點 `A`（牙冠側）、`B`（牙根側）。它是臨床冠根比（CRR）計算的第一步：

```
CRR = |AO| / |OB|        O = 左右齒槽骨頂連線 M 與主軸 L 的交點
```

`A`、`B` 決定了 `|AO|` 與 `|OB|` 的兩端，所以它們的誤差會直接進入 CRR。齒槽骨頂（`C`、`D`、`O`）如何求出不在本文件範圍，見 `src/segmentation/alveolar.py` 與 `notebooks/find_crown_root.ipynb`。

## 1. 輸入與輸出

| | 內容 |
|---|---|
| 輸入 | 目標牙的二元 mask（`H×W`，值 0／1）；單位為像素，座標 `x` 向右、`y` 向下 |
| 輸出 `origin` | 質心 `C′` |
| 輸出 `v` | 單位向量，主軸方向，**由牙冠指向牙根** |
| 輸出 `u` | 垂直於 `v` 的單位向量，`u = (v_y, −v_x)`（與 `v` 構成不鏡像的座標系） |
| 輸出 `t_top`、`t_bot` | `A`、`B` 在軸上的有號位置（以 `C′` 為原點，沿 `v` 量） |
| 輸出 `length` | 軸長 `|AB| = t_bot − t_top` |
| 輸出 `elongation`、`width_ratio`、`warnings` | 軸的可靠度指標（見第 5 節） |

端點座標為 `A = C′ + t_top·v`、`B = C′ + t_bot·v`。因為 `t_top < 0 < t_bot`，`A` 在質心的牙冠側、`B` 在牙根側。

## 2. 演算法

與手寫的演算法（下稱「原稿」）的對應：

| 原稿 | 實作 |
|---|---|
| 1. 邊界點集合 `B = {x, y}` → 質心 `C′` | mask 所有像素座標的平均 |
| 2. 給定 `B`，PCA 第一主軸 `L` | 共變異矩陣最大特徵值的特徵向量 |
| 4. `A = max(Bᵢ C′ · L)`（`C′` 上方）、`B = max(Bᵢ C′ · L)`（`C′` 下方） | 每個像素對 `v` 的有號投影的最小值與最大值 |

### 步驟

1. **質心**：`C′ = mean(p)`，`p` 為 mask 內所有像素的 `(x, y)`。
2. **主軸**：`Σ = cov(p − C′)`（2×2），取最大特徵值的特徵向量為 `v`。同時記錄 `elongation = λ₁ / λ₂`。
3. **每個像素的軸向位置與側向位置**：
   `tᵢ = (pᵢ − C′) · v`、`sᵢ = (pᵢ − C′) · u`。
4. **端點位置**：`t_lo, t_hi = percentile(t, [q, 100 − q])`，`q = end_percentile`。
   - `q = 0`：`t_lo = min(t)`、`t_hi = max(t)`，也就是原稿所稱「與主軸投影最長」的點。
   - `q = 1`（`AlveolarConfig` 預設）：取 1%／99% 百分位，捨棄最極端的少量像素，以降低 mask 雜訊（毛刺、分割多出的小突起）的影響。
5. **判斷牙冠端與牙根端**（見第 3 節）。若 `t_lo` 端是牙根，翻轉 `v`（`v ← −v`、`t_lo, t_hi ← −t_hi, −t_lo`），使 `v` 永遠由牙冠指向牙根，之後 `t_top = t_lo`、`t_bot = t_hi`。
6. **端點**：`A = C′ + t_top·v`、`B = C′ + t_bot·v`。

`A`、`B` 是極端像素**投影在軸上的位置**，位於軸線上，並非輪廓上的原像素。這與原稿的投影長度一致：`|AO|`、`|OB|` 都沿著軸量。因為輪廓上的點才可能是投影極值，所以對全部 mask 像素取極值與只對邊界點取極值結果相同。

## 3. 牙冠端與牙根端怎麼分

PCA 的特徵向量正負號任意，必須另外決定哪一端是牙冠。實作用**形狀啟發式**：

- 取軸的兩端各 `end_segment_frac = 25%` 軸長的像素；
- 兩段各自量側向寬度 `w = P98(s) − P2(s)`（側向座標 2%～98% 百分位的跨度）；
- **較窄的一端判為牙根**。

`width_ratio = min(w) / max(w)`。`width_ratio > ambiguity (0.85)`，也就是兩端寬度太接近時，回報警告 `orientation_unclear`，不靜默使用。

## 4. 之後如何使用 `A`、`B`

- `axis_warp` 以 `v`、`u`、`C′` 與 `t_top` 建立仿射變換，把牙齒轉成「軸垂直、牙冠朝上」的座標系：`x′ = u·(p − C′) + 常數`（側向）、`y′ = v·(p − C′) − t_top + 邊界`（沿軸）。`A` 在 `y′ = margin`，`B` 在 `y′ = margin + 軸長`。
- `scan_profile` 在 `A` 到 `B` 之間每個高度做一條**垂直於軸的掃描線**，量軸到左右輪廓的距離，得到左右半寬度剖面（對應草圖中的綠色箭頭）。
- `O` 是兩側骨頂連線與軸的交點，`|AO| = y′_O − y′_A`、`|OB| = y′_B − y′_O`，`CRR = |AO| / |OB|`。

## 5. 可靠度檢查與已知限制

| 檢查 | 觸發 | 意義 |
|---|---|---|
| `axis_unreliable_low_elongation` | `λ₁ / λ₂ < 1.5` | mask 接近圓形，主軸方向不穩定 |
| `orientation_unclear` | `width_ratio > 0.85` | 兩端寬度太接近，牙冠／牙根判定不可靠 |
| `tooth_truncated_apex_or_crown_unreliable` | 呼叫端傳入 `truncated=True` | mask 在影像或 ROI 邊界被截斷，`A` 或 `B` 不是真實的牙冠頂或根尖 |

已知限制（尚未以真值驗證）：

1. **牙根較窄的假設會失效**：多根牙（臼齒）的牙根向外分叉，根端的 2%～98% 側向跨度可能不小於牙冠；此時方向會被判反，且只要 `width_ratio ≤ 0.85` 就**不會**有警告。若 `A`、`B` 看起來顛倒，應優先檢查這一點。可行的替代：以影像亮度（琺瑯質較亮）或已知的牙位／上下顎判定方向。
2. **端點對 mask 雜訊敏感**：`end_percentile = 0` 時，單一毛刺就會移動 `A` 或 `B` 並改變 CRR；`1%` 較穩，但會略微縮短軸長。兩者的差異應在有真值的資料上量測，不能假設其中一個較好。
3. **PCA 軸不等於牙齒解剖長軸**：彎曲牙根、傾斜或不對稱的牙冠會使 PCA 軸偏離牙根實際走向，進而影響 `|OB|`。
4. **只看 mask**：`A`、`B` 完全由 mask 形狀決定，mask 品質（含鄰牙黏連、修復體）直接傳遞；ROI 的 mask 來自人工多邊形標註，U-Net 預測 mask 會再增加誤差。
5. 垂直方向的長度未做毫米校正，只能以像素或軸長比例比較。

## 6. 驗證

`tests/segmentation/test_alveolar.py`：

- `test_axis_points_from_crown_to_root_whatever_the_rotation`：合成牙齒旋轉 0°／35°／140°／−70° 後，`v` 仍由寬端指向窄端，軸長與真值差 < 15 px（1%／99% 會略微截掉尖細的根尖）。
- `test_extreme_end_points_follow_max_projection_from_the_centroid`：`end_percentile = 0` 時，`t_top`、`t_bot` 等於全部像素投影的最小與最大值，質心等於像素平均。
- `test_scan_profile_measures_half_widths_perpendicular_to_the_axis`：掃描線量到的冠部半寬接近真值，每條掃描線的軸上垂足確實在主軸上。
- `test_flat_blob_gets_a_low_elongation_warning`：圓形 mask 觸發低拉長比警告。

目視檢查：`notebooks/find_crown_root.ipynb` 第 1、2 節畫出 `C′`、`L`、`A`、`B` 與掃描線。尚未與醫師標註的牙冠頂端或根尖比對。

## 7. 參數（`AlveolarConfig`）

| 參數 | 預設 | 作用 |
|---|---|---|
| `end_percentile` | 1.0 | `A`、`B` 取投影的 `q`%／(100−`q`)% 百分位；`0` ＝ 極值 |
| `end_segment_frac` | 0.25 | 判斷牙冠／牙根時，兩端各取軸長的比例 |
| `ambiguity` | 0.85 | 兩端寬度比超過此值就警告 |
| `margin_frac` | 0.10 | 軸座標系在 `A` 上方、`B` 下方保留的邊界（軸長比例），不影響 `A`、`B` 本身 |

筆記本 `find_crown_root.ipynb` 依原稿使用 `end_percentile = 0`；`find_cej.ipynb` 沿用預設的 1%。兩者的 CRR 不能直接比較，除非固定同一個端點定義。
