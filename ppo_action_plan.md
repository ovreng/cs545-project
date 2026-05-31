# BrainBlock DRL Projesi — Çalışma Planı ve Ortak Spesifikasyonlar

> Bu döküman iki amaca hizmet eder:
> 1. **Benim (PPO tarafı) detaylı iş planım.**
> 2. **Arkadaşımın (DQN tarafı) hizalanması gereken teknik spesifikasyonlar ve gerekçeleri.**
>
> Arkadaşım bu dökümanı okuyup §2'deki ortak spesifikasyonları onaylamalı veya alternatif önermelidir. §3 ve §4 yalnızca benim tarafımı kapsar.

---

## 1. İş bölümü özeti

| | Ben (Member A) | Arkadaş (Member B) |
|---|---|---|
| Algoritma | PPO (sıfırdan, PyTorch) | DQN (sıfırdan, PyTorch) |
| Encoder deneyleri | MLP → CNN+MLP (ikisi de) | Kendi tercihi |
| Reward deneyleri | R1 sparse + R2 potential-based shaped | Kendi reward tasarımı |
| Ek deney | Maskeli vs maskesiz (failure mode) | — |
| Environment | Kendi pipeline'ım (§2'ye uyumlu) | Kendi pipeline'ı (§2'ye uyumlu) |

Her iki taraf **kendi tam pipeline'ını** kurar (env + agent + training + eval). Birbirimize bağımlılık yok. Sonuçlar Faz 4'te birleştirilir.

---

## 2. Kilitlenmiş teknik spesifikasyonlar (ORTAK — iki taraf da buna uyar)

Bu bölümdeki her madde iki pipeline'ın sonuçlarını kıyaslanabilir kılmak için **birebir aynı** olmalıdır.

### 2.1 Koordinat konvansiyonu ve aksiyon flatten

```
Board: numpy array, shape (H, W) = (5, 8)
Erişim: board[y, x]
  x ∈ {0..7}  → yatay (sütun, soldan sağa)
  y ∈ {0..4}  → dikey (satır, yukarıdan aşağıya)
Hücre değeri: 0 = boş, 1 = dolu

Aksiyon: a ∈ {0..319}
Kodlama:  a = orient * 40 + x * 5 + y
Çözme:    orient = a // 40
          r      = a  % 40
          x      = r // 5
          y      = r  % 5

orient ∈ {0..7}, x ∈ {0..7}, y ∈ {0..4}
```

Bu formül **değişmez**. İki env de aynı `a` için aynı `(orient, x, y)` üretmelidir.

### 2.2 Parça tanımları (referans şekiller)

Her parça `(dx, dy)` offset kümesi olarak tanımlı. Normalize: `min(dx) = 0`, `min(dy) = 0`.

```
I:  [(0,0), (1,0), (2,0), (3,0)]     # yatay çubuk
O:  [(0,0), (1,0), (0,1), (1,1)]     # 2×2 kare
L:  [(0,0), (1,0), (2,0), (2,1)]     # yatay bar + sağ ucunda yukarı
Z:  [(0,0), (1,0), (1,1), (2,1)]     # basamak şekli
T:  [(0,0), (1,0), (2,0), (1,1)]     # yatay bar + ortadan yukarı
```

**Anchor `(x, y)`:** parçanın lokal origin'i board'da `(x, y)`'ye taşınır. Parça `{(x+dx, y+dy) | (dx,dy) ∈ offsets}` hücrelerini kaplar.

**Legallik:** tüm `x+dx ∈ {0..7}`, tüm `y+dy ∈ {0..4}`, kaplanan hiçbir hücre dolu değil, ve orient indeksi o parça tipi için geçerli (redundant değilse).

### 2.3 Yönelim üretimi (D4 grubu)

Her referans şekle 8 dönüşüm (4 rotasyon × 2 yansıma) uygulanır:

```
D4 dönüşümleri — (dx, dy) → :
  0: ( dx,  dy)          identity
  1: (-dy,  dx)          90° CCW
  2: (-dx, -dy)          180°
  3: ( dy, -dx)          270° CCW
  4: (-dx,  dy)          yansıma (x ekseni)
  5: (-dy, -dx)          yansıma + 90°
  6: ( dx, -dy)          yansıma + 180°
  7: ( dy,  dx)          yansıma + 270°
```

Her dönüşüm sonrası `min(dx)=0, min(dy)=0` olacak şekilde normalize edilir. `frozenset` ile dedup yapılır.

Benzersiz yönelim sayıları:

| Parça | Benzersiz | Maskelenen indeks sayısı (8 − benzersiz) |
|---|---|---|
| O | 1 | 7 |
| I | 2 | 6 |
| Z | 2 | 6 |
| L | 4 | 4 |
| T | 4 | 4 |

Redundant indeksler **maskelenir** (illegal kabul edilir, action mask'te 0). Bu, policy'nin aynı yerleştirmenin kopyalarına olasılık dağıtmasını engeller.

### 2.4 Observation formatı

```python
observation = {
    "grid": np.array, shape (1, 5, 8), dtype float32,   # binary occupancy
    "vec":  np.array, shape (10,),     dtype float32,    # aşağıda detay
}

vec[0:5]  = current_piece_onehot   # {I, O, L, Z, T} sırasıyla
vec[5:10] = remaining_counts       # her tip için kalan adet / 2  (normalize)
```

Tip indeksleme sırası: `I=0, O=1, L=2, Z=3, T=4`. İki taraf da bu sırayı kullanır.

### 2.5 `remaining_counts` semantiği

**Current parça hariç.** Kuyrukta current'tan *sonra* gelecek parçalar arasındaki tip sayıları.

Somut örnek — episode başı, queue = `[T, I, O, L, Z, T, I, O, L, Z]`:
- Adım 0: current = T. remaining (current hariç 9 parça): `{I:2, O:2, L:2, Z:2, T:1}` → normalize → `[1.0, 1.0, 1.0, 1.0, 0.5]`
- Adım 1 (T yerleşti): current = I. remaining (8 parça): `{I:1, O:2, L:2, Z:2, T:1}` → `[0.5, 1.0, 1.0, 1.0, 0.5]`
- ...
- Adım 9 (son parça): current = Z. remaining (0 parça): `[0, 0, 0, 0, 0]`

**Neden current hariç:** `remaining[t]=0` ve `current≠t` → t tipi tamamen tükendi. `current=t` ve `remaining[t]=0` → bu son t. Sinyal ek hesap gerektirmeden doğrudan okunur.

### 2.6 Farklı çözüm tanımı

Başarılı bir episode'da board her zaman tamamen dolu. "Farklı çözüm" = farklı **döşeme partisyonu**: 40 hücrenin 10 tetromino bölgesine ayrışımı + her bölgenin tipi farklıysa çözümler farklıdır. Yerleştirme **sırası** tek başına farklılık oluşturmaz (aynı tiling'e farklı sırayla ulaşmak aynı çözüm sayılır).

### 2.7 Random seed'ler

Her major deney 5 seed ile koşulur: `[42, 123, 456, 789, 1024]`.

### 2.8 Zorunlu metrikler (her deney, her seed)

- Success rate (çözülen episode oranı)
- Episodic return: ortalama ± std
- Ortalama episode uzunluğu
- Invalid-action rate
- Learning curve figürleri: reward/episode, kaplanan alan/episode, episode uzunluğu/zaman, invalid-rate/zaman

---

## 3. Benim deney matrisim

### 3.1 Reward fonksiyonları (ikisi de env'de flag ile seçilir)

**R1 — Sparse / terminal**
```
Başarılı yerleştirme (terminal değil):  0
Tam çözüm (10/10 yerleşti):            +1
Illegal aksiyon veya dead-end:          0, terminated=True
```
Saf credit-assignment testi. 10 adımlık kısa ufukta sparse'ın çalışıp çalışmayacağı ampirik soru.

**R2 — Potential-based shaped (ana reward)**
```
r_t = r_base + (γ · Φ(s_{t+1}) − Φ(s_t))

r_base: R1 ile aynı (success +1, aksi 0)
Φ(s)  = dolu_hücre_sayısı / 40
γ     = 0.99

Her başarılı yerleştirmede Φ farkı = 4/40 = 0.1 → anlık yoğun sinyal.
```
**Teorik güç:** potential-based shaping optimal policy'yi değiştirmez (Ng et al. 1999). Raporda "neden bu reward" sorusuna teoriyle cevap verilebilir.

**Not (arkadaşa):** Arkadaşın planındaki R2'de `+5 complete row` bonusu var. BrainBlock satır temizleme oyunu değil, tam kaplama oyunu — satır bonusu ajanı satır doldurmaya yönlendirip başka yerlerde doldurulamaz boşluk bıraktırabilir (reward hacking). Potential-based shaping bu riski taşımıyor. Arkadaşım kendi reward'ını kullanabilir ama bu riski bilmeli. Raporda iki farklı shaping stratejisinin kıyası güçlü bir bölüm olur.

### 3.2 Encoder kıyası

| Config | Encoder | Girdi |
|---|---|---|
| MLP | flatten(grid) ∥ vec = 50-dim → 256 → 256 | Basit, hızlı debug |
| CNN+MLP | Conv(1→32, 3×3, pad=1) → Conv(32→64, 3×3, pad=1) → flatten → ∥ vec → 256 | Uzamsal yapıyı korur |

İkisi de aynı actor (320 logit) + critic (1 skaler) head'i paylaşır. Masking: logitlere −1e9 eklenir.

### 3.3 Tam deney matrisi

| # | Encoder | Reward | Masking | Seed sayısı | Amaç |
|---|---|---|---|---|---|
| 1 | MLP | R1 (sparse) | Evet | 5 | Sparse baseline |
| 2 | MLP | R2 (shaped) | Evet | 5 | Shaped baseline |
| 3 | CNN+MLP | R1 (sparse) | Evet | 5 | Encoder kıyası (sparse) |
| 4 | CNN+MLP | R2 (shaped) | Evet | 5 | Encoder kıyası (shaped) |
| 5 | MLP | R2 (shaped) | Hayır | 5 | Failure mode analizi |

Toplam: 25 koşu. Config 5 maskesiz: failure-mode kanıtı (invalid-rate tavanı, öğrenme felci → "neden masking gerekli" argümanı).

---

## 4. Görev planı

### Faz 1 — Environment + görselleştirme (Gün 1–3)

**Gün 1:**
- `common/pieces.py`: §2.2 ve §2.3'teki parça tanımları + D4 yönelim üretimi + dedup. Bunu bitirip arkadaşa göndermek — iki taraf da aynı dosyadan import edecek.
- `BrainBlockEnv` iskeleti: `__init__`, `reset()`, board + queue mekaniği.

**Gün 2:**
- `step()`: action decode → legallik → board güncelleme → reward (R1/R2 flag) → dead-end tespiti → obs + info.
- Action mask hesaplama (320 aksiyon brute-force kontrol; board küçük, hızlı).
- Observation dict üretimi (§2.4 formatında).

**Gün 3:**
- `render()`: matplotlib renkli grid (tip başına renk).
- Episode replay fonksiyonu (adım adım board gösterimi).
- Unit testler:
  - (a) Bilinen çözümü elle oyna → 10 adımda success, board tamamen dolu.
  - (b) Kasıtlı illegal → terminated=True.
  - (c) Action mask brute-force doğrulama.
  - (d) Flatten/unflatten roundtrip (0..319).
  - (e) remaining_counts semantik testi (current hariç olduğunu doğrula).

### Faz 2 — PPO from scratch + training loop (Gün 4–7)

**Gün 4–5:**
- Actor-critic ağı (MLP encoder, §3.2).
- PPO core: rollout buffer, GAE(λ) hesaplama, clipped surrogate loss, entropy loss, value loss.
- Action masking entegrasyonu (logitlere −1e9).

**Gün 6–7:**
- Training loop: rollout toplama → PPO update → logging.
- Configurable hyperparameters (§5'teki tablo).
- Tensorboard / CSV logging (§2.8 metrikleri).
- Sanity check: R2 + maskeli PPO ile birkaç bin episode → en az birkaç parça yerleştirebiliyor mu? (Tam çözüm henüz beklenmez, sinyal akışı doğru mu kontrolü.)

### Faz 3 — Deneyler (Gün 8–10)

**Gün 8:**
- CNN+MLP encoder eklenmesi (§3.2), flag ile seçilebilir.
- Config 1–4 koşuları başlatılır (5 seed × 4 config = 20 koşu).

**Gün 9:**
- Config 5 (maskesiz) koşusu.
- Her config'den ≥5 farklı çözüm çıkarılır ve görselleştirilir.

**Gün 10:**
- Learning curve figürleri üretilir (reward, covered area, episode length, invalid-rate).
- Qualitative rollout analizi: bir başarılı + bir başarısız episode adım adım trace.

### Faz 4 — Rapor + sunum (Gün 11–14)

**Gün 11–12 (bireysel bölümler):**
- MDP formülasyonu (Faz 0 dökümanından).
- R1 vs R2 tanım + potential-based shaping teorik gerekçesi.
- PPO implementasyon detayları + hiperparametre tablosu.
- MLP vs CNN+MLP ablation sonuçları ve analizi.
- Maskeli vs maskesiz failure-mode tartışması.
- ≥5 farklı çözüm görselleri.

**Gün 13–14 (ortak):**
- PPO vs DQN karşılaştırma figürleri.
- İki reward stratejisinin kıyası (benim potential-based vs arkadaşın heuristic).
- Rapor finalize + sunum slaytları.
- Canlı demo scripti (pretrained model yükleme → board'da adım adım çözüm gösterimi).

---

## 5. Başlangıç hiperparametreleri

| Parametre | Değer |
|---|---|
| Learning rate | 3e-4 |
| Discount (γ) | 0.99 |
| GAE λ | 0.95 |
| PPO clip ε | 0.2 |
| Entropy katsayısı | 0.01 |
| Value loss katsayısı | 0.5 |
| Max gradient norm | 0.5 |
| Rollout steps | 2048 |
| Mini-batch size | 64 |
| PPO epochs/update | 4 |
| Hedef training episode | 500K+ |
| MLP hidden dim | 256 |
| CNN channels | [32, 64] |
| CNN kernel | 3×3, padding=1 |

---

## 6. Arkadaşa notlar

**Mutlaka hizalanması gerekenler (§2'nin tamamı):** Parça offset'leri, flatten formülü, observation dict formatı, remaining_counts semantiği (current hariç), tip indeks sırası (I=0 O=1 L=2 Z=3 T=4), farklı çözüm tanımı, seed'ler. Bunlardan herhangi birinde uyumsuzluk olursa sonuçlar kıyaslanamaz.

**`common/pieces.py`:** Ben Gün 1'de yazıp paylaşacağım. İkimiz de buradan import ederiz — böylece parça geometrisi garantili aynı olur.

**Reward tasarımı hakkında öneri:** Planındaki R2'de `+5 complete row` bonusu var. BrainBlock satır temizleme değil tam kaplama oyunu. Satır bonusu, ajanı satır doldurmaya yönlendirip boşluk bıraktırma riskine sahip. Bunu korumak istiyorsan raporda bilinçli bir tercih olarak gerekçelendir; ya da potential-based shaping'e geçmeyi düşün. İki farklı shaping stratejisi kullanmamız rapora güçlü bir kıyaslama bölümü katar.

**DQN ile masking:** DQN'de masking PPO'dan farklı çalışır — Q-değerlerinde illegal aksiyonlara −∞ atayıp argmax'ta hariç tutmak gerekir. ε-greedy keşifte de sadece legal aksiyonlardan seçilmeli. Bunu implementasyonda gözetmeli.