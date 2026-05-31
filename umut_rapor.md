# BrainBlock — Member A Teknik Dokümantasyon

## 1. Problem Tanımı

BrainBlock, 8×5 birimlik bir tahtaya tetromino parçaları yerleştirilen bir bulmaca oyunudur. Her bölüm başında 10 parçadan oluşan karıştırılmış bir kuyruk oluşturulur. Agent, kuyruğun başındaki parçayı her adımda uygun bir orientation ve konum seçerek tahtaya yerleştirmek zorundadır. Bölüm şu durumlarda sona erer:

- Tüm 10 parça başarıyla yerleştirilirse → **success**
- Geçersiz bir aksiyon seçilirse → **illegal_action**
- Sıradaki parça için tahtada hiç geçerli konum kalmamışsa → **dead_end**

Problem yalnızca tek bir çözüme sahip değildir; tahtayı tamamen dolduran birden fazla farklı yerleşim kombinasyonu mevcuttur. Bu çalışmanın temel motivasyonlarından biri, agentin bu çeşitliliği keşfedebilmesi ve tek bir çözüme kilitlenmeden farklı stratejiler üretebilmesidir.

---

## 2. Parça Yapısı

### 2.1 Parça Tipleri ve Envanter

Beş tetromino tipi kullanılmaktadır. Her tip 4 birimlik hücreyi kaplar. Envanterde her tipten 2 adet bulunur, toplamda 10 parça (10 × 4 = 40 hücre = tam tahta):

| Tip | Adet | Temel Form |
|-----|------|------------|
| I   | 2    | Yatay çizgi |
| O   | 2    | 2×2 kare   |
| L   | 2    | L şekli    |
| Z   | 2    | Z şekli    |
| T   | 2    | T şekli    |

### 2.2 Koordinat Sistemi

Tahta `board[y, x]` şeklinde indekslenir:
- `x ∈ {0, 1, ..., 7}` → sütun (sol → sağ)
- `y ∈ {0, 1, ..., 4}` → satır (alt → üst)

Her parça, bir **anchor** noktasına göre tanımlanan `(dx, dy)` offsetlerinden oluşur. Anchor + offsetler = parçanın kapladığı hücreler.

### 2.3 Orientasyonlar

Her parça D4 dönüşüm grubuna göre en fazla 8 orientasyon alabilir (4 rotasyon × 2 ayna):

```
D4 = { 0°, 90°, 180°, 270° } × { normal, ayna }
```

Geometrik olarak özdeş orientasyonlar tekrar sayılmaz. Her geçerli orientasyon benzersiz bir frozenset ile temsil edilir. Parça başına geçerli orientasyon sayıları:

| Parça | Geçerli Orient. Sayısı | Geçerli İndeksler | Açıklama |
|-------|------------------------|-------------------|----------|
| I     | 2  | [0, 1]           | Yatay / dikey |
| O     | 1  | [0]              | Simetrik, tek orientasyon |
| L     | 8  | [0,1,2,3,4,5,6,7]| L + J (ayna) tüm rotasyonlar |
| Z     | 4  | [0,1,4,5]        | Z + S (ayna); 2,3 rotasyon tekrarı |
| T     | 4  | [0,1,2,3]        | 4 rotasyon, ayna özdeş |

**Not:** L parçasının ayna görüntüsü J parçasını, Z'nin ayna görüntüsü S parçasını verir. Fiziksel olarak çevrilebilen parçalar oldukları için her ikisi de aynı L ve Z parçası sayılır ve 8 orientasyon slotunun tamamı kullanılır.

---

## 3. Aksiyon Uzayı

### 3.1 Aksiyon Tanımı

Her aksiyon üç bileşenden oluşur:

```
a = (orient, x, y)
orient ∈ {0, ..., 7}   — orientasyon indeksi
x      ∈ {0, ..., 7}   — anchor sütun
y      ∈ {0, ..., 4}   — anchor satır
```

Tüm kombinasyonlar düz bir tam sayıya kodlanır:

```
encode(orient, x, y) = orient × 40 + x × 5 + y
decode(a)            = (a // 40, (a % 40) // 5, a % 5)
```

Bu kodlama ile `|A| = 8 × 8 × 5 = 320` benzersiz aksiyon elde edilir. Encode/decode döngüsü 320 aksiyonun tamamı için kayıpsız doğrulanmıştır.

### 3.2 Geçerli vs Geçersiz Aksiyonlar

320 aksiyonun büyük çoğunluğu herhangi bir durumda geçersizdir:
- Parça tahta dışına taşıyorsa
- Dolu hücrelerin üstüne biniyorsa
- İlgili parça için o orientasyon indeksi `None` ise (tekrar/hariç tutulan orientasyon)

Boş tahtada her parça için geçerli aksiyon sayıları:

| Parça | Boş Tahtada Geçerli Aksiyon |
|-------|-----------------------------|
| I     | 41                          |
| O     | 28                          |
| L     | 180                         |
| Z     | 90                          |
| T     | 90                          |

---

## 4. Aksiyon Maskeleme

### 4.1 Neden Gerekli?

Maskeleme olmadan agent, episodlar boyunca çok yüksek oranda geçersiz aksiyon seçer. Bu durum episodları erken sonlandırır, agentin anlamlı bir öğrenme sinyali almasını engeller ve başarı oranını sıfırda tutar. Bu etkiyi doğrulamak amacıyla aynı konfigürasyonla maskeleme kapatılarak 2M timestep eğitim gerçekleştirilmiştir. Sonuçlar bölüm 4.4'te detaylandırılmaktadır: invalid action rate başlangıçta %100 iken 2M adım sonunda ancak %39'a gerileyebilmiş, başarı oranı ise hiçbir noktada %0'ın üzerine çıkamamıştır.

### 4.2 Nasıl Çalışır?

Her adımda `compute_action_mask(board, piece)` fonksiyonu çalıştırılır:

```python
mask = np.zeros(320, dtype=np.int8)   # başlangıçta hepsi yasadışı

for orient in VALID_ORIENTS[piece]:   # yalnızca parçanın geçerli orientasyonları
    cells = ORIENT_TABLE[piece][orient]
    for x in range(8):
        for y in range(5):
            if parça (orient, x, y)'da tamamen tahtada ve boş hücrelerdeyse:
                mask[encode(orient, x, y)] = 1

return mask  # 1 = geçerli, 0 = geçersiz
```

### 4.3 Network'e Entegrasyonu

Maske, actor head'in logit çıktısına uygulanır:

```python
logits = actor_head(features)              # (B, 320) ham skor
logits = logits + (action_mask - 1) * 1e9 # geçersiz → -1,000,000,000
probs  = softmax(logits)                   # geçersiz aksiyonların prob ≈ 0
```

**Formül nasıl çalışır:**

```
mask = 1 (yasal)   → logit + (1-1) × 1e9 = logit + 0          → değişmez
mask = 0 (yasadışı)→ logit + (0-1) × 1e9 = logit - 1,000,000,000
```

Somut örnek — 5 aksiyonlu basit durum:

```
logits = [-0.3,  +1.2,  +0.8,  -0.5,  +2.1]
mask   = [  1,     0,     1,     0,     1  ]
sonuç  = [-0.3,  -999999998.8,  +0.8,  -1000000000.5,  +2.1]

softmax sonrası:
  yasal aksiyonlar   → olasılık paylaşır
  yasadışı aksiyonlar → e^(-1e9) ≈ 5×10^(-434) ≈ 0
```

**Neden doğrudan `probs = 0` yapmadık?**

Olası ilk yaklaşım olasılıkları doğrudan sıfırlamaktır:

```python
probs = softmax(logits)
probs[mask == 0] = 0.0      # yasadışıları sıfırla
probs = probs / probs.sum() # yeniden normalize et
```

Bu yaklaşım iki kritik soruna yol açar:

**Sorun 1 — Hesaplama grafiği kırılır.** PyTorch backpropagation için tüm operasyonları bir hesaplama grafiğinde takip eder. `probs[mask==0] = 0.0` satırı bu grafiği keser; gradient yasadışı aksiyonlardan geçemez ve ağırlık güncellemeleri bozulur.

**Sorun 2 — log(0) = -∞ → NaN.** PPO güncellemesinde iki yerde log-olasılık hesaplanır:

```python
log_prob = log(probs[action])        # policy loss için
H = -Σ probs[i] × log(probs[i])     # entropy için
```

`probs[i] = 0.0` olduğunda `log(0) = -∞`, ve `0 × (-∞) = NaN`. Tüm loss NaN olur, eğitim durur.

**Neden -1e9 bu sorunları yaşatmaz?**

`e^(-1e9) = 5×10^(-434)` — teknik olarak sıfır değil, yalnızca astronomik küçük bir sayı. Bu sayının logaritması:

```
log(5×10^(-434)) ≈ -998   ← finite, NaN değil
```

Softmax tamamen differentiable kalır, gradient düzgün akar, NaN oluşmaz. Yasadışı aksiyonlar `≈ 10^(-434)` olasılıkla seçilebilir — bu olasılık, bir trilyonun trilyonunun... katı kez küçük olduğu için pratikte sıfırla eşdeğerdir.

Bu teknik, RL literatüründe standart kabul gören yaklaşımdır (AlphaStar, kombinatoryal optimizasyon çalışmaları, Transformer attention masking).

### 4.4 Masking Olmadan Deney

Aynı konfigürasyonla (PPO R2 MLP, ent=0.05, div=1.0) maskeleme kapatılarak 2M timestep eğitim yapılmıştır:

| Metrik | Masking VAR | Masking YOK |
|--------|-------------|-------------|
| Final success rate | **%94.4** | **%0.0** |
| Unique tilings | **106** | **0** |
| Ortalama ep. uzunluğu | 9.8 adım | 6.6 adım |
| Invalid action rate | **%0.0** | %30-100 |
| Episode sayısı (2M step) | 226K | 321K |

Maskeleme olmadan agent zamanla bazı geçersiz aksiyonları önlemeyi öğrenir (invalid rate: %100 → %35) ancak tüm bölüm boyunca hiçbir zaman tahtayı tamamlayamaz. Bu sonuç, action masking'in bu problem için zorunlu bir tasarım kararı olduğunu kanıtlamaktadır.

---

## 5. Environment Tasarımı

### 5.1 Gymnasium Uyumluluğu

Environment `gymnasium.Env` sınıfından türetilmiştir. `reset()` ve `step()` fonksiyonları standart Gymnasium arayüzüne uygundur.

### 5.2 Episode Başında Rastgele Karıştırma

Her episode'da envanterin sırası sıfırdan rastgele belirlenir:

```python
queue = INVENTORY.copy()       # [I, I, O, O, L, L, Z, Z, T, T]
self.np_random.shuffle(queue)  # rastgele karıştır
self._queue = queue
```

Bu tasarım iki kritik amaca hizmet eder:

1. **Genelleşebilirlik:** Agent belirli bir parça sırasını ezberleyemez, her durumda doğru hamle yapmayı öğrenmek zorundadır. Her episode bağımsız bir bulmaca örneğidir.

2. **Çeşitlilik:** Farklı sıralamalar farklı tiling stratejileri gerektirir. Aynı tiling için bile parçalar farklı sırada gelebilir, bu da 3.106 geometrik çözümün çok üzerinde farklı episode senaryosu oluşturur.

### 5.3 Gözlem Uzayı ve Kısmi Gözlenebilirlik

Her adımda agent iki bileşenden oluşan bir gözlem alır:

**`grid` — (1, 5, 8) float32:**
Tahtanın ikili doluluk haritası. `grid[0, y, x] = 1.0` o hücrenin dolu olduğunu gösterir. Hangi parçanın o hücreyi doldurduğu bilgisi tutulmaz — yalnızca dolu/boş ikili durum.

**`vec` — (10,) float32:**
İki kısımdan oluşur:

```
vec[:5]  = one-hot(current_piece)   — mevcut (sıradaki) parça
vec[5:]  = remaining_counts / 2.0  — kalan parça tipi sayıları
```

**Kritik tasarım kararı — Kısmi Gözlenebilirlik:**

Agent kalan parçaların **sayısını** görür ama **sırasını görmez**. Bu bilinçli bir tercih:

```
queue = [L, Z, I, Z, O, O, L, T, I, T]

vec[:5] = [0, 0, 1, 0, 0]                → mevcut: L (tam bilgi)
vec[5:] = [1.0, 1.0, 0.5, 1.0, 1.0]     → I:2, O:2, L:1, Z:2, T:2 kaldı
```

Agent "sonra Z, ondan sonra I geliyor" bilgisini **almaz**. Yalnızca "2 adet Z ve 2 adet I kalmış" bilgisini alır. Bu tasarım şu avantajları sağlar:

- **Daha zor ama daha gerçekçi:** Gelecekteki parça sırasını bilen bir agent, mevcut kararını o sıraya göre optimize eder. Sıra bilinmeden karar vermek, gerçek dünyadaki belirsizlik altında karar verme problemine daha yakındır.
- **Genelleşebilirlik:** Agent sıra bağımsız bir strateji geliştirmek zorunda kalır.
- **Kalan sayılar** ise faydalıdır: "2 adet I parçası kaldı" bilgisi, tahtanın hangi bölgelerini o parça için rezerve etmek gerektiğine dair ipucu verir.

Kalan sayılar maksimum 2 olabileceğinden `/2` ile `[0, 1]` aralığına normalize edilir.

**Agent'ın görebildikleri ve göremedikleri:**

| Bilgi | Agent görüyor mu? | Açıklama |
|---|---|---|
| Mevcut (sıradaki) parça | ✅ Tam | One-hot kodlama |
| Kalan parça sayıları | ✅ Var | Tip bazında sayım |
| Kalan parçaların sırası | ❌ Yok | Kısmi gözlenebilirlik |
| Tahtanın doluluk durumu | ✅ Tam | Binary grid |
| Hangi parçanın nerede olduğu | ❌ Yok | Yalnızca dolu/boş |

### 5.4 Bölüm Dinamiği

**`reset()`:**
- Tahta sıfırlanır (5×8 sıfır matris)
- Envanter `[I,I,O,O,L,L,Z,Z,T,T]` yeni bir seed ile karıştırılır
- İlk gözlem ve `action_mask` döndürülür

**`step(action)`:**
1. Aksiyon decode edilir → `(orient, x, y)`
2. Yerleşim geçerliliği kontrol edilir
3. Geçerliyse parça tahtaya yazılır, kuyruktan çıkarılır
4. Bitiş koşulu kontrol edilir:
   - Kuyruk bittiyse → `success`
   - Yeni parça için geçerli aksiyon yoksa → `dead_end`
   - Aksiyon geçersizse → `illegal_action` (masking ile pratikte gerçekleşmez)
5. Reward hesaplanır
6. `(obs, reward, terminated, truncated, info)` döndürülür

**`info` içeriği:**
- `action_mask`: (320,) int8 — bir sonraki adım için geçerli aksiyonlar
- `termination_reason`: `"success"` | `"dead_end"` | `"illegal_action"`
- `coverage`: dolu_hücre / 40
- `pieces_placed`: yerleştirilen parça sayısı

---

## 6. Ödül Fonksiyonları

### 6.1 R1 — Sparse Reward

Yalnızca tüm 10 parça başarıyla yerleştirildiğinde `+1`, diğer tüm durumlarda `0`:

```
r(s, a, s') = 1.0   eğer termination_reason == "success"
            = 0.0   diğer her durumda
```

Bu formül proje kılavuzundaki en sade tanıma karşılık gelir. Ancak sinyalin son adıma kadar tamamen yokluğu, agent'in ilerleme alıp almadığını öğrenmesini son derece güçleştirir.

### 6.2 R2 — Potential-Based Shaped Reward

Seyrek ödülün yarattığı öğrenme güçlüğünü aşmak amacıyla Ng, Harada ve Russell (1999) tarafından literatüre kazandırılan Potansiyel Tabanlı Ödül Şekillendirme yöntemi uygulanmıştır. Bu yaklaşımın teorik temeli şudur: şekillendirme ödülü `F(s, a, s') = γΦ(s') − Φ(s)` formunda tasarlandığında, orijinal problemin optimal politikası değişmeden korunur (*policy invariance*). Bu matematiksel güvence, şekillendirmenin yalnızca öğrenmeyi hızlandırdığını, doğru davranışı bozmadığını garanti eder. Ayrıca bu formun pozitif ödül döngülerine (*positive-reward cycles*) yol açmadığı da kanıtlanmıştır; agent, gerçekten ilerleme sağlamayan döngüsel hareketlerden ödül elde edemez.

Bu teorik çerçeveye dayanarak potansiyel fonksiyonu Φ(s) olarak tahtanın doluluk oranı seçilmiştir:

```
Φ(s) = dolu_hücre_sayısı / 40
```

Her adımda verilen reward:

```
r(s, a, s') = γ · Φ(s') − Φ(s)
```

Başarılı bölüm sonunda verilen reward:

```
r_final = 1.0 + γ · Φ(s') − Φ(s)
```

`γ = 0.99` discount faktörü kullanılır. Bu tasarım sayesinde agent her doğru yerleşimde anlık bir öğrenme sinyali alır; reward şekillendirmesinin teorik garantisi gereği optimal politika değişmez, yalnızca yakınsama dramatik biçimde hızlanır.

**Sayısal örnek** (4 dolu hücre → 8 dolu hücreye giden adım):
```
Φ(s)  = 4/40 = 0.10
Φ(s') = 8/40 = 0.20
r = 0.99 × 0.20 − 0.10 = +0.098
```

**R1 vs R2 karşılaştırması:**

| Özellik | R1 (Sparse) | R2 (Shaped) |
|---------|-------------|-------------|
| Adım başı sinyal | Yok (0) | Her adımda var |
| Öğrenme kolaylığı | Çok zor | Kolay |
| Optimal politika | Değişmiyor | Değişmiyor |
| PPO sonucu (2M step) | **%0 success** | **%99-100 success** |

---

## 7. Ağ Mimarileri

### 7.1 MLP Encoder

Tahta ve parça vektörü düzleştirilerek bir tam bağlantı ağına verilir:

```
Girdi: grid (1×5×8 = 40) + vec (10) → concat → 50 boyut
       Linear(50 → 256) → ReLU
       Linear(256 → 256) → ReLU
Çıktı: 256 boyutlu özellik vektörü
Parametre sayısı: ~78,848
```

### 7.2 CNN+MLP Encoder

Tahta önce konvolüsyon katmanlarından geçirilir, ardından vektörle birleştirilerek MLP'ye verilir:

```
Girdi: grid (1, 5, 8)
       Conv2d(1 → 32, kernel=3, pad=1) → ReLU   # (32, 5, 8)
       Conv2d(32 → 64, kernel=3, pad=1) → ReLU  # (64, 5, 8)
       Flatten → 2560 boyut
       Concat vec (10) → 2570 boyut
       Linear(2570 → 256) → ReLU
       Linear(256 → 256) → ReLU
Çıktı: 256 boyutlu özellik vektörü
Parametre sayısı: ~742,784
```

CNN+MLP, MLP'ye kıyasla 3.3× daha fazla parametreye sahiptir. Bu nedenle CNN+MLP eğitiminde öğrenme oranı `1e-4` (MLP için `3e-4`) kullanılmıştır.

### 7.3 ActorCritic Mimarisi

Her iki encoder da aynı ActorCritic wrapper'ı ile kullanılır:

```
Encoder(grid, vec) → features (256)
    ↓                    ↓
Actor Head           Critic Head
Linear(256→256)      Linear(256→256)
ReLU                 ReLU
Linear(256→320)      Linear(256→1)
+ masking            scalar
→ Categorical dist   → V(s)
```

**Toplam parametre sayıları:**
- MLP ActorCritic: 292,929
- CNN+MLP ActorCritic: 956,865

---

## 8. PPO Algoritması

Proximal Policy Optimization (PPO), on-policy bir aktör-eleştirmen algoritmasıdır. Her güncelleme döngüsünde sabit sayıda çevre adımı (`rollout_steps = 2048`) toplanır ve bu veri üzerinden birden fazla gradient güncellemesi yapılır.

### 8.1 Rollout Toplama

Her rollout döngüsünde:
1. Agent mevcut politikayla `rollout_steps` adım boyunca çevreyle etkileşir
2. Her adımda `(obs, action, reward, done, log_prob, value)` tampona yazılır
3. Bölüm içinde birden fazla episode bulunabilir

### 8.2 GAE — Genelleştirilmiş Avantaj Tahmini

Avantajlar, TD artığı `δ` ve üstel ağırlıklandırma ile hesaplanır:

```
δ_t = r_t + γ · V(s_{t+1}) · (1 - done_t) - V(s_t)

A_t = δ_t + (γ · λ) · A_{t+1}   (geriye doğru hesaplanır)

Returns_t = A_t + V(s_t)
```

`λ = 0.95`: Varyans-bias dengesi. `λ=1` → Monte Carlo (yüksek varyans), `λ=0` → TD(0) (yüksek bias).

### 8.3 PPO Güncelleme

Toplanan rollout verisi üzerinde `ppo_epochs = 4` kez ve `mini_batch_size = 64` ile mini-batch güncellemesi yapılır:

**Policy Loss (Clipped Surrogate):**
```
ratio = exp(log π_new(a|s) - log π_old(a|s))

L_policy = -min(
    ratio · A,
    clip(ratio, 1-ε, 1+ε) · A
)
```
`ε = 0.2`: Politika güncellemesinin büyüklüğünü sınırlar, kararsızlığı önler.

**Value Loss:**
```
L_value = 0.5 · (Returns - V(s))²
```

**Entropy Bonus:**
```
L_entropy = -H(π)   →   keşfi teşvik eder
```

**Toplam Loss:**
```
L = L_policy + 0.5 · L_value + c_ent · L_entropy
```

`c_ent` (entropy coefficient): Keşif miktarını kontrol eden kritik hiperparametdir. Düşük değer (0.01) → hızlı convergence, az keşif. Yüksek değer (0.05) → yavaş convergence, geniş keşif.

**Gradient clipping:** `max_grad_norm = 0.5` ile eğitim stabilitesi sağlanır.

### 8.4 Hiperparametreler

| Parametre | Değer |
|-----------|-------|
| Öğrenme oranı (MLP) | 3e-4 |
| Öğrenme oranı (CNN+MLP) | 1e-4 |
| Discount faktörü γ | 0.99 |
| GAE lambda λ | 0.95 |
| Clip epsilon ε | 0.2 |
| Value loss katsayısı | 0.5 |
| Gradient clip | 0.5 |
| Rollout steps | 2048 |
| Mini-batch size | 64 |
| PPO epochs | 4 |
| Toplam timestep | 2,000,000 |

---

## 9. Diversity Bonus

Agent'in eğitim sırasında yeni tilingler keşfetmesini teşvik etmek için opsiyonel bir çeşitlilik bonusu eklenmiştir. Eğitim boyunca bulunan tüm benzersiz tilingler bir frozenset yapısıyla takip edilir:

```python
tiling_key = frozenset(
    (piece_type, frozenset(cells)) for piece_type, cells in env._placed
)
```

Eğer bu tiling daha önce görülmemişse, terminal reward'a `diversity_bonus` miktarı eklenir. Bu mekanizma yalnızca `--diversity-bonus > 0` argümanıyla aktif olur.

### 9.1 Entropy Bonus ile Farkı

Entropy bonus ve diversity bonus birbirini tamamlayan ama farklı seviyelerde çalışan iki mekanizmadır:

**Entropy Bonus — adım seviyesi:**
Her tek adımda politikanın olasılık dağılımının yayvan kalmasını zorunlu kılar. "Bu durumda her zaman aynı aksiyonu seçme" der. Ancak her adımda stokastik davranmak, her bölümün farklı bir çözümle bitmesini **garanti etmez** — yüksek avantajlı yollar yine de baskın çıkar.

**Diversity Bonus — tiling seviyesi:**
Yalnızca 10 adımlık bölüm tamamlandığında ve o tiling yeni ise tetiklenir. "Bu çözüm yolunu daha önce buldun, başka bir yol ara" der. Tek başına yetersizdir çünkü düşük entropi ile policy çok hızlı deterministikleşir ve bonus hiç tetiklenemez.

**Deneysel kanıt (2M step, seed=42, tüm deneyler):**

| ent_coef | div | Training UniqueT | Eval Unique Solutions | Eval Success |
|---|---|---|---|---|
| 0.01 | 0 | 13 | 3 | %97.1 |
| 0.01 | 1.0 | 13 | 3 | %98.5 |
| 0.05 | 0 | 65 | 4 | %97.6 |
| **0.05** | **1.0** | **106** | **7** | **%94.4** |

`ent=0.01, div=1.0` satırı şunu göstermektedir: entropy düşük olunca policy 200K adımda deterministikleşir, diversity bonus hiç tetiklenemez — 13 tiling baseline ile aynı kalır.

`ent=0.05, div=0` satırı ise entropy'nin tek başına büyük bir sıçrama sağladığını göstermektedir (13 → 65 tiling). Diversity bonus üstüne eklenerek bu sayıyı 65 → 106'ya çıkarmaktadır (%63 ek kazanım). Kritik nokta: agentin eval sırasında birden fazla farklı çözüm üretebilmesi yalnızca diversity bonus ile mümkün olmuştur (div=0 → 4 farklı çözüm, div=1.0 → 7 farklı çözüm).

Entropy bonus çeşitliliğin **koşulunu** yaratır; diversity bonus **yönünü** verir. İkisi birbirini tamamlar.

---

## 10. Deney Sonuçları ve Model Seçimi

Tüm deneyler seed=42 ile, 2M timestep boyunca MPS (Apple Silicon) üzerinde çalıştırılmıştır. Her deney ~2 saatte tamamlanmıştır.

### 10.1 Deney 1: Reward Fonksiyonu Karşılaştırması (R1 vs R2)

İlk deneyimiz, reward fonksiyonu tasarımının öğrenme üzerindeki etkisini anlamak amacıyla yapılmıştır.

**R1 — Sparse Reward** (`ppo_r1_mlp_seed42`):

2M timestep boyunca success rate **%0** olarak kaldı. Entropy ~2.0 nats'ta saplanarak hiç düşmedi — policy hiçbir zaman ödül sinyali alamadı. 286K episode'da yalnızca 10 unique tiling tespit edildi; bunlar tamamen rastgele keşiflerdir.

```
step=409K:   succ=0.000, ent=1.70
step=819K:   succ=0.000, ent=1.88   ← entropy düşmüyor
step=1.6M:   succ=0.000, ent=1.79   ← öğrenme yok
```

**R2 — Potential-Based Shaped Reward** (`ppo_r2_mlp_seed42`):

Aynı mimari ile R2 reward kullanıldığında sadece 200K adımda **%98 success rate** elde edildi. Her adımda doluluk oranından hesaplanan sinyal, agentin ilerlediğini anlamasını sağladı.

```
step=204K:   succ=0.980, ent=0.11   ← çok hızlı convergence
step=614K:   succ=0.990, ent=0.23
step=1.98M:  succ=0.990, ent=0.16
```

**Sonuç:** R1 bu problemde tamamen öğretilemezdir. R2 hem hızlı hem kararlı convergence sağlar. Potential-based shaping, optimal politikayı değiştirmeden yoğun öğrenme sinyali üretir.

---

### 10.2 Deney 2: Mimari Karşılaştırması (MLP vs CNN+MLP)

R2 reward ile iki encoder mimarisi, hem düşük entropy (ent=0.01) hem de en iyi konfigürasyon (ent=0.05, div=1.0) altında karşılaştırılmıştır.

**Baseline konfigürasyonu (ent=0.01, div=0):**

| Metrik | MLP (292K param, lr=3e-4) | CNN+MLP (956K param, lr=1e-4) |
|--------|--------------------------|-------------------------------|
| Eval success rate | %97.1 | %96.7 |
| Training UniqueT | 13 | 16 |
| Eval unique solutions | 3 | 3 |
| Convergence adımı | ~200K | ~600K |
| Eğitim süresi (2M step) | ~1.7 saat | ~2.9 saat |

**En iyi konfigürasyonla (ent=0.05, div=1.0):**

| Metrik | MLP | CNN+MLP |
|--------|-----|---------|
| Eval success rate | %94.4 | **%99.0** |
| Training UniqueT | **106** | 31 |
| Eval unique solutions | **7** | 3 |
| Eğitim süresi | ~1.7 saat | ~4.5 saat |

CNN+MLP yüksek entropy konfigürasyonunda daha yüksek başarı oranına (%99) ulaşmış ancak keşif açısından MLP'nin çok gerisinde kalmıştır: yalnızca 31 benzersiz tiling bulmuş ve eval'da 3 farklı çözüm üretmiştir. Bunun muhtemel nedeni, CNN+MLP'nin daha düşük öğrenme oranı (lr=1e-4) ile birleşince keşif fazını daha erken terk etmesi ve daha az sayıda ancak daha "baskın" çözüm yoluna yerleşmesidir. Başarı oranı yüksek görünse de policy daha hızlı deterministikleşmektedir. Ayrıca CNN+MLP, lr=3e-4 ile hiç converge etmemiş; uygun öğrenme oranı ancak deneme yanılma yoluyla bulunabilmiştir. 8×5 gibi küçük bir tahtada konvolüsyonel özelliklerin sağladığı avantaj da kısıtlıdır.

**Sonuç:** MLP encoder bu problem için hem hesaplama maliyeti hem de çeşitlilik açısından üstündür. CNN+MLP ekstra parametre ve eğitim süresi karşılığında daha az çeşitlilik ve daha hassas hiperparametre bağımlılığı getirmektedir.

---

### 10.3 Deney 3: Çeşitlilik Analizi

R2+MLP baseline ile başarı oranı yüksekti (%99), ancak eval sırasında agent yalnızca 2–3 benzersiz çözüm üretiyordu.

**Gözlem:** Baseline konfigürasyonda (ent=0.01) her iki mimari de training sırasında yalnızca 13–16 unique tiling buldu. Policy convergence sonrası aynı 2–3 yolda kilitleniyor.

**Deneme 1 — Yalnızca Diversity Bonus** (`ppo_r2_mlp_div1_seed42`, ent=0.01, div=1.0):

Diversity bonus eklenmesine rağmen training UniqueT yine **13** kaldı. Sebebi: ent=0.01 ile policy 200K adımda deterministikleşiyor, bonus hiç tetiklenemiyor.

**Deneme 2 — Yalnızca Entropy Artışı** (`ppo_r2_mlp_ent005_nodiv_seed42`, ent=0.05, div=0):

Diversity bonus olmadan, yalnızca entropy katsayısı 5× artırılarak 2M step eğitim yapıldı:

```
step=307K:   succ=0.000, ent=1.64, UniqueT=19    ← keşif fazı
step=573K:   succ=0.000, ent=1.50, UniqueT=57    ← hızla artıyor
step=696K:   succ=0.910, ent=0.27, UniqueT=59    ← convergence patlıyor
step=1M:     succ=0.930, ent=0.47, UniqueT=60    ← tiling artışı durdu
step=1.98M:  succ=0.960, ent=0.58, UniqueT=65    ← stabil
```

Training UniqueT: **65**, Eval unique solutions: **4**. Entropy tek başına büyük bir sıçrama sağladı (13→65), ancak agentin birden fazla farklı çözüm üretebilmesi için yetersiz kaldı.

**Deneme 3 — Entropy + Diversity Bonus** (`ppo_r2_mlp_ent005_div1_seed42`, ent=0.05, div=1.0):

Entropy katsayısı 5× artırılarak ve diversity bonus=1.0 eklenerek 2M step eğitim yapıldı:

```
step=307K:   succ=0.000, ent=1.95, UniqueT=27    ← keşif fazı, daha geniş
step=614K:   succ=0.000, ent=1.66, UniqueT=73    ← hızla artıyor
step=819K:   succ=0.200, ent=1.23, UniqueT=94    ← convergence başlıyor
step=921K:   succ=0.830, ent=0.31, UniqueT=102   ← patlama noktası
step=1.98M:  succ=0.930, ent=0.49, UniqueT=106   ← stabil
```

Training UniqueT: **106**, Eval unique solutions: **7**. Diversity bonus, keşif fazını daha agresif hale getirerek 65 → 106 tilingler elde edildi (+63%).

Eğitim iki aşamalı bir süreç sergiledi:
- **Keşif fazı (0–820K):** Entropy yüksek (~1.6–2.0), success düşük (~%0–3), tiling sayısı hızla artıyor (0→94)
- **Convergence fazı (820K–2M):** Entropy düşüyor (0.3–0.5), success yükseliyor (%83→%99), tiling artışı yavaşlıyor (94→106)

Tilinglerin **%89'u (94/106) keşif fazında**, policy henüz %5'ten az başarılıyken bulundu. Bu bulgu, diversity araştırmasının yüksek performanslı ancak deterministik bir policy tarafından değil, **keşif sırasındaki stokastik davranış** tarafından yürütüldüğünü göstermektedir.

---

### 10.4 Deney 4: Aksiyon Maskeleme Ablasyonu

En iyi konfigürasyon (ent=0.05, div=1.0) ile maskeleme kapatılarak 2M step eğitim yapıldı.

```
step=20K:   succ=0.000, Inv=1.000   ← tüm aksiyonlar yasadışı
step=102K:  succ=0.000, Inv=0.950   ← biraz öğreniyor
step=614K:  succ=0.000, Inv=0.890   ← hâlâ %89 geçersiz
step=1.98M: succ=0.000, Inv=0.390   ← %39 geçersiz, ama hiç başarı yok
```

Agent kısmen öğreniyor — invalid rate %100'den %39'a iniyor, coverage %42'ye ulaşıyor — ancak 2M adım boyunca **tek bir kez bile** tahtayı tamamlayamıyor. Unique tiling = 0.

| Metrik | Masking VAR | Masking YOK |
|--------|-------------|-------------|
| Final success rate | **%94.4** | **%0.0** |
| Training UniqueT | **106** | **0** |
| Ortalama ep. uzunluğu | 9.8 adım | 6.6 adım |
| Invalid action rate | **%0.0** | %39 |

**Sonuç:** Action masking bu problem için zorunludur. 320 aksiyon içinde anlamsız kombinasyonları denemek, agent'in gerçek öğrenme sinyaline ulaşmasını engeller.

---

### 10.5 En İyi Model Seçimi

Tüm deneylerin özet tablosu:

| Model | Reward | Encoder | ent | div | Mask | Train UniqueT | Eval Succ | Eval Uniq |
|-------|--------|---------|-----|-----|------|---------------|-----------|-----------|
| ppo_r1_mlp | R1 | MLP | 0.01 | 0 | ✓ | 10 | %0.0 | 0 |
| ppo_r2_mlp | R2 | MLP | 0.01 | 0 | ✓ | 13 | %97.1 | 3 |
| ppo_r2_mlp_div1 | R2 | MLP | 0.01 | 1.0 | ✓ | 13 | %98.5 | 3 |
| ppo_r2_cnn | R2 | CNN+MLP | 0.01 | 0 | ✓ | 16 | %96.7 | 3 |
| ppo_r2_mlp_ent005_nodiv | R2 | MLP | 0.05 | 0 | ✓ | 65 | %97.6 | 4 |
| ppo_r2_cnn_ent005_div1 | R2 | CNN+MLP | 0.05 | 1.0 | ✓ | 31 | %99.0 | 3 |
| **ppo_r2_mlp_ent005_div1** | **R2** | **MLP** | **0.05** | **1.0** | **✓** | **106** | **%94.4** | **7** |
| ppo_r2_mlp_ent005_div1_nomask | R2 | MLP | 0.05 | 1.0 | ✗ | 0 | %0.0 | 0 |

**Model seçimi:** Bu çalışmanın temel amacı yalnızca tahtayı dolduran bir agent geliştirmek değil, tek bir çözüme collapse etmeden farklı stratejiler üretebilen bir policy elde etmektir. Yalnızca başarı oranı optimize edildiğinde agent deterministik hale gelir ve her seferinde aynı 2–3 çözümü tekrarlar. Bu nedenle model seçiminde başarı oranı ile çeşitlilik birlikte değerlendirilmiştir. `ppo_r2_mlp_ent005_div1` konfigürasyonu, eğitim sırasında 106 benzersiz tiling keşfederek diğer modellerin 8 katı çeşitliliğe ulaşmış, eval'da 7 farklı çözüm üretmiş ve %94.4 başarı oranı ile yüksek performansını korumuştur. Invalid action rate %0.0 olarak gerçekleşmiş olup bu, action masking'in etkin biçimde çalıştığını doğrulamaktadır. Bu denge, söz konusu konfigürasyonun nihai model olarak seçilmesinin temel gerekçesidir.

---

## 11. İstatistiksel Doğrulama — 5 Farklı Seed

Elde edilen sonuçların şans eseri mi yoksa konfigürasyonun gerçek bir başarısı mı olduğunu anlamak için en iyi konfigürasyon (PPO R2 MLP, ent=0.05, div=1.0) 5 farklı random seed ile bağımsız olarak eğitilmiştir. Her seed farklı ağırlık başlangıcı, farklı episode dizisi ve farklı örnekleme paterni ürettiğinden her biri bağımsız bir deney niteliği taşır. Bu sayede hem sonuçların tekrarlanabilirliği hem de istatistiksel güvenilirliği ölçülmüştür.

### 11.1 Seed Seçimi

Seed değeri üç şeyi etkiler: (1) network ağırlıklarının başlangıç değerleri, (2) episode başına parça sırasının belirleneceği RNG zincirinin başlangıç noktası, (3) mini-batch karıştırma ve stochastic örnekleme. Her seed bağımsız bir eğitim çalışması anlamına gelir. Kullanılan seed'ler: **42, 55, 27, 17, 2710**.

### 11.2 Sonuçlar

| Seed | Train UniqueT | Eval Success | Eval Unique Solutions |
|------|:------------:|:------------:|:--------------------:|
| 42   | 106          | %94.4        | 7                    |
| 55   | 73           | %94.1        | 6                    |
| 27   | 111          | %90.3        | **17**               |
| 17   | 25           | %84.6        | 1                    |
| 2710 | 38           | %97.1        | 6                    |
| **Ortalama** | **70.6 ± 34.7** | **%92.1 ± 4.3** | **7.4 ± 5.2** |

### 11.3 Analiz

**Başarı oranı tutarlı:** 5 seedin tamamı %84–97 arasında converge etti. 0.921 ± 0.043 standart sapması oldukça düşük — konfigürasyonun farklı başlangıç koşullarına karşı sağlam olduğunu gösteriyor.

**Diversity yüksek varyans gösteriyor:** Eval unique solutions 1–17 arasında değişiyor (std=5.2). Bu varyansın kaynağı iki farklı davranış:

- **Seed=27 (en iyi):** Train UniqueT=111, eval'da 17 farklı çözüm. Policy keşif fazında geniş bir tiling havuzu oluşturmuş, convergence sonrasında stochastic örnekleme bu çeşitliliği koruyabilmiş.

- **Seed=17 (en düşük):** Train UniqueT=25, eval'da 1 çözüm. Convergence başarılı (%84.6) ama keşif fazı kısa ve dar olmuş — policy erken bir moda kilitlenmiş.

Bu varyans, çeşitliliğin başarı oranına kıyasla başlangıç koşullarına çok daha duyarlı olduğunu ortaya koymaktadır. Tüm seed'ler "iyi bir policy" öğreniyor, ama keşfedilen tiling havuzunun genişliği init'e bağlı.

**Demo modeli:** Seed=27 hem en yüksek eval unique solutions (17) hem de tatmin edici success rate (%90.3) ile demo için seçilen modeldir.

### 11.4 Özet Değerlendirme

| Metrik | Sonuç |
|--------|-------|
| Eval success rate | **%92.1 ± 4.3** |
| Mean ± std episodic return | **1.832 ± 0.063** |
| Mean episode length | **9.72 ± 0.12** |
| Invalid-action rate | **%0.0** |
| Unique solutions (ort.) | **7.4 ± 5.2** |
| Unique solutions (max) | **17** |

---

## 12. Sonuç

Bu çalışmada BrainBlock 8×5 tetromino bulmacası, sıfırdan geliştirilen bir PPO pipeline ile çözülmüştür. Çalışma boyunca yapılan deneyler, bu problem için üç tasarım kararının belirleyici olduğunu ortaya koymuştur: reward fonksiyonu seçimi, action masking ve çeşitlilik mekanizması. Sparse reward (R1) 2M adım boyunca hiçbir öğrenme üretemezken, potential-based shaped reward (R2) 200K adımda %98 başarı oranına ulaşmıştır. Maskelemenin kapatıldığı ablasyon deneyinde ise aynı konfigürasyon 2M adım sonunda hâlâ %0 başarı göstermiştir; bu sonuç, action masking'in bu problem için vazgeçilmez bir bileşen olduğunu deneysel olarak kanıtlamaktadır.

Çeşitlilik analizi, başarı oranı ile policy çeşitliliği arasında doğrudan bir tradeoff bulunduğunu göstermiştir. Düşük entropy katsayısı (0.01) ile eğitilen modeller %99 başarı oranına ulaşırken eval sırasında 2–3 çözümde kilitlenmektedir. Entropy katsayısının 0.05'e yükseltilmesi ve diversity bonus eklenmesiyle bu durum köklü biçimde değişmiş; eğitim sırasında 106 benzersiz tiling keşfedilmiş ve eval'da 7 farklı çözüm üretilmiştir. 5 farklı seed ile gerçekleştirilen istatistiksel doğrulama, %92.1 ± 4.3 başarı oranı ve ortalama 7.4 farklı çözüm ile sonuçların tutarlı ve tekrarlanabilir olduğunu teyit etmiştir. Final demo modeli olarak seed=27 konfigürasyonu seçilmiştir: eval'da 17 farklı çözüm, %90.3 başarı oranı ve %0 geçersiz aksiyon.
