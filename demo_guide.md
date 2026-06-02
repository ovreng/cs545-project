# BrainBlock Demo Kılavuzu

## Hazırlık

```bash
cd ~/Documents/cs545-project
source venv/bin/activate
```

---

## Klavye Kontrolleri (tüm senaryolarda aynı)

| Tuş | Ne Yapar |
|-----|----------|
| `A` | Auto-play aç / kapat |
| `SPACE` | Tek adım ilerle (auto-play kapalıyken) |
| `+` | Auto-play hızlandır |
| `-` | Auto-play yavaşlat |
| `R` | Yeni episode başlat |
| `Q` / `ESC` | Çıkış |

---

## Senaryo 1 — PPO Ajanı (Stochastic)

**Ne gösterir:** PPO öğrenilmiş bir politika dağılımı kullanır. Her `R` tuşunda farklı bir çözüm üretebilir.

```bash
python -m member_umut.demo \
  --model results/ppo_r2_mlp_ent005_div1_seed27/best_model.pt \
  --algo ppo \
  --stochastic \
  --speed 0.5
```

**Demo adımları:**
1. `A` ile auto-play aç, tahtanın dolduğunu izle
2. Birkaç kez `R` bas — renk dağılımının değiştiğini göster
3. "Aynı model, farklı çözümler" diyerek vurgula

---

## Senaryo 2 — DQN Ajanı (Farklı Parça Sıraları)

**Ne gösterir:** DQN de 14 farklı çözüm bulabiliyor — ama bu çeşitlilik farklı parça sıralarından geliyor (Senaryo A). `--seed` verilmediğinde her `R` farklı parça sırası getirir, DQN her sıra için aynı deterministik cevabı üretir.

```bash
python -m member_umut.demo \
  --model results/dqn_r2_mlp_seed456/best_model.pt \
  --algo dqn \
  --speed 0.5
```

**Demo adımları:**
1. `A` ile auto-play aç, tahtanın dolduğunu izle
2. Birkaç kez `R` bas — farklı parça sıraları geldiği için farklı çözümler çıkar
3. "DQN de farklı çözümler bulabiliyor — ama sebebi farklı" diyerek Senaryo 3'e geçişi hazırla

> **Not:** DQN'in deterministikliğini göstermek için `--seed 42` ekle — aynı parça sırası sabitlenir, `R` tuşuna kaç kez basılırsa basılsın tahta hiç değişmez. Senaryo 3'te bunu PPO ile kıyaslıyoruz.

---

## Senaryo 3 — Senaryo B Karşılaştırması (Ana Nokta)

**Ne gösterir:** Aynı parça sırası (`--seed 42`) verildiğinde DQN hep aynı çözümü, PPO stochastic ise farklı çözümler üretir.

**İki terminal aç, yan yana diz:**

```bash
# Terminal 1 — DQN (deterministik)
python -m member_umut.demo \
  --model results/dqn_r2_mlp_seed456/best_model.pt \
  --algo dqn \
  --seed 42 \
  --speed 0.4
```

```bash
# Terminal 2 — PPO Stochastic
python -m member_umut.demo \
  --model results/ppo_r2_mlp_ent005_div1_seed27/best_model.pt \
  --algo ppo \
  --stochastic \
  --seed 42 \
  --speed 0.4
```

**Demo adımları:**
1. Her iki pencereyi yan yana aç
2. Her ikisinde `A` ile auto-play aç
3. İkisi de aynı parça sırasıyla başlıyor (`seed=42`)
4. DQN'de `R` bas — hep aynı tahta
5. PPO'da `R` bas — farklı tahta
6. "Aynı problem, iki farklı yaklaşım" mesajını ver

---

## Senaryo 4 — İnsan vs Agent

**Ne gösterir:** İnsan oyuncunun nasıl oynadığını, ardından ajanın aynı ortamda nasıl oynadığını karşılaştır.

```bash
# İnsan oyunu
python -m member_umut.play

# Ardından PPO ajanı
python -m member_umut.demo \
  --model results/ppo_r2_mlp_ent005_div1_seed27/best_model.pt \
  --algo ppo \
  --speed 0.6
```

**`play.py` kontrolleri:**
| Tuş | Ne Yapar |
|-----|----------|
| Mouse hover | Parça yerleştirme önizlemesi |
| Sol tık | Parça yerleştir |
| `←` `→` / Scroll | Orientasyon değiştir |
| `R` | Yeni oyun |
| `Q` / `ESC` | Çıkış |

---

## Senaryo 5 — Backtracking Solver (Referans)

**Ne gösterir:** RL olmadan kaba kuvvetle çözüm. Her zaman çözüm bulur ama öğrenilmiş bir şey yok — RL'in ne öğrendiğini anlamak için referans.

```bash
python -m member_umut.demo --solver --speed 0.3
```

---

## Hızlı Referans — Model Yolları

| Model | Açıklama | Yol |
|-------|----------|-----|
| PPO seed=27 | En çeşitli (17 unique çözüm) | `results/ppo_r2_mlp_ent005_div1_seed27/best_model.pt` |
| PPO seed=42 | Standart iyi model (%94.4 stoch) | `results/ppo_r2_mlp_ent005_div1_seed42/best_model.pt` |
| DQN seed=456 | Demo modeli (14 unique çözüm) | `results/dqn_r2_mlp_seed456/best_model.pt` |
| DQN seed=42 | Standart DQN (%99.9) | `results/dqn_r2_mlp_seed42/best_model.pt` |

---

## Hızlı Referans — `--speed` Değerleri

| Değer | Etki |
|-------|------|
| `0.2` | Çok hızlı (sadece göstermek için) |
| `0.5` | Hızlı ama takip edilebilir |
| `1.0` | Varsayılan |
| `2.0` | Yavaş, adım adım anlatmak için |

---

## Sık Yapılan Hatalar

**`ModuleNotFoundError: member_goktug`**
→ `python -m member_umut.demo` yerine doğrudan `python demo.py` çalıştırılmış. Her zaman `-m` ile çalıştır.

**Pencere açılmıyor / siyah ekran**
→ `venv` aktif değil. `source venv/bin/activate` çalıştır.

**`FileNotFoundError: best_model.pt`**
→ Proje kök dizininde olmayabilirsin. `cd ~/Documents/cs545-project` ile kök dizine dön.
