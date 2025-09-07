import os
import pandas as pd
import modules.shared as shared
import modules.processing as processing
import modules.sd_samplers
from modules import devices

# === Настройки ===
csv_path = "/content/drive/MyDrive/test 10 row.csv"
outdir = "/content/drive/MyDrive/generated"
os.makedirs(outdir, exist_ok=True)

# Загружаем промты
prompts = pd.read_csv(csv_path)
print("Загружено промтов:", len(prompts))

# Настройки генерации
steps = 20
width = 768
height = 1024
sampler = "Euler a"
seed = 42

# Генерация
for i, row in prompts.iterrows():
    prompt = str(row.iloc[0])  # первый столбец CSV
    print(f"▶ Генерация {i+1}/{len(prompts)}: {prompt[:60]}...")

    p = processing.StableDiffusionProcessingTxt2Img(
        sd_model=shared.sd_model,
        prompt=prompt,
        steps=steps,
        width=width,
        height=height,
        sampler_name=sampler,
        seed=seed+i
    )

    # Генерация
    processed = processing.process_images(p)

    # Сохраняем
    if processed.images:
        filename = os.path.join(outdir, f"gen_{i+1:03d}.png")
        processed.images[0].save(filename)
        print("💾 Сохранено:", filename)
    else:
        print("❌ Ошибка генерации")