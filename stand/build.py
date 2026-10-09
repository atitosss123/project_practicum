"""Сборка стенда: подстановка предрасчитанных данных в страницу.

Страница хранится отдельно от данных, чтобы её можно было читать и править.
Сборка вставляет JSON в тег script и копирует изображения, которые страница
показывает в разделе «Что посчитала программа».

Запуск (после scripts/build_stand_data.py):
    python stand/build.py
"""

import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PLOTS = ["dendrogram.png", "silhouette_profile.png", "mds_scatter.png", "pca_scatter.png"]
PLACEHOLDER = "__DATA__"


def main():
    template = (HERE / "index.html").read_text(encoding="utf-8")
    if PLACEHOLDER not in template:
        raise SystemExit(f"в index.html нет метки {PLACEHOLDER}")

    data = (HERE / "data.json").read_text(encoding="utf-8")
    json.loads(data)  # падаем сразу, если данные битые

    # Последовательность </ внутри строки закрыла бы тег script раньше времени.
    page = template.replace(PLACEHOLDER, data.replace("</", "<\\/"))
    destination = HERE / "index.built.html"
    destination.write_text(page, encoding="utf-8")

    missing = []
    for name in PLOTS:
        source = ROOT / "out" / name
        if source.exists():
            shutil.copy2(source, HERE / name)
        else:
            missing.append(name)

    print(f"{destination.relative_to(ROOT)}: {destination.stat().st_size / 1024:.0f} КБ")
    if missing:
        print(f"нет изображений (запустите main.py): {', '.join(missing)}")


if __name__ == "__main__":
    main()
