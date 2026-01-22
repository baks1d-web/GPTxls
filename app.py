import os
import re
import json
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import streamlit as st
import pyarrow.feather as feather
from pathlib import Path
import hashlib

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Модель — можно поменять на gpt-4o-mini (дешевле) или gpt-4o
MODEL = "gpt-4o-mini"

# Папка для кэша
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)


def get_cache_path(files) -> Path:
    """Создаём уникальный ключ на основе имён и размеров загруженных файлов"""
    hasher = hashlib.md5()
    for file in files:
        hasher.update(file.name.encode())
        hasher.update(str(file.size).encode())
    hash_key = hasher.hexdigest()
    return CACHE_DIR / f"processed_{hash_key}.feather"


def extract_info_with_ai(text: str) -> dict:
    """Извлечение цены, типа (оригинал/аналог) и описания через OpenAI"""
    if not text or pd.isna(text):
        return {"price": None, "is_original": False, "description": ""}

    prompt = f"""
Ты эксперт по прайс-листам автозапчастей.
Из текста извлеки:
- цену в рублях — только число (1450.00 или 3200), игнорируй "руб", "с НДС", скидки и т.п.
- является ли запчасть оригинальной — true/false (ищи "оригинал", "OEM", "original", "genuine", "заводской" и подобные слова)
- краткое описание/наименование запчасти (если удаётся понять)

Верни ТОЛЬКО валидный JSON:
{{
  "price": число или null,
  "is_original": true/false,
  "description": "строка или пустая"
}}

Текст:
{text}
"""
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=120,
        )
        answer = response.choices[0].message.content.strip()

        # пытаемся распарсить как JSON
        data = json.loads(answer)

        price = data.get("price")
        if isinstance(price, str):
            cleaned = re.sub(r'[^\d.,]', '', price).replace(',', '.')
            price = float(cleaned) if cleaned.strip() else None

        return {
            "price": price,
            "is_original": bool(data.get("is_original", False)),
            "description": str(data.get("description", "")).strip()
        }

    except Exception as e:
        st.error(f"Ошибка при вызове OpenAI: {e}")
        return {"price": None, "is_original": False, "description": ""}


def process_files(uploaded_files):
    all_rows = []

    for uploaded_file in uploaded_files:
        try:
            df = pd.read_excel(uploaded_file)
        except Exception:
            try:
                df = pd.read_csv(uploaded_file)
            except Exception as e:
                st.error(f"Не удалось прочитать файл {uploaded_file.name}: {e}")
                continue

        df = df.astype(str).fillna("")

        # Пытаемся найти ключевые столбцы
        artic_col = next((c for c in df.columns if any(k in str(c).lower() for k in ["артикул", "арт", "code", "part", "номер"])), df.columns[0])
        price_col = next((c for c in df.columns if any(k in str(c).lower() for k in ["цена", "price", "стоим", "cost"])), None)

        for _, row in df.iterrows():
            # Собираем весь текст строки для анализа
            row_text = " | ".join(row.astype(str).values)
            info = extract_info_with_ai(row_text)

            artic = str(row.get(artic_col, "—")).strip()
            if not artic or artic == "—":
                continue

            price = info["price"]
            if price is None and price_col:
                try:
                    price_str = str(row.get(price_col, "")).strip()
                    cleaned = re.sub(r'[^\d.,]', '', price_str).replace(',', '.')
                    price = float(cleaned) if cleaned else None
                except:
                    pass

            if price is not None and price > 0:
                all_rows.append({
                    "Артикул": artic,
                    "Описание": info["description"] or "—",
                    "Оригинал": info["is_original"],
                    "Цена запчасти": price,
                    "Стоимость услуги": round(price * 0.25 + 1000, 2),
                    "Источник": uploaded_file.name
                })

    if not all_rows:
        return pd.DataFrame()

    df_extracted = pd.DataFrame(all_rows)

    # Группируем по артикулу → оригинал или самая дорогая
    def select_best(group):
        originals = group[group["Оригинал"] == True]
        if not originals.empty:
            return originals.sort_values("Цена запчасти", ascending=False).iloc[0]
        else:
            return group.loc[group["Цена запчасти"].idxmax()]

    df_best = df_extracted.groupby("Артикул", as_index=False).apply(select_best).reset_index(drop=True)

    return df_best


def main():
    st.set_page_config(page_title="Калькулятор цен на ремонт", layout="wide")
    st.title("Калькулятор цен на ремонт в сервисном центре")
    st.markdown("Загружайте прайс-листы поставщиков (xlsx/csv). Программа выберет оригинал или самую дорогую запчасть и посчитает стоимость услуги.")

    uploaded_files = st.file_uploader(
        "Выберите один или несколько файлов",
        type=["xlsx", "xls", "csv"],
        accept_multiple_files=True
    )

    if not uploaded_files:
        st.info("Загрузите файлы с прайсами поставщиков")
        return

    cache_path = get_cache_path(uploaded_files)

    if cache_path.exists():
        with st.spinner("Загружаем из кэша..."):
            df = feather.read_feather(cache_path)
        st.success(f"Данные загружены из кэша ({len(df)} позиций)")
    else:
        with st.spinner("Первичная обработка файлов (это будет один раз)..."):
            df = process_files(uploaded_files)
            if df.empty:
                st.error("Не удалось извлечь ни одной запчасти с ценой")
                return

            # Сохраняем в feather
            df.to_feather(cache_path)
            st.success(f"Обработано и сохранено в кэш: {len(df)} уникальных артикулов")

    # Редактируемая таблица
    st.subheader("Запчасти и цены")

    edited_df = st.data_editor(
        df,
        column_config={
            "Артикул": st.column_config.TextColumn("Артикул", disabled=True),
            "Описание": st.column_config.TextColumn("Описание", disabled=True),
            "Оригинал": st.column_config.CheckboxColumn("Оригинал", disabled=True),
            "Цена запчасти": st.column_config.NumberColumn(
                "Цена запчасти", min_value=0.0, step=0.01, format="%.2f"
            ),
            "Стоимость услуги": st.column_config.NumberColumn(
                "Стоимость услуги", format="%.2f", disabled=True
            ),
            "Источник": st.column_config.TextColumn("Источник", disabled=True),
        },
        use_container_width=True,
        hide_index=False,
        num_rows="dynamic"
    )

    if st.button("Пересчитать стоимость услуг"):
        edited_df["Стоимость услуги"] = edited_df["Цена запчасти"].apply(
            lambda x: round(x * 0.25 + 1000, 2) if pd.notna(x) and x > 0 else None
        )
        st.session_state["edited_df"] = edited_df.copy()
        st.rerun()

    # Кнопка скачивания
    if "edited_df" in st.session_state:
        output_df = st.session_state["edited_df"]
    else:
        output_df = edited_df

    st.download_button(
        label="Скачать результат (Excel)",
        data=output_df.to_excel(index=False, engine="openpyxl"),
        file_name="цены_на_ремонт.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


if __name__ == "__main__":
    main()