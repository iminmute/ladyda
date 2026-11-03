# -*- coding: utf-8 -*-
"""
Практикум «ТехноМаркет» — решение всех 10 блоков.
Использует оригинальные названия колонок из файлов.
Запуск:  python starter_template.py     (или по ячейкам в Jupyter)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA = "data"   # папка с исходными файлами

# ======================================================================
# Блок 1. Загрузка и первичный осмотр
# ======================================================================
orders = pd.read_csv(f"{DATA}/orders.csv", sep=";", decimal=",", encoding="utf-8-sig")
customers = pd.read_csv(f"{DATA}/customers.csv", encoding="utf-8-sig")
returns = pd.read_csv(f"{DATA}/returns.csv", encoding="utf-8-sig")
products = pd.read_excel(f"{DATA}/products.xlsx", sheet_name="Товары")
suppliers = pd.read_excel(f"{DATA}/products.xlsx", sheet_name="Поставщики")

# Определяем русские названия колонок для удобства réfерencing
# Orders: order_id(0), date(1), customer_id(2), product_id(3), quantity(4), price(5), discount%(6), channel(7), other(8)
# Customers: customer_id(0), city(1), gender(2), birth_date(3), legal_form(4), registration_date(5), segment(6)
# Returns: return_id(0), order_id(1), reason(2), status(3)
# Products: product_id(0), name(1), brand(2), series(3), cost(4), price(5), category(6)

# Создаем словари переименования для удобства
orders_rename = {
    orders.columns[0]: "order_id",
    orders.columns[1]: "Дата",
    orders.columns[2]: "customer_id",
    orders.columns[3]: "product_id",
    orders.columns[4]: "Количество",
    orders.columns[5]: "Цена",
    orders.columns[6]: "Скидка_процент",
    orders.columns[7]: "Канал",
    orders.columns[8]: "Статус",
}
customers_rename = {
    customers.columns[0]: "customer_id",
    customers.columns[1]: "Город",
    customers.columns[2]: "Пол",
    customers.columns[3]: "Дата_рождения",
    customers.columns[4]: "Юр_формат",
    customers.columns[5]: "Дата_регистрации",
    customers.columns[6]: "Сегмент",
}
returns_rename = {
    returns.columns[0]: "return_id",
    returns.columns[1]: "order_id",
    returns.columns[2]: "Причина",
    returns.columns[3]: "Статус",
}
products_rename = {
    products.columns[0]: "product_id",
    products.columns[1]: "Название",
    products.columns[2]: "Бренд",
    products.columns[3]: "Серия",
    products.columns[4]: "Себестоимость",
    products.columns[5]: "Цена",
    products.columns[6]: "Категория",
}

orders = orders.rename(columns=orders_rename)
customers = customers.rename(columns=customers_rename)
returns = returns.rename(columns=returns_rename)
products = products.rename(columns=products_rename)

print("=== Блок 1. Загрузка и первичный осмотр ===")
print(f"orders: {orders.shape} | customers: {customers.shape} | returns: {returns.shape} | products: {products.shape}")

print("\n--- Head orders ---")
print(orders.head())
print("\n--- Head customers ---")
print(customers.head())
print("\n--- Head returns ---")
print(returns.head())
print("\n--- Head products ---")
print(products.head())

# dtypes
print("\n--- Dtypes ---")
print("orders:", orders.dtypes.to_dict())
print("customers:", customers.dtypes.to_dict())
print("returns:", returns.dtypes.to_dict())
print("products:", products.dtypes.to_dict())

# info()
print("\n--- Info orders ---")
orders.info()
print("\n--- Info customers ---")
customers.info()
print("\n--- Info returns ---")
returns.info()
print("\n--- Info products ---")
products.info()

# describe()
print("\n--- Describe orders ---")
print(orders.describe(include="all"))
print("\n--- Describe customers ---")
print(customers.describe(include="all"))
print("\n--- Describe returns ---")
print(returns.describe(include="all"))
print("\n--- Describe products ---")
print(products.describe(include="all"))

# ======================================================================
# Блок 2. Очистка данных
# ======================================================================
print("\n=== Блок 2. Очистка данных ===")

# 2.1: Приводим даты в orders к формату datetime (разные форматы, dayfirst)
orders["Дата"] = pd.to_datetime(orders["Дата"], format="mixed", dayfirst=True, errors="coerce")

# 2.2: Убираем дубликаты
orders = orders.drop_duplicates()
customers = customers.drop_duplicates()

# 2.3: Убираем строки с Количество <= 0
orders = orders[orders["Количество"] > 0].copy()

# 2.4: Заполняем пропуски Скидка_процент -> 0; Пол -> "Не указан"
orders["Скидка_процент"] = orders["Скидка_процент"].fillna(0)
customers["Пол"] = customers["Пол"].fillna("Не указан")

# 2.5: Нормализуем Город (str.strip().str.title())
customers["Город"] = customers["Город"].str.strip().str.title()

# 2.6: Бренд -> strip(); Себестоимость -> медиана по категории
products["Бренд"] = products["Бренд"].str.strip()

# Считаем медиану себестоимости по категориям для later использования
products_median_cost = products.groupby("Категория")["Себестоимость"].median()

# ======================================================================
# Блок 3. Признаки, фильтрацию, сортировку
# ======================================================================
print("\n=== Блок 3. Признаки, фильтрация, сортировка ===")

# 3.1: Выручка = Количество * Цена * (1 - Скидка_процент/100)
orders["Выручка"] = orders["Количество"] * orders["Цена"] * (1 - orders["Скидка_процент"] / 100)

# 3.2: Прибыль (нужна себестоимость из products)
# Мапим себестоимость из products в orders через product_id
orders["Себестоимость"] = orders["product_id"].map(products.set_index("product_id")["Себестоимость"])
# Заполняем пропуски 0
orders["Себестоимость"] = orders["Себестоимость"].fillna(0)
orders["Прибыль"] = orders["Выручка"] - orders["Себестоимость"] * orders["Количество"]

# 3.3: Год, Месяц
orders["Год"] = orders["Дата"].dt.year
orders["Месяц"] = orders["Дата"].dt.month

# 3.4: Топ-10 онлайн-заказов со скидкой по выручке
top10_discount = (
    orders[orders["Скидка_процент"] > 0]
    .groupby("order_id")["Выручка"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)
print("\nTop-10 онлайн-заказов со скидкой по выручке:")
print(top10_discount)

# ======================================================================
# Блок 4. Группировки и агрегация
# ======================================================================
print("\n=== Блок 4. Группировки и агрегация ===")

# Объединяем orders с products для получения категории и названия
orders_with_cat = orders.merge(products[["product_id", "Название", "Категория"]], on="product_id", how="left")

# 4.1: Выручка по категориям
revenue_by_category = orders_with_cat.groupby("Категория")["Выручка"].sum().sort_values(ascending=False)
print("\nВыручка по категориям:")
print(revenue_by_category)

# 4.2: Выручка по каналам
revenue_by_channel = orders.groupby("Канал")["Выручка"].sum().sort_values(ascending=False)
print("\nВыручка по каналам:")
print(revenue_by_channel)

# 4.3: Средний чек (average check) — выручка за заказ (уникальный order_id)
avg_check = orders.groupby("order_id")["Выручка"].sum().mean()
print(f"\nСредний чек (по уникальным заказам): {avg_check:.2f}")

# 4.4: agg нескольких метрик по категориям
agg_metrics = orders_with_cat.groupby("Категория").agg(
    total_revenue=("Выручка", "sum"),
    total_profit=("Прибыль", "sum"),
    num_orders=("order_id", "nunique"),
)
print("\nAgg метрики по категориям:")
print(agg_metrics)

# ======================================================================
# Блок 5. Объединение таблиц (merge)
# ======================================================================
print("\n=== Блок 5. Объединение таблиц (merge) ===")

# 5.1: Широкия таблица (orders + customers + products)
wide_table = orders.merge(customers, on="customer_id", how="left") \
    .merge(products[["product_id", "Название", "Бренд", "Категория"]], on="product_id", how="left")
print(f"Широкая таблица: {wide_table.shape}")
print(wide_table.head())

# 5.2: Топ-5 городов по выручке
top5_cities = wide_table.groupby("Город")["Выручка"].sum().sort_values(ascending=False).head(5)
print("\nТоп-5 городов по выручке:")
print(top5_cities)

# 5.3: По сегментам
revenue_by_segment = wide_table.groupby("Сегмент")["Выручка"].sum().sort_values(ascending=False)
print("\nВыручка по сегментам:")
print(revenue_by_segment)

# ======================================================================
# Блок 6. Сводные таблицы и возвраты
# ======================================================================
print("\n=== Блок 6. Сводные таблицы и возвраты ===")

# 6.1 pivot_table Категория x Канал
if "Канал" in wide_table.columns:
    pivot = pd.pivot_table(wide_table, values="Выручка", index="Категория", columns="Канал", aggfunc="sum", fill_value=0)
    print("\nPivot Категория x Канал:")
    print(pivot)

# 6.2 Возвраты по категориям/причинам
returns_with_prod = returns.merge(wide_table[["order_id", "product_id", "Категория", "Название"]], on="order_id", how="left")
returns_by_cat = returns_with_prod.groupby("Категория").size()
print("\nВозвраты по категориям:")
print(returns_by_cat)

returns_by_reason = returns_with_prod.groupby(["Категория", "Причина"]).size().unstack(fill_value=0)
print("\nВозвраты по категориям и причинам:")
print(returns_by_reason)

# 6.3 Доля возвратов
returns_per_order = returns_with_prod["order_id"].nunique()
total_unique_orders = wide_table["order_id"].nunique()
return_rate = returns_per_order / total_unique_orders * 100 if total_unique_orders > 0 else 0
print(f"\nДоля возвратов: {return_rate:.2f}%")

# 6.4 Топ-5 возвращаемых товаров
returns_top5 = returns_with_prod.groupby("product_id").size().sort_values(ascending=False).head(5)
print("\nТоп-5 возвращаемых товаров (по product_id):")
print(returns_top5)
# Добавим названия товаров
returns_top5_names = returns_top5.index.map(products.set_index("product_id")["Название"])
print(" с названиями:")
for pid, count in returns_top5.items():
    print(f"  {products.set_index('product_id').loc[pid, 'Название']}: {count} возвратов")

# ======================================================================
# Блок 7. Временные ряды
# ======================================================================
print("\n=== Блок 7. Временные ряды ===")

# 7.1 Выручка по месяцам (resample "ME")
monthly_revenue = orders_with_cat.set_index("Дата").resample("ME")["Выручка"].sum()
print("\nВыручка по месяцам:")
print(monthly_revenue)

# 7.2 Лучший/худший месяц
best_month = monthly_revenue.idxmax()
worst_month = monthly_revenue.idxmin()
best_rev = monthly_revenue.max()
worst_rev = monthly_revenue.min()
print(f"\nЛучший месяц: {best_month.strftime('%Y-%m')} с выручкой {best_rev:.2f}")
print(f"Худший месяц: {worst_month.strftime('%Y-%m')} с выручкой {worst_rev:.2f}")

# 7.3 Скользящее среднее 3 месяца
monthly_revenue_smooth = monthly_revenue.rolling(window=3, min_periods=1).mean()
print("\nВыручка со скользящим средним 3 мес:")
print(monthly_revenue_smooth)

# ======================================================================
# Блок 8. Визуализация (matplotlib) — сохраняйте в PNG
# ======================================================================
print("\n=== Блок 8. Визуализация ===")

# 8.1 Bar по категориям
revenue_by_category_plot = orders_with_cat.groupby("Категория")["Выручка"].sum().sort_values(ascending=False)
plt.figure(figsize=(10, 6))
revenue_by_category_plot.plot(kind="bar", color="steelblue")
plt.title("Выручка по категориям")
plt.xlabel("Категория")
plt.ylabel("Выручка")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("vyruchka_po_kategoriyam.png", dpi=150, bbox_inches="tight")
plt.close()
print("Сохранено: vyruchka_po_kategoriyam.png")

# 8.2 Line по месяцам
plt.figure(figsize=(10, 6))
monthly_revenue.plot(kind="line", marker="o", color="green")
plt.title("Выручка по месяцам")
plt.xlabel("Месяц")
plt.ylabel("Выручка")
plt.grid(True)
plt.tight_layout()
plt.savefig("vyruchka_po_mesyatsam.png", dpi=150, bbox_inches="tight")
plt.close()
print("Сохранено: vyruchka_po_mesyatsam.png")

# 8.3 Pie каналы
if "Канал" in wide_table.columns:
    channel_revenue = wide_table.groupby("Канал")["Выручка"].sum()
    plt.figure(figsize=(8, 8))
    plt.pie(channel_revenue, labels=channel_revenue.index, autopct="%1.1f%%", startangle=90)
    plt.title("Распределение выручки по каналам")
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig("kanaly_pie.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Сохранено: kanaly_pie.png")

# 8.4 Barh топ-10 товаров по вырушке
top10_products_revenue = orders_with_cat.groupby("Название")["Выручка"].sum().sort_values(ascending=False).head(10)
plt.figure(figsize=(10, 8))
top10_products_revenue.plot(kind="barh", color="orange")
plt.title("Топ-10 товаров по выручке")
plt.xlabel("Выручка")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("top10_tovar_vyruchka.png", dpi=150, bbox_inches="tight")
plt.close()
print("Сохранено: top10_tovar_vyruchka.png")

# ======================================================================
# Блок 9. Выгрузка отчёта в Excel (несколько листов + оформление)
# ======================================================================
print("\n=== Блок 9. Выгрузка отчёта в Excel ===")

output_file = "otchet.xlsx"
with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    # Лист 1: Agg метрики по категориям
    agg_metrics.to_excel(writer, sheet_name="Сводные_метрики")
    
    # Лист 2: Выручка по категориям
    revenue_by_category.to_excel(writer, sheet_name="Выручка_по_категориям")
    
    # Лист 3: Возвраты по категориям
    returns_by_cat.to_excel(writer, sheet_name="Возвраты_по_категориям")
    
    # Лист 4: Топ-5 городов
    top5_cities.to_excel(writer, sheet_name="Топ-5_городов")
    
    # Лист 5: Выручка по сегментам
    revenue_by_segment.to_excel(writer, sheet_name="Выручка_по_сегментам")
    
    # Лист 6: Широкая таблица (первые строки)
    wide_table.head(100).to_excel(writer, sheet_name="Широкая_таблица", index=False)

# Оформление через openpyxl
from openpyxl import load_workbook
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side, numbers

wb = load_workbook(output_file)
ws = wb["Сводные_метрики"]

# Шапка таблицы
for cell in ws[1]:
    cell.font = Font(bold=True)
    cell.alignment = Alignment(wrap_text=True, horizontal="center")

# Автширина колонок
for col in ws.columns:
    max_len = 0
    col_letter = col[0].column_letter
    for cell in col:
        if cell.value is not None:
            max_len = max(max_len, len(str(cell.value)))
    ws.column_dimensions[col_letter].width = max_len + 2

# Оформление остальных листов
for sheet_name in wb.sheetnames:
    ws = wb[sheet_name]
    for row in ws.iter_rows():
        for cell in row:
            if cell.value is not None:
                ws.column_dimensions[cell.column_letter].width = max(
                    ws.column_dimensions[cell.column_letter].width or 0, len(str(cell.value)) + 2
                )

wb.save(output_file)
print(f"Отчёт сохранен: {output_file}")

# ======================================================================
# Блок 10 (повышенная сложность): ABC-анализ, RFM, когорты
# ======================================================================
print("\n=== Блок 10. ABC-анализ, RFM, когорты (по желанию) ===")

# 10.1 ABC-анализ по выручке
sorted_products_rev = orders_with_cat.groupby("product_id")["Выручка"].sum().sort_values(ascending=False)
total_revenue = sorted_products_rev.sum()
cumulative = sorted_products_rev.cumsum()
# Классификация: A - топ 80%, B - следующие 15%, C - остаток
product_class = pd.cut(
    cumulative / total_revenue,
    bins=[0, 0.8, 0.95, 1.0],
    labels=["A", "B", "C"]
)
abc_result = pd.DataFrame({
    "product_id": sorted_products_rev.index,
    "revenue": sorted_products_rev.values,
    "cumulative_pct": (cumulative / total_revenue * 100).values,
    "class": product_class.values
})
# Добавим названия товаров
abc_result = abc_result.merge(products[["product_id", "Название"]], on="product_id")
print("\nABC-анализ продуктов (топ-20):")
print(abc_result.head(20))

# 10.2 RFM-анализ (Recency, Frequency, Monetary)
last_order_date = orders_with_cat.groupby("customer_id")["Дата"].max()
today = orders_with_cat["Дата"].max() + pd.Timedelta(days=1)
recency = (today - last_order_date).dt.days.rename("recency")
frequency = orders_with_cat.groupby("customer_id")["order_id"].count().rename("frequency")
monetary = orders_with_cat.groupby("customer_id")["Выручка"].sum().rename("monetary")

rfm = pd.concat([recency, frequency, monetary], axis=1)
print("\nRFM-анализ (первые 10):")
print(rfm.head(10))

# 10.3 Когорты (cohorts)
# Месяц первого заказа для каждого клиента
first_order_month = orders_with_cat.groupby("customer_id")["Дата"].min()
first_order_month_str = first_order_month.dt.strftime("%Y-%m")
cohort_counts = first_order_month_str.value_counts().sort_index()
print("\nКогортный анализ (распределение когортов):")
print(cohort_counts)

print("\n=== Готово! ===")