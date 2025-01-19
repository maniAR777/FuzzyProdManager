import pulp
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# ------------------------------
# کلاس اعداد فازی مثلثی
# ------------------------------


class TriangularFuzzyNumber:
    def __init__(self, l, m, r):
        self.l = l  # مقدار کمینه
        self.m = m  # مقدار میانه
        self.r = r  # مقدار بیشینه

    def get_jimenez_defuzz(self):
        """
        روش جیمنز برای دیفازی کردن عدد فازی مثلثی:
        EV = (l + 2*m + r) / 4
        """
        return (self.l + 2 * self.m + self.r) / 4

    def get_e1_e2(self):
        """
        محاسبه E1 و E2 برای دیفازی‌سازی با آلفا
        E1 = (a1 + a2) / 2 = (l + m) / 2
        E2 = (a3 + a4) / 2 = (m + r) / 2
        """
        E1 = (self.l + self.m) / 2
        E2 = (self.m + self.r) / 2
        return E1, E2


def defuzzify_with_alpha(fuzzy_number, alpha):
    """
    دیفازی‌سازی یک عدد فازی مثلثی با آلفا
    بر اساس فرمول‌های داده شده:
    برای سمت چپ قیود: E2 + alpha * E1 * (1 - alpha)
    برای سمت راست قیود: alpha * E2 + (1 - alpha) * E1
    """
    E1, E2 = fuzzy_number.get_e1_e2()
    left_side = E2 + alpha * E1 * (1 - alpha)
    right_side = alpha * E2 + (1 - alpha) * E1
    return left_side, right_side


# ------------------------------
# داده‌های ورودی (4 دوره)
# ------------------------------
data_new = [
    {
        "Period": 1,
        "Date Range": "1403/05/02 تا 1403/05/08",
        "Demand (Optimistic)": 17600,
        "Demand (Probable)": 16000,
        "Demand (Pessimistic)": 14400,
        "Cost Normal Time": 27.0,
        "Cost Overtime": 28.0,
        "Cost Inventory": 4.0,
        "Cost Backlog": 144.0,
        "Cost Maintenance": 720.0,
        "Total System Failure Cost": 1713549,  # فقط جهت نمایش
        "Max Production Normal": 8000,
        "Max Production Overtime": 10000,
        "Storage Capacity": 16158,
        "Failure Rate (%)": 1.48
    },
    {
        "Period": 2,
        "Date Range": "1403/05/09 تا 1403/05/15",
        "Demand (Optimistic)": 18260,
        "Demand (Probable)": 16600,
        "Demand (Pessimistic)": 14940,
        "Cost Normal Time": 27.0,
        "Cost Overtime": 28.0,
        "Cost Inventory": 4.0,
        "Cost Backlog": 145.0,
        "Cost Maintenance": 720.0,
        "Total System Failure Cost": 2566497,  # فقط جهت نمایش
        "Max Production Normal": 8300,
        "Max Production Overtime": 9000,
        "Storage Capacity": 16046,
        "Failure Rate (%)": 1.07
    },
    {
        "Period": 3,
        "Date Range": "1403/05/16 تا 1403/05/22",
        "Demand (Optimistic)": 6821,
        "Demand (Probable)": 6201,
        "Demand (Pessimistic)": 5580,
        "Cost Normal Time": 27.0,
        "Cost Overtime": 28.0,
        "Cost Inventory": 4.2,
        "Cost Backlog": 145.0,
        "Cost Maintenance": 720.0,
        "Total System Failure Cost": 1565221,  # فقط جهت نمایش
        "Max Production Normal": 7800,
        "Max Production Overtime": 8000,
        "Storage Capacity": 14540,
        "Failure Rate (%)": 1.19
    },
    {
        "Period": 4,
        "Date Range": "1403/05/23 تا 1403/05/29",
        "Demand (Optimistic)": 18480,
        "Demand (Probable)": 16800,
        "Demand (Pessimistic)": 15120,
        "Cost Normal Time": 27.0,
        "Cost Overtime": 28.0,
        "Cost Inventory": 4.1,
        "Cost Backlog": 167.0,
        "Cost Maintenance": 720.0,
        "Total System Failure Cost": 0,
        "Max Production Normal": 8400,
        "Max Production Overtime": 10000,
        "Storage Capacity": 16766,
        "Failure Rate (%)": 0.0
    }
]

df = pd.DataFrame(data_new)
print("داده‌های ورودی (4 دوره):")
print(df, "\n")

# ------------------------------
# ساخت دیکشنری داده‌ها
# (اعداد فازی برای تقاضا؛ و بقیه اگر قطعی‌اند، به صورت (l=m=r))
# ------------------------------
dates = df["Date Range"].tolist()
D = len(dates)

# فهرست فازی/قطعی
planned_demand_fuzzy = []
max_capacity_fuzzy = []
overtime_capacity_fuzzy = []
normal_cost_fuzzy = []

overtime_cost_list = []
inventory_cost_list = []
backlog_cost_list = []
repair_cost_list = []
storage_capacity_list = []

# مقادیر واقعی استفاده (برای تصمیم PM)
actual_usage_list = [0.6, 0.85, 0.8, 0.9]
usage_threshold = 0.75

for i, row in df.iterrows():
    # عدد فازی تقاضا
    pess = row["Demand (Pessimistic)"]
    prob = row["Demand (Probable)"]
    opti = row["Demand (Optimistic)"]
    planned_demand_fuzzy.append(TriangularFuzzyNumber(pess, prob, opti))

    # ظرفیت عادی (قطعی => به شکل فازی سه نقطه یکسان)
    cap_normal = row["Max Production Normal"]
    max_capacity_fuzzy.append(TriangularFuzzyNumber(
        cap_normal, cap_normal, cap_normal))

    # ظرفیت اضافه‌کاری (قطعی)
    cap_overtime = row["Max Production Overtime"]
    overtime_capacity_fuzzy.append(TriangularFuzzyNumber(
        cap_overtime, cap_overtime, cap_overtime))

    # هزینه عادی (قطعی)
    c_normal = row["Cost Normal Time"]
    normal_cost_fuzzy.append(
        TriangularFuzzyNumber(c_normal, c_normal, c_normal))

    # سایر هزینه‌ها (قطعی)
    overtime_cost_list.append(row["Cost Overtime"])
    inventory_cost_list.append(row["Cost Inventory"])
    backlog_cost_list.append(row["Cost Backlog"])
    repair_cost_list.append(row["Cost Maintenance"])
    storage_capacity_list.append(row["Storage Capacity"])

# ------------------------------
# مرحله 2: دیفازی‌سازی با روش جیمنز
# ------------------------------
planned_demand_defuzz = []
max_capacity_defuzz = []
overtime_capacity_defuzz = []
normal_cost_defuzz = []

for t in range(D):
    # تقاضا
    d_crisp = planned_demand_fuzzy[t].get_jimenez_defuzz()
    planned_demand_defuzz.append(d_crisp)

    # ظرفیت عادی
    cap_crisp = max_capacity_fuzzy[t].get_jimenez_defuzz()
    max_capacity_defuzz.append(cap_crisp)

    # ظرفیت اضافه‌کاری
    cap_ot_crisp = overtime_capacity_fuzzy[t].get_jimenez_defuzz()
    overtime_capacity_defuzz.append(cap_ot_crisp)

    # هزینه عادی
    costn_crisp = normal_cost_fuzzy[t].get_jimenez_defuzz()
    normal_cost_defuzz.append(costn_crisp)

# ------------------------------
# مرحله 3: حل مدل قطعی
# ------------------------------
alpha_values = np.arange(0.1, 1.1, 0.1)  # 0.1 تا 1 با گام 0.1

# ذخیره نتایج برای هر آلفا
results_alpha = []

for alpha in alpha_values:
    print(f"\nحل مدل برای آلفا = {alpha:.1f}")

    # ایجاد مدل جدید برای هر آلفا
    model = pulp.LpProblem(
        f"Production_Maintenance_Optimization_Alpha_{alpha:.1f}", pulp.LpMinimize)

    # متغیرها
    P_ri = pulp.LpVariable.dicts(
        "P_ri", range(D), lowBound=0)       # تولید عادی
    P_oi = pulp.LpVariable.dicts("P_oi", range(
        D), lowBound=0)       # تولید اضافه‌کاری
    Inv_it = pulp.LpVariable.dicts("Inv_it", range(D), lowBound=0)   # موجودی
    B_it = pulp.LpVariable.dicts("B_it", range(D), lowBound=0)       # کمبود
    PM_t = pulp.LpVariable.dicts("PM_t", range(
        D), cat=pulp.LpBinary)  # نگهداری (1 یا 0)

    # تابع هدف:
    # Z = sum(P_ri[t] * C_normal + P_oi[t] * C_overtime + Inv_it[t] * C_inventory + B_it[t] * C_backlog) + sum(PM_t[t] * C_repair)
    production_cost = pulp.lpSum([
        P_ri[t] * normal_cost_defuzz[t] +
        P_oi[t] * overtime_cost_list[t] +
        Inv_it[t] * inventory_cost_list[t] +
        B_it[t] * backlog_cost_list[t]
        for t in range(D)
    ])
    maintenance_cost = pulp.lpSum([
        repair_cost_list[t] * PM_t[t]
        for t in range(D)
    ])

    # تابع هدف با وزن‌ها
    weight_production = 0.6
    weight_maintenance = 0.4
    model += weight_production * production_cost + \
        weight_maintenance * maintenance_cost, "ObjectiveCost"

    # قیود مربوط به نگهداری
    required_maintenance_periods = sum(
        1 for usage in actual_usage_list if usage >= usage_threshold)
    model += pulp.lpSum([PM_t[t] for t in range(D)]
                        ) == required_maintenance_periods, "Required_PM_count"

    for t in range(D):
        if actual_usage_list[t] >= usage_threshold:
            model += PM_t[t] == 1, f"PM_Force_{t}"
        else:
            # اگر نگهداری اختیاری باشد، نیازی به تعریف محدودیت خاص نیست
            pass  # می‌توانید محدودیت‌های دیگری اضافه کنید اگر نیاز بود

    # قیود تقاضا با دیفازی‌سازی بر اساس آلفا
    for t in range(D):
        demand_fuzzy = planned_demand_fuzzy[t]
        # جایگزینی D_planned(t) با alpha * E2 + (1 - alpha) * E1
        E1_a, E2_a = demand_fuzzy.get_e1_e2()
        D_planned_defuzz = alpha * E2_a + (1 - alpha) * E1_a

        if t == 0:
            model += (P_ri[t] + P_oi[t] - Inv_it[t] + B_it[t]
                      ) >= D_planned_defuzz, f"Demand_{t}"
        else:
            model += (P_ri[t] + P_oi[t] + Inv_it[t-1] - Inv_it[t] +
                      B_it[t]) >= D_planned_defuzz, f"Demand_{t}"

    # قیود ظرفیت تولید با آلفا
    for t in range(D):
        # ظرفیت عادی: P_max_normal(t) * (1 - 0.5 * PM_t)
        # ظرفیت اضافه‌کاری: P_max_overtime(t) * (1 - 0.5 * PM_t)
        model += P_ri[t] <= max_capacity_defuzz[t] * \
            (1 - 0.5 * PM_t[t]), f"CapNormal_{t}"
        model += P_oi[t] <= overtime_capacity_defuzz[t] * \
            (1 - 0.5 * PM_t[t]), f"CapOvertime_{t}"

    # قید حداقل موجودی
    for t in range(D):
        # جایگزینی Inv_it[t] >= 0.05 * D_planned(t)
        D_planned_defuzz = alpha * (planned_demand_fuzzy[t].get_e1_e2()[1]) + (
            1 - alpha) * (planned_demand_fuzzy[t].get_e1_e2()[0])
        model += Inv_it[t] >= 0.05 * D_planned_defuzz, f"MinInv_{t}"

    # حل مدل
    model.solve(pulp.PULP_CBC_CMD(msg=0))
    status = pulp.LpStatus[model.status]
    print("وضعیت حل:", status)

    if status == "Optimal":
        Z_opt = pulp.value(model.objective)
        print("مقدار تابع هدف (هزینه):", round(Z_opt, 2))

        # استخراج مقادیر
        P_ri_opt = [pulp.value(P_ri[t]) for t in range(D)]
        P_oi_opt = [pulp.value(P_oi[t]) for t in range(D)]
        Inv_opt = [pulp.value(Inv_it[t]) for t in range(D)]
        B_opt = [pulp.value(B_it[t]) for t in range(D)]
        PM_opt = [int(pulp.value(PM_t[t])) for t in range(D)]

        # جدول نتایج
        results_dict = {
            "Regular Production": [round(x, 2) for x in P_ri_opt],
            "Overtime Production": [round(x, 2) for x in P_oi_opt],
            "Inventory": [round(x, 2) for x in Inv_opt],
            "Backlog": [round(x, 2) for x in B_opt],
            "Maintenance(1=Yes)": PM_opt
        }
        df_results = pd.DataFrame(
            results_dict, index=[f"Period {t+1}" for t in range(D)])
        print("\nخلاصه نتایج:")
        print(df_results)

        # ذخیره نتایج برای تحلیل حساسیت
        results_alpha.append({
            'Alpha': alpha,
            'Objective': Z_opt,
            'Regular Production': P_ri_opt,
            'Overtime Production': P_oi_opt,
            'Inventory': Inv_opt,
            'Backlog': B_opt,
            'Maintenance': PM_opt
        })
    else:
        print("مدل پاسخ بهینه نیافت یا شدنی نیست.")
        results_alpha.append({
            'Alpha': alpha,
            'Objective': None,
            'Regular Production': [None]*D,
            'Overtime Production': [None]*D,
            'Inventory': [None]*D,
            'Backlog': [None]*D,
            'Maintenance': [None]*D
        })

    # تعریف مجموعه‌های فازی F̃ و S̃
    # μ_F̃("X"(α_K))=α_K
    mu_F = alpha

    # μ_S̃(X (α_K))=K_G̃(Z̃(α_K))
    # تعریف تابع K_G̃، به عنوان مثال:
    def K_G(Z): return 1 / (1 + Z)  # تابع نمونه

    mu_S = K_G(Z_opt)

    # تعریف مجموعه فازی D̃=F̃∩S̃ با T-norm ضرب جبری
    mu_D = mu_F * mu_S  # استفاده از ضرب جبری به عنوان T-norm

    # اضافه کردن μ_D به نتایج
    results_alpha[-1]['mu_D'] = mu_D

# ------------------------------
# مرحله 4: تحلیل حساسیت و مقایسه نتایج
# ------------------------------
# ساخت DataFrame از نتایج
df_results_alpha = pd.DataFrame(results_alpha)

# حذف ردیف‌هایی که Objective ندارد
df_valid = df_results_alpha.dropna(subset=['Objective'])

if not df_valid.empty:
    # انتخاب بهترین alpha (بر اساس بیشترین μ_D)
    best_alpha_row = df_valid.loc[df_valid['mu_D'].idxmax()]
    best_alpha = best_alpha_row['Alpha']
    best_objective = best_alpha_row['Objective']
    best_regular = best_alpha_row['Regular Production']
    best_overtime = best_alpha_row['Overtime Production']
    best_inventory = best_alpha_row['Inventory']
    best_backlog = best_alpha_row['Backlog']
    best_maintenance = best_alpha_row['Maintenance']

    print(f"\nبهترین مقدار Alpha: {best_alpha}")
    print(f"کمترین هزینه تابع هدف: {round(best_objective, 2)}")

    # ساخت DataFrame برای بهترین alpha
    results_dict_best = {
        "Regular Production": [round(x, 2) for x in best_regular],
        "Overtime Production": [round(x, 2) for x in best_overtime],
        "Inventory": [round(x, 2) for x in best_inventory],
        "Backlog": [round(x, 2) for x in best_backlog],
        "Maintenance(1=Yes)": best_maintenance
    }
    df_best = pd.DataFrame(
        results_dict_best, index=[f"Period {t+1}" for t in range(D)])
    print("\nخلاصه نتایج برای بهترین Alpha:")
    print(df_best)

    # نمودار تولید در مقابل تقاضا
    periods = np.arange(1, D + 1)
    plt.figure(figsize=(10, 4))
    plt.plot(periods, best_regular, marker="o", label="Regular Production")
    plt.plot(periods, best_overtime, marker="o", label="Overtime Production")
    plt.plot(periods, planned_demand_defuzz, marker="x",
             linestyle="--", color="red", label="Demand (Jimenez)")
    plt.xlabel("Period")
    plt.ylabel("Quantity")
    plt.title(f"Production vs. Demand (Alpha = {best_alpha:.1f})")
    plt.legend()
    plt.grid(True)
    plt.show()

    # نمودار موجودی و کمبود
    plt.figure(figsize=(10, 4))
    width = 0.3
    plt.bar(periods - width/2, best_inventory, width=width,
            label="Inventory", color="green")
    plt.bar(periods + width/2, best_backlog, width=width,
            label="Backlog", color="red")
    plt.xlabel("Period")
    plt.ylabel("Units")
    plt.title(f"Inventory and Backlog (Alpha = {best_alpha:.1f})")
    plt.legend()
    plt.grid(True)
    plt.show()

    # نمودار نگهداری
    plt.figure(figsize=(6, 3))
    colors = ["blue" if x == 1 else "gray" for x in best_maintenance]
    plt.bar(periods, best_maintenance, color=colors)
    plt.xlabel("Period")
    plt.ylabel("Maintenance (1=Yes, 0=No)")
    plt.title(f"Maintenance Plan (Alpha = {best_alpha:.1f})")
    plt.grid(True)
    plt.yticks([0, 1], ["No", "Yes"])
    plt.show()
else:
    print("هیچ مدل بهینه‌ای یافت نشد.")

# نمودار تابع هدف نسبت به آلفا
plt.figure(figsize=(10, 6))
plt.plot(df_results_alpha['Alpha'], df_results_alpha['Objective'], marker='o')
plt.xlabel('Alpha')
plt.ylabel('Objective Cost')
plt.title('Objective Cost vs Alpha')
plt.grid(True)
plt.show()

# نمایش نتایج به صورت جدول برای هر آلفا
for idx, row in df_results_alpha.iterrows():
    alpha = row['Alpha']
    objective = row['Objective']
    print(f"\nAlpha: {alpha}")
    print(f"Objective Cost: {round(objective, 2)
                             }" if objective else "Objective Cost: None")
    for t in range(D):
        if row['Regular Production'][t] is not None:
            print(f"  Period {t+1}: Regular Production = {round(row['Regular Production'][t], 2)}, "
                  f"Overtime Production = {
                      round(row['Overtime Production'][t], 2)}, "
                  f"Inventory = {round(row['Inventory'][t], 2)}, "
                  f"Backlog = {round(row['Backlog'][t], 2)}, "
                  f"Maintenance = {row['Maintenance'][t]}")
        else:
            print(f"  Period {t+1}: No optimal solution.")

اخرین کد 