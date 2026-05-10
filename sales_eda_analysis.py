# ============================================================
#  Sales Data Analysis - Exploratory Data Analysis (EDA)
#  Author  : Abdul Hadi
#  GitHub  : github.com/abdulhadi1126
#  Tools   : Python, Pandas, Matplotlib, Seaborn, NumPy
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# ── Style ────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0a0a0f',
    'axes.facecolor':   '#0f0f18',
    'axes.edgecolor':   '#2a2a3a',
    'axes.labelcolor':  '#cccccc',
    'xtick.color':      '#888888',
    'ytick.color':      '#888888',
    'text.color':       '#ffffff',
    'grid.color':       '#1e1e2e',
    'grid.linewidth':   0.5,
    'font.family':      'monospace',
})
ACCENT  = '#e8521a'
ACCENT2 = '#1a6ee8'
PALETTE = [ACCENT, ACCENT2, '#28c840', '#febc2e', '#bf5af2']

# ============================================================
# 1.  GENERATE SAMPLE DATA
#     (Replace this section with: df = pd.read_csv('your_file.csv'))
# ============================================================
np.random.seed(42)
n = 500

categories   = ['Electronics', 'Clothing', 'Food & Beverage', 'Home & Garden', 'Sports']
regions      = ['North', 'South', 'East', 'West', 'Central']
months       = pd.date_range('2024-01-01', periods=12, freq='MS')

data = {
    'date':          np.random.choice(months, n),
    'category':      np.random.choice(categories, n, p=[0.30, 0.25, 0.20, 0.15, 0.10]),
    'region':        np.random.choice(regions, n),
    'sales':         np.random.lognormal(mean=5.5, sigma=0.8, size=n).round(2),
    'units_sold':    np.random.randint(1, 50, n),
    'discount_pct':  np.random.choice([0, 5, 10, 15, 20], n, p=[0.4, 0.2, 0.2, 0.1, 0.1]),
    'customer_age':  np.random.randint(18, 70, n),
    'is_returned':   np.random.choice([0, 1], n, p=[0.92, 0.08]),
}

df = pd.DataFrame(data)
df['month']   = df['date'].dt.strftime('%b')
df['month_n'] = df['date'].dt.month
df['profit']  = (df['sales'] * (1 - df['discount_pct'] / 100) * 0.35).round(2)

# ============================================================
# 2.  BASIC INFO & DATA QUALITY CHECK
# ============================================================
print("=" * 60)
print("  SALES DATA — EXPLORATORY DATA ANALYSIS")
print("  Author: Abdul Hadi | github.com/abdulhadi1126")
print("=" * 60)
print(f"\n📋 Dataset shape : {df.shape[0]} rows × {df.shape[1]} columns")
print(f"📅 Date range    : {df['date'].min().date()} → {df['date'].max().date()}")
print(f"💰 Total sales   : ${df['sales'].sum():,.2f}")
print(f"📦 Total units   : {df['units_sold'].sum():,}")
print(f"🔄 Return rate   : {df['is_returned'].mean()*100:.1f}%")

print("\n── Null Values ─────────────────────────────────────────")
print(df.isnull().sum())

print("\n── Descriptive Stats ───────────────────────────────────")
print(df[['sales', 'units_sold', 'discount_pct', 'profit']].describe().round(2))

# ============================================================
# 3.  VISUALISATIONS  (6-panel dashboard)
# ============================================================
fig = plt.figure(figsize=(18, 14))
fig.suptitle('Sales Performance Dashboard  |  Abdul Hadi  |  Data Analyst',
             fontsize=14, color='white', y=0.98, fontweight='bold')

gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.35)

# ── 3.1  Monthly Revenue Trend ───────────────────────────────
ax1 = fig.add_subplot(gs[0, :2])
monthly = df.groupby('month_n').agg(sales=('sales','sum'), profit=('profit','sum')).reset_index()
monthly = monthly.sort_values('month_n')
month_labels = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
monthly['label'] = monthly['month_n'].apply(lambda x: month_labels[x-1])

ax1.fill_between(monthly['label'], monthly['sales'], alpha=0.15, color=ACCENT)
ax1.plot(monthly['label'], monthly['sales'], color=ACCENT, linewidth=2.5, marker='o', markersize=5, label='Revenue')
ax1.fill_between(monthly['label'], monthly['profit'], alpha=0.15, color=ACCENT2)
ax1.plot(monthly['label'], monthly['profit'], color=ACCENT2, linewidth=2, marker='s', markersize=4, label='Profit')
ax1.set_title('Monthly Revenue vs Profit', color='white', fontsize=11, pad=10)
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
ax1.legend(facecolor='#1a1a2e', edgecolor='none', labelcolor='white', fontsize=9)
ax1.grid(True, axis='y')

# ── 3.2  Sales by Category  (horizontal bar) ─────────────────
ax2 = fig.add_subplot(gs[0, 2])
cat_sales = df.groupby('category')['sales'].sum().sort_values()
bars = ax2.barh(cat_sales.index, cat_sales.values, color=PALETTE[:len(cat_sales)], height=0.6)
ax2.set_title('Revenue by Category', color='white', fontsize=11, pad=10)
ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x/1000:.0f}k'))
for bar in bars:
    ax2.text(bar.get_width() + 200, bar.get_y() + bar.get_height()/2,
             f'${bar.get_width():,.0f}', va='center', fontsize=7, color='#aaaaaa')

# ── 3.3  Region Heatmap ──────────────────────────────────────
ax3 = fig.add_subplot(gs[1, :2])
pivot = df.pivot_table(values='sales', index='region', columns='month_n', aggfunc='sum')
pivot.columns = [month_labels[c-1] for c in pivot.columns]
sns.heatmap(pivot, ax=ax3, cmap='YlOrRd', linewidths=0.3, linecolor='#1a1a2e',
            fmt='.0f', annot=True, annot_kws={'size': 7},
            cbar_kws={'shrink': 0.8})
ax3.set_title('Sales Heatmap — Region × Month', color='white', fontsize=11, pad=10)
ax3.set_xlabel('')
ax3.tick_params(colors='#aaaaaa')

# ── 3.4  Discount Impact on Sales  (scatter) ─────────────────
ax4 = fig.add_subplot(gs[1, 2])
scatter_colors = [PALETTE[categories.index(c)] for c in df['category']]
ax4.scatter(df['discount_pct'], df['sales'], c=scatter_colors, alpha=0.4, s=15)
ax4.set_title('Discount % vs Sales', color='white', fontsize=11, pad=10)
ax4.set_xlabel('Discount %')
ax4.set_ylabel('Sales ($)')
ax4.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))

# ── 3.5  Customer Age Distribution ───────────────────────────
ax5 = fig.add_subplot(gs[2, 0])
ax5.hist(df['customer_age'], bins=20, color=ACCENT, edgecolor='#0a0a0f', alpha=0.85)
ax5.set_title('Customer Age Distribution', color='white', fontsize=11, pad=10)
ax5.set_xlabel('Age')
ax5.set_ylabel('Count')
ax5.axvline(df['customer_age'].mean(), color=ACCENT2, linestyle='--', linewidth=1.5,
            label=f'Mean: {df["customer_age"].mean():.0f}')
ax5.legend(facecolor='#1a1a2e', edgecolor='none', labelcolor='white', fontsize=9)

# ── 3.6  Return Rate by Category  (donut) ────────────────────
ax6 = fig.add_subplot(gs[2, 1])
ret = df.groupby('category')['is_returned'].mean() * 100
wedges, texts, autotexts = ax6.pie(ret, labels=ret.index, autopct='%1.1f%%',
                                    colors=PALETTE, startangle=90,
                                    wedgeprops={'width': 0.6, 'edgecolor': '#0a0a0f', 'linewidth': 2})
for t in texts:     t.set_color('#aaaaaa'); t.set_fontsize(7)
for t in autotexts: t.set_color('white');   t.set_fontsize(7)
ax6.set_title('Return Rate by Category', color='white', fontsize=11, pad=10)

# ── 3.7  Top 10 Sales Days ───────────────────────────────────
ax7 = fig.add_subplot(gs[2, 2])
top_days = df.groupby('date')['sales'].sum().nlargest(10).sort_values()
ax7.barh([str(d.date()) for d in top_days.index], top_days.values,
         color=ACCENT, height=0.6)
ax7.set_title('Top 10 Revenue Days', color='white', fontsize=11, pad=10)
ax7.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x/1000:.0f}k'))
ax7.tick_params(axis='y', labelsize=7)

plt.savefig('sales_eda_dashboard.png', dpi=150, bbox_inches='tight',
            facecolor='#0a0a0f')
plt.show()
print("\n✅ Dashboard saved as 'sales_eda_dashboard.png'")

# ============================================================
# 4.  KEY INSIGHTS SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("  KEY INSIGHTS")
print("=" * 60)

best_cat   = df.groupby('category')['sales'].sum().idxmax()
best_region = df.groupby('region')['sales'].sum().idxmax()
best_month  = monthly.loc[monthly['sales'].idxmax(), 'label']
avg_profit_margin = (df['profit'].sum() / df['sales'].sum()) * 100

print(f"\n🏆 Top Category    : {best_cat}")
print(f"🌍 Top Region      : {best_region}")
print(f"📅 Best Month      : {best_month}")
print(f"💹 Avg Profit Margin: {avg_profit_margin:.1f}%")
print(f"🔄 Return Rate     : {df['is_returned'].mean()*100:.1f}%")
print(f"🎯 Avg Discount    : {df['discount_pct'].mean():.1f}%")

high_disc  = df[df['discount_pct'] >= 15]['sales'].mean()
low_disc   = df[df['discount_pct'] == 0]['sales'].mean()
print(f"\n📉 Avg sale (no discount)      : ${low_disc:,.2f}")
print(f"📈 Avg sale (≥15% discount)    : ${high_disc:,.2f}")
print(f"   → Discount lifts avg sale by ${high_disc - low_disc:,.2f}")

print("\n" + "=" * 60)
print("  Analysis complete! — Abdul Hadi | github.com/abdulhadi1126")
print("=" * 60)
