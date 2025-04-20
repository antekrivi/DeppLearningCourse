import pandas as pd
import matplotlib.pyplot as plt

file_path = 'chocolate_sales.csv'

df = pd.read_csv(file_path)

print(df.head())
print(df.info())
print(df.isnull().sum())  # NaN po kolonama

df.fillna(df.select_dtypes(include=['float64', 'int64']).mean(), inplace=True)

for column in df.select_dtypes(include=['object']).columns:
    df[column].fillna(df[column].mode()[0], inplace=True)

df.drop_duplicates(inplace=True)

if 'Amount' in df.columns:
    df['Amount'] = df['Amount'].str.replace(r'[\$,]', '', regex=True).str.strip()
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
    df['price_eur'] = df['Amount'] * 0.85  # Pretpostavljeni tečaj


# Boxplot za cijene (EUR)
plt.figure(figsize=(8, 6))
plt.boxplot(df['price_eur'].dropna())
plt.title('Boxplot za cijene (EUR)')
plt.ylabel('Cijena (EUR)')
plt.tight_layout()
plt.show()

# Histogram za cijene (EUR)
plt.figure(figsize=(8, 6))
plt.hist(df['price_eur'].dropna(), bins=30, edgecolor='black')
plt.title('Histogram za cijene (EUR)')
plt.xlabel('Cijena u EUR')
plt.ylabel('Frekvencija')
plt.tight_layout()
plt.show()

sales_by_person = df.groupby('Sales Person')['Amount'].sum()

plt.figure(figsize=(8, 8))
plt.pie(sales_by_person, labels=sales_by_person.index, autopct='%1.1f%%', startangle=140)
plt.title('Udio prodaje po prodavaču')
plt.axis('equal')  # Jednaki omjeri osovina za kružni oblik
plt.tight_layout()
plt.show()


# Spremanje obrađenih podataka
output_path = 'C:\\Users\\akrivacic\\Downloads\\chocolate_sales_processed.csv'
df.to_csv(output_path, index=False)
