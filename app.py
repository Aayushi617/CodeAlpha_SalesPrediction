import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

st.set_page_config(page_title="Sales Prediction App", layout="wide")

st.title("📈 Sales Prediction using Advertising Spend")

# ---- Load Data ----
@st.cache_data
def load_data():
    df = pd.read_csv("Advertising.csv")
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")
    return df

df = load_data()

# ---- Dataset Preview ----
st.subheader("🧾 Dataset Preview")
st.dataframe(df.head())

# ---- 1. Correlation Heatmap ----
st.subheader("📊 Correlation Heatmap")
fig1, ax1 = plt.subplots()
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax1)
st.pyplot(fig1)

# ---- 2. Pairplot (All Features) ----
st.subheader("🔗 Pairplot: Relationships Between All Variables")
sns_plot = sns.pairplot(df)
buf = io.BytesIO()
sns_plot.savefig(buf, format="png")
st.image(buf)

# ---- 3. Feature vs Sales: TV, Radio, Newspaper ----
st.subheader("📈 Advertising Spend vs Sales")
fig2, ax2 = plt.subplots(1, 3, figsize=(18, 5))

sns.regplot(x='TV', y='Sales', data=df, ax=ax2[0], color='green')
ax2[0].set_title("TV vs Sales")

sns.regplot(x='Radio', y='Sales', data=df, ax=ax2[1], color='orange')
ax2[1].set_title("Radio vs Sales")

sns.regplot(x='Newspaper', y='Sales', data=df, ax=ax2[2], color='red')
ax2[2].set_title("Newspaper vs Sales")

st.pyplot(fig2)

# ---- Feature Engineering for Bar Plot ----
df['TV_Level'] = ['High' if x > df['TV'].median() else 'Low' for x in df['TV']]

# ---- 4. Bar Plot: Avg Sales by TV Spend Level ----
st.subheader("📊 Average Sales: High vs Low TV Budget")
avg_sales_by_tv = df.groupby('TV_Level')['Sales'].mean().reset_index()
fig3, ax3 = plt.subplots()
sns.barplot(x='TV_Level', y='Sales', data=avg_sales_by_tv, palette='Set2', ax=ax3)
ax3.set_title("Average Sales by TV Budget Level")
st.pyplot(fig3)

# ---- Train/Test Split ----
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---- Model Training ----
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ---- 5. Actual vs Predicted Plot ----
st.subheader("📉 Actual vs Predicted Sales")
result_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
fig4, ax4 = plt.subplots()
sns.scatterplot(x='Actual', y='Predicted', data=result_df, ax=ax4, color="blue")
ax4.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')  # Diagonal line
ax4.set_xlabel("Actual Sales")
ax4.set_ylabel("Predicted Sales")
ax4.set_title("Actual vs Predicted Sales")
st.pyplot(fig4)

# ---- 6. Residual Plot ----
st.subheader("📉 Residual Plot")
residuals = y_test - y_pred
fig5, ax5 = plt.subplots()
sns.histplot(residuals, kde=True, color='purple', bins=20, ax=ax5)
ax5.set_title("Distribution of Residuals (Prediction Errors)")
ax5.set_xlabel("Residual (Actual - Predicted)")
st.pyplot(fig5)

# ---- Model Evaluation ----
st.subheader("📌 Model Evaluation")
r2 = metrics.r2_score(y_test, y_pred)
mae = metrics.mean_absolute_error(y_test, y_pred)
rmse = metrics.mean_squared_error(y_test, y_pred) ** 0.5

st.markdown(f"- **R² Score:** `{r2:.2f}`")
st.markdown(f"- **MAE (Mean Absolute Error):** `{mae:.2f}`")
st.markdown(f"- **RMSE (Root Mean Squared Error):** `{rmse:.2f}`")

# ---- Prediction Input ----
st.subheader("🎯 Predict Sales Based on New Advertising Budget")

tv = st.slider("📺 TV Advertising Budget", 0, 300, 150)
radio = st.slider("📻 Radio Advertising Budget", 0, 50, 25)
newspaper = st.slider("🗞 Newspaper Advertising Budget", 0, 100, 20)

input_df = pd.DataFrame({'TV': [tv], 'Radio': [radio], 'Newspaper': [newspaper]})
predicted_sales = model.predict(input_df)[0]
st.success(f"📢 Predicted Sales: **{predicted_sales:.2f} units**")
