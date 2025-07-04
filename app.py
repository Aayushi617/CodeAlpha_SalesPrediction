import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

st.set_page_config(page_title="📈 Sales Prediction", layout="wide")

st.title("📊 Sales Prediction based on Advertising Spend")

@st.cache_data
def load_data():
    df = pd.read_csv("Advertising.csv")
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")
    return df

df = load_data()

st.subheader("🔍 Dataset Preview")
st.dataframe(df.head())

# --- 1. Correlation Heatmap ---
st.subheader("📌 Heatmap: Feature Correlation")
fig1, ax1 = plt.subplots()
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax1)
st.pyplot(fig1)

# --- 2. TV vs Sales ---
st.subheader("📺 TV Budget vs Sales")
fig2, ax2 = plt.subplots()
sns.regplot(x='TV', y='Sales', data=df, ax=ax2, color="green")
ax2.set_title("TV vs Sales")
st.pyplot(fig2)

# --- 3. Radio vs Sales ---
st.subheader("📻 Radio Budget vs Sales")
fig3, ax3 = plt.subplots()
sns.regplot(x='Radio', y='Sales', data=df, ax=ax3, color="orange")
ax3.set_title("Radio vs Sales")
st.pyplot(fig3)

# --- 4. Average Sales by High/Low TV Spend ---
df['TV_Level'] = ['High' if x > df['TV'].median() else 'Low' for x in df['TV']]
st.subheader("🔺 Average Sales by TV Spend Level")
avg_sales = df.groupby('TV_Level')['Sales'].mean().reset_index()
fig4, ax4 = plt.subplots()
sns.barplot(x='TV_Level', y='Sales', data=avg_sales, palette='Set2', ax=ax4)
ax4.set_title("High vs Low TV Budget - Avg Sales")
st.pyplot(fig4)

# --- 5. Actual vs Predicted Sales Scatter ---
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

st.subheader("📉 Actual vs Predicted Sales")
result_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
fig5, ax5 = plt.subplots()
sns.scatterplot(x='Actual', y='Predicted', data=result_df, ax=ax5, color='blue')
ax5.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
ax5.set_title("Actual vs Predicted Sales")
st.pyplot(fig5)

# --- Model Metrics ---
st.subheader("📌 Model Performance")
st.markdown(f"- **R² Score**: `{metrics.r2_score(y_test, y_pred):.2f}`")
st.markdown(f"- **MAE**: `{metrics.mean_absolute_error(y_test, y_pred):.2f}`")
st.markdown(f"- **RMSE**: `{metrics.mean_squared_error(y_test, y_pred) ** 0.5:.2f}`")

# --- Prediction Interface ---
st.subheader("🎯 Predict Your Own Sales")
tv = st.slider("TV Budget", 0, 300, 150)
radio = st.slider("Radio Budget", 0, 50, 25)
newspaper = st.slider("Newspaper Budget", 0, 100, 20)

input_data = pd.DataFrame({'TV': [tv], 'Radio': [radio], 'Newspaper': [newspaper]})
prediction = model.predict(input_data)[0]

st.success(f"📢 Predicted Sales: **{prediction:.2f} units**")
