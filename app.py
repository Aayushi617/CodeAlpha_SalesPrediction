import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Title
st.title("📈 Sales Prediction using Advertising Spend")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("Advertising.csv")
    df = df.drop(columns=["Unnamed: 0"])
    return df

df = load_data()

st.subheader("Dataset Preview")
st.dataframe(df.head())

# Correlation Heatmap
st.subheader("📊 Correlation Heatmap")
fig, ax = plt.subplots()
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

# Model Training
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

r2 = metrics.r2_score(y_test, y_pred)
mae = metrics.mean_absolute_error(y_test, y_pred)
rmse = metrics.mean_squared_error(y_test, y_pred, squared=False)

st.subheader("📌 Model Evaluation")
st.markdown(f"- **R² Score:** {r2:.2f}")
st.markdown(f"- **MAE:** {mae:.2f}")
st.markdown(f"- **RMSE:** {rmse:.2f}")

# Prediction input
st.subheader("📤 Predict Sales")
tv = st.slider("TV Advertising Budget", 0, 300, 100)
radio = st.slider("Radio Advertising Budget", 0, 50, 25)
newspaper = st.slider("Newspaper Advertising Budget", 0, 100, 20)

input_data = pd.DataFrame({'TV': [tv], 'Radio': [radio], 'Newspaper': [newspaper]})
predicted_sales = model.predict(input_data)[0]
st.success(f"Predicted Sales: {predicted_sales:.2f}")
