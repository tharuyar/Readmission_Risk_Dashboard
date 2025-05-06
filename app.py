import streamlit as st
st.set_page_config(page_title="Readmission Risk Dashboard", layout="wide")  # MUST BE IMMEDIATELY AFTER streamlit import

import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_extras.metric_cards import style_metric_cards
import shap


# ---- SET SEABORN STYLE ----
sns.set_theme(style="whitegrid", rc={
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9
})

# ---- LOAD ARTIFACTS ----
model = joblib.load("xgb_tuned_model.pkl")
preprocessor = joblib.load("xgb_tuned_preprocessor.pkl")
features = pd.read_csv("xgb_features.csv", header=None)[0].tolist()
features = [f for f in features if f != '0']
data = pd.read_csv("readmission_test_data.csv")

# ---- PAGE SETUP ----
st.set_page_config(page_title="Readmission Risk Dashboard", layout="wide")
st.markdown("""
<style>
    .patient-card {
        background: linear-gradient(to right, #f8fbff, #e6f0ff);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05);
        margin-bottom: 15px;
        font-size: 16px;
    }
    .card-header {
        font-size: 18px;
        font-weight: bold;
        color: #005f99;
    }
    .card-line {
        margin-top: 5px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🏥 HealthOS: 30-Day Readmission Risk Dashboard")

# ---- TABS ----
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔍 Predict by Patient", "📝 Manual Entry", "📊 Insights", "📋 High-Risk Patients", "💡 Feature Importance"])


# ---- TAB 1: PREDICT BY PATIENT ----
with tab1:
    st.subheader("🔍 Predict by Selecting a Patient")
    patient_options = data.index.tolist()
    selected_index = st.selectbox("Select Patient Index", options=patient_options)
    selected_row = data.loc[selected_index]
    input_df = pd.DataFrame([selected_row[features]])

    st.markdown("### 🧬 Patient Clinical Summary")
    st.markdown(f"""
    <div class='patient-card'>
        <div class='card-line'><strong>Age:</strong> {selected_row['AGE']}</div>
        <div class='card-line'><strong>Gender:</strong> {selected_row['GENDER']}</div>
        <div class='card-line'><strong>Race:</strong> {selected_row['RACE']}</div>
        <div class='card-line'><strong>Condition:</strong> {selected_row['DESCRIPTION']}</div>
        <div class='card-line'><strong>BMI:</strong> {round(selected_row['BMI'], 1)}</div>
        <div class='card-line'><strong>Length of Stay:</strong> {int(selected_row['LOS'])} days</div>
        <div class='card-line'><strong>Smoking:</strong> {selected_row['Smoking_Status']}</div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🧠 Predict Readmission Risk", key="dropdown_predict"):
        # Make prediction
        X_transformed = preprocessor.transform(input_df)
        prob = model.predict_proba(X_transformed)[0][1]
        label = "High Risk" if prob >= 0.30 else "Low Risk"

        # Show prediction result
        st.markdown(f"""
        <div style='background-color:#fff3cd; padding:15px; border-radius:10px;'>
            <strong>🔮 Predicted Probability:</strong> <span style='color:#0d6efd; font-size:18px;'>{prob:.2f}</span><br>
            {'<span style="color:red; font-weight:bold;">⚠️ High Risk — Consider early intervention.</span>' if label == 'High Risk' else '<span style="color:green; font-weight:bold;">✅ Low Risk — Patient likely stable.</span>'}
        </div>
        """, unsafe_allow_html=True)

        


# ---- TAB 3: INSIGHTS ----
with tab3:
    st.subheader("\U0001F4CA Dataset Insights")
    total_patients = len(data)
    readmit_rate = data['READMISSION_30'].mean() * 100
    avg_los = data['LOS'].mean()
    col1, col2, col3 = st.columns(3)
    col1.metric("\U0001F465 Total Patients", f"{total_patients}")
    col2.metric("\U0001F4C9 Readmission Rate", f"{readmit_rate:.2f}%")
    col3.metric("\U0001F6CC Avg LOS", f"{avg_los:.1f} days")
    style_metric_cards()

    st.markdown("### 📊 Total Patients by Predicted Risk Level")

    # Ensure Risk_Score is computed
    if 'Risk_Score' not in data.columns:
        X_all = preprocessor.transform(data[features])
        data['Risk_Score'] = model.predict_proba(X_all)[:, 1]

    # Create Risk Label column
    data['Risk_Label'] = data['Risk_Score'].apply(lambda x: 'High Risk' if x >= 0.30 else 'Low Risk')

    # Count risk categories
    risk_counts = data['Risk_Label'].value_counts().reindex(['High Risk', 'Low Risk']).reset_index()
    risk_counts.columns = ['Risk_Level', 'Count']

    # Define color mapping
    color_map = {'High Risk': 'red', 'Low Risk': 'green'}

    # Plot
    fig, ax = plt.subplots()
    bars = sns.barplot(
        data=risk_counts,
        x='Risk_Level',
        y='Count',
        palette=[color_map[label] for label in risk_counts['Risk_Level']]
    )

    # Add count labels above bars
    for i, row in risk_counts.iterrows():
        ax.text(i, row['Count'] + 2, f"{int(row['Count'])}", ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_title("High vs Low Risk Patients")
    ax.set_ylabel("Number of Patients")
    ax.set_xlabel("Risk Category")

    st.pyplot(fig)
    plt.clf()


    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Readmission by Age Group**")
        data['Age_Group'] = pd.cut(data['AGE'], bins=[0, 30, 45, 60, 75, 90], labels=['0-30', '31-45', '46-60', '61-75', '76-90'])
        age_chart = data.groupby('Age_Group', observed=True)['READMISSION_30'].mean().reset_index()
        sns.barplot(data=age_chart, x='Age_Group', y='READMISSION_30', palette='Set2')
        st.pyplot(plt.gcf())
        plt.clf()
    with col2:
        st.markdown("**LOS vs Readmission**")
        sns.boxplot(data=data, x='READMISSION_30', y='LOS', palette='Set3')
        st.pyplot(plt.gcf())
        plt.clf()

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**Readmission by Gender**")
        gender_chart = data.groupby('GENDER')['READMISSION_30'].mean().reset_index()
        sns.barplot(data=gender_chart, x='GENDER', y='READMISSION_30', palette='pastel')
        st.pyplot(plt.gcf())
        plt.clf()
    with col4:
        st.markdown("**Readmission by Race**")
        race_chart = data.groupby('RACE')['READMISSION_30'].mean().reset_index()
        sns.barplot(data=race_chart, x='RACE', y='READMISSION_30', palette='coolwarm')
        st.pyplot(plt.gcf())
        plt.clf()

    st.markdown("---")
    st.markdown("**Readmission by Top Conditions**")
    top_conditions = data['DESCRIPTION'].value_counts().nlargest(10).index.tolist()
    condition_chart = data[data['DESCRIPTION'].isin(top_conditions)]
    cond_rate = condition_chart.groupby('DESCRIPTION')['READMISSION_30'].mean().reset_index()
    sns.barplot(data=cond_rate, y='DESCRIPTION', x='READMISSION_30', palette='viridis')
    st.pyplot(plt.gcf())
    plt.clf()

    st.markdown("**Hemoglobin Distribution by Readmission**")
    sns.violinplot(data=data, x='READMISSION_30', y='Hemoglobin', palette='Set3')
    st.pyplot(plt.gcf())
    plt.clf()


# ---- TAB 4: HIGH-RISK PATIENTS ----
with tab4:
    st.subheader("\U0001F4CB High-Risk Patients")
    X_all = preprocessor.transform(data[features])
    data['Risk_Score'] = model.predict_proba(X_all)[:, 1]
    high_risk_df = data[data['Risk_Score'] >= 0.30]
    st.dataframe(high_risk_df[['AGE', 'GENDER', 'DESCRIPTION', 'LOS', 'Risk_Score']].sort_values(by='Risk_Score', ascending=False))

# ---- TAB 5: FEATURE IMPORTANCE ----
with tab5:
    st.subheader("\U0001F4A1 Feature Importance")
    importances = model.feature_importances_
    feat_names = preprocessor.get_feature_names_out()
    importance_df = pd.DataFrame({"Feature": feat_names, "Importance": importances})
    importance_df = importance_df.sort_values(by="Importance", ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=importance_df, y='Feature', x='Importance', palette='Spectral', ax=ax)
    ax.set_title("Top 10 Important Features")
    st.pyplot(fig)
    plt.clf()


# ---- TAB 2: MANUAL ENTRY ----
with tab2:
    st.subheader("\U0001F4DD Manual Entry")
    st.info("\u2139\ufe0f Fill out details below. Hemoglobin: 13.8–17.2 (men), 12.1–15.1 (women)")
    manual_input = {}

    for feat in features:
        # Rename labels for better UX
        label_name = feat
        if feat == "DESCRIPTION":
            label_name = "Primary Condition (Diagnosis/Procedure)"
        elif feat == "REASONDESCRIPTION":
            label_name = "Reason for Admission (Clinical Complaint)"

        # Handle categorical
        if data[feat].dtype == 'object':
            manual_input[feat] = st.selectbox(
                f"Select {label_name}",
                options=sorted(data[feat].dropna().unique()),
                index=None,
                key=f"{feat}_input"  # Prevent duplicate keys
            )
        # Handle numeric
        else:
            val = st.number_input(
                f"Enter {label_name}",
                min_value=0.0,
                value=0.0,
                step=0.1,
                key=f"{feat}_input",
                format="%.2f"
            )
            manual_input[feat] = None if val == 0.0 else val

            # Hemoglobin check
            if feat == 'Hemoglobin' and val != 0.0:
                if val < 12:
                    st.warning("\U0001F9B8 Hemoglobin is below normal range!")
                elif val > 17.5:
                    st.warning("\U0001F9B8 Hemoglobin is above normal range!")

    if st.button("\U0001F9E0 Predict Readmission Risk", key="manual_predict"):
        input_df = pd.DataFrame([manual_input])
        X_transformed = preprocessor.transform(input_df)
        prob = model.predict_proba(X_transformed)[0][1]
        label = "High Risk" if prob >= 0.30 else "Low Risk"
        st.markdown(f""" 
        <div style='background-color:#fff3cd; padding:15px; border-radius:10px;'>
            <strong>🔮 Predicted Probability:</strong> <span style='color:#0d6efd; font-size:18px;'>{prob:.2f}</span><br>
            {'<span style="color:red; font-weight:bold;">⚠️ High Risk — Consider early intervention.</span>' if label == 'High Risk' else '<span style="color:green; font-weight:bold;">✅ Low Risk — Patient likely stable.</span>'}
        </div>
        """, unsafe_allow_html=True)

# ---- TAB 3: INSIGHTS ----
with tab3:
    st.subheader("\U0001F4CA Dataset Insights")
    total_patients = len(data)
    readmit_rate = data['READMISSION_30'].mean() * 100
    avg_los = data['LOS'].mean()
    col1, col2, col3 = st.columns(3)
    col1.metric("\U0001F465 Total Patients", f"{total_patients}")
    col2.metric("\U0001F4C9 Readmission Rate", f"{readmit_rate:.2f}%")
    col3.metric("\U0001F6CC Avg LOS", f"{avg_los:.1f} days")
    style_metric_cards()

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Readmission by Age Group**")
        data['Age_Group'] = pd.cut(data['AGE'], bins=[0, 30, 45, 60, 75, 90], labels=['0-30', '31-45', '46-60', '61-75', '76-90'])
        age_chart = data.groupby('Age_Group', observed=True)['READMISSION_30'].mean().reset_index()
        sns.barplot(data=age_chart, x='Age_Group', y='READMISSION_30', palette='Set2')
        st.pyplot(plt.gcf())
        plt.clf()
    with col2:
        st.markdown("**LOS vs Readmission**")
        sns.boxplot(data=data, x='READMISSION_30', y='LOS', palette='Set3')
        st.pyplot(plt.gcf())
        plt.clf()

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**Readmission by Gender**")
        gender_chart = data.groupby('GENDER')['READMISSION_30'].mean().reset_index()
        sns.barplot(data=gender_chart, x='GENDER', y='READMISSION_30', palette='pastel')
        st.pyplot(plt.gcf())
        plt.clf()
    with col4:
        st.markdown("**Readmission by Race**")
        race_chart = data.groupby('RACE')['READMISSION_30'].mean().reset_index()
        sns.barplot(data=race_chart, x='RACE', y='READMISSION_30', palette='coolwarm')
        st.pyplot(plt.gcf())
        plt.clf()

    st.markdown("---")
    st.markdown("**Readmission by Top Conditions**")
    top_conditions = data['DESCRIPTION'].value_counts().nlargest(10).index.tolist()
    condition_chart = data[data['DESCRIPTION'].isin(top_conditions)]
    cond_rate = condition_chart.groupby('DESCRIPTION')['READMISSION_30'].mean().reset_index()
    sns.barplot(data=cond_rate, y='DESCRIPTION', x='READMISSION_30', palette='viridis')
    st.pyplot(plt.gcf())
    plt.clf()

    st.markdown("**Hemoglobin Distribution by Readmission**")
    sns.violinplot(data=data, x='READMISSION_30', y='Hemoglobin', palette='Set3')
    st.pyplot(plt.gcf())
    plt.clf()





