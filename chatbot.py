import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Load mock groundwater dataset (adjusted to repo layout)
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "groundwater_data.csv")
data = pd.read_csv(DATA_PATH)

st.title("AI-Driven Groundwater Chatbot (Prototype)")

query = st.text_input("Ask about groundwater (e.g. Groundwater status of Pune in 2022)")

if query:
    words = query.split()
    if "status" in query.lower():
        try:
            district = words[-3]   # crude text extraction
            year = int(words[-1])  # last word assumed to be year

            result = data[(data["District"].str.lower() == district.lower()) & (data["Year"] == year)]

            if not result.empty:
                st.write("### Results")
                st.dataframe(result)

                # Show category + stage
                category = result["Category"].values[0]
                stage = result["Stage"].values[0]
                st.success(f"Groundwater Category: {category} (Stage: {stage}%)")

                if stage > 100:
                    st.warning("⚠️ Over-Exploited! Immediate action needed.")
                elif stage >= 90:
                    st.warning("⚠️ Semi-Critical zone, monitor usage.")
                else:
                    st.info("✅ Safe zone, sustainable extraction.")

                # Chart
                fig, ax = plt.subplots()
                recharge = int(result["Recharge"].values[0])
                extraction = int(result["Extraction"].values[0])
                ax.bar(["Recharge", "Extraction"], [recharge, extraction])
                ax.set_title(f"{district} {year} — Recharge vs Extraction")
                ax.set_ylabel("Volume")
                st.pyplot(fig)

                # Stage history line chart for the same district
                history = data[data["District"].str.lower() == district.lower()].copy()
                if not history.empty:
                    history["Year"] = history["Year"].astype(int)
                    history = history.sort_values("Year")
                    fig2, ax2 = plt.subplots()
                    ax2.plot(history["Year"], history["Stage"], marker="o")
                    ax2.set_title(f"{district} — Stage % Trend Over Years")
                    ax2.set_xlabel("Year")
                    ax2.set_ylabel("Stage (%)")
                    ax2.set_ylim(0, max(110, history["Stage"].max() + 10))
                    st.pyplot(fig2)
            else:
                st.error("No data found for that district/year.")
        except Exception as e:
            st.error(f"Could not process query: {e}")
