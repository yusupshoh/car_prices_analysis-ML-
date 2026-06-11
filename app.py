import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

st.set_page_config(
    page_title="Avtomobil Auksion Tahlili",
    layout="wide",
    initial_sidebar_state="expanded",
)

sns.set_theme(style="whitegrid", palette="viridis")
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.titleweight"] = "bold"

DATA_PATH = "car_prices.csv"


@st.cache_data(show_spinner="Ma'lumotlar yuklanmoqda...")
def load_data(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, on_bad_lines='skip')
        df = df.drop_duplicates()
        df.columns = df.columns.str.lower()
        numeric_cols = ["year", "condition", "odometer", "mmr", "sellingprice"]
        for col in numeric_cols:
            if col in df.columns:df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=numeric_cols)
        text_cols = ["make", "model", "trim", "body", "transmission", "color", "state"]
        for col in text_cols:
            if col in df.columns:df[col] = df[col].astype(str).str.strip().str.title()
        df = df[(df["sellingprice"] > 100) & (df["odometer"] > 0) & (df["year"] >= 1990)]
        df = df.reset_index(drop=True)
        return df
    except Exception as exc:st.error(f"Faylni o'qishda xatolik: {exc}");return pd.DataFrame()


def page_overview(df: pd.DataFrame) -> None:
    st.header("Umumiy Statistika")
    st.markdown("Dataset haqida umumiy ma'lumotlar va asosiy ko'rsatkichlar.")
    try:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Jami avtomobillar soni", f"{df.shape[0]:,} ta")
        col2.metric("Noyob brendlar", f"{df['make'].nunique()} ta")
        col3.metric("O'rtacha sotilish narxi", f"${df['sellingprice'].mean():,.0f}")
        col4.metric("O'rtacha MMR (Bozor bahosi)", f"${df['mmr'].mean():,.0f}")
        st.divider()
        st.subheader("Top-10 eng ko'p sotilgan brendlar statistikasi")
        st.markdown("Bozorda eng ko'p aylanayotgan brendlarning o'rtacha ko'rsatkichlari:")
        top_10_makes_list = df["make"].value_counts().head(10).index
        top_10_df = df[df["make"].isin(top_10_makes_list)]
        brand_table = (top_10_df.groupby("make")
            .agg(
                Sotuvlar_Soni=("sellingprice", "count"),
                Ortacha_Narx=("sellingprice", "mean"),
                Ortacha_Probeg=("odometer", "mean"),
                Ortacha_Holat=("condition", "mean")
            ).sort_values(by="Sotuvlar_Soni", ascending=False))
        brand_table["Sotuvlar_Soni"] = brand_table["Sotuvlar_Soni"].map("{:,} ta".format)
        brand_table["Ortacha_Narx"] = brand_table["Ortacha_Narx"].map("${:,.0f}".format)
        brand_table["Ortacha_Probeg"] = brand_table["Ortacha_Probeg"].map("{:,.0f} mil".format)
        brand_table["Ortacha_Holat"] = brand_table["Ortacha_Holat"].map("{:.1f}".format)
        brand_table.columns = ["Sotuvlar soni", "O'rtacha narxi", "O'rtacha probegi", "O'rtacha holati (Condition)"]
        st.dataframe(brand_table, use_container_width=True)
        st.divider()
        st.subheader("Vizual tahlil va grafiklar")
        graph_col1, graph_col2 = st.columns(2)
        with graph_col1:
            st.markdown("#### Top-10 brendlar (Grafik ko'rinishida)")
            top_makes = df["make"].value_counts().head(10)
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            sns.barplot(x=top_makes.values, y=top_makes.index, hue=top_makes.index, palette="viridis", legend=False, ax=ax1)
            ax1.set_xlabel("Sotuvlar soni")
            ax1.set_ylabel("Brend")
            for i, v in enumerate(top_makes.values): ax1.text(v, i, f" {v:,}", va="center", fontsize=10, weight='bold')
            st.pyplot(fig1)
            plt.close(fig1)
        with graph_col2:
            st.markdown("#### Sotilish narxlari taqsimoti (Distribution)")
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            max_limit = df["sellingprice"].quantile(0.99)
            sns.histplot(df["sellingprice"], bins=50, kde=True, color="#2c3e50", ax=ax2)
            ax2.set_xlabel("Narx ($)")
            ax2.set_ylabel("Avtomobillar soni")
            ax2.set_xlim(0, max_limit)
            st.pyplot(fig2)
            plt.close(fig2)
        st.divider()
        st.subheader("Yillar kesimida dinamik sotuv hajmi")
        st.markdown("Quyidagi ko'p tarmoqli filtrlar yordamida ma'lumotlarni o'zingizga moslab boshqaring:")
        f_col1, f_col2, f_col3 = st.columns(3)
        with f_col1:
            min_year, max_year = int(df["year"].min()), int(df["year"].max())
            selected_years = st.slider("Yillar oralig'i:", min_value=min_year, max_value=max_year, value=(int(df["year"].quantile(0.15)), max_year), step=1)
        with f_col2:
            all_makes = sorted(df["make"].dropna().unique())
            default_makes = [m for m in top_10_makes_list[:3] if m in all_makes]
            selected_makes = st.multiselect("Brendlarni tanlang:", options=all_makes, default=default_makes)
        with f_col3:
            all_bodies = sorted(df["body"].dropna().unique())
            common_bodies = [b for b in ["Sedan", "Suv", "Coupe", "Convertible"] if b in all_bodies]
            selected_bodies = st.multiselect("Kuzov turini tanlang:", options=all_bodies, default=common_bodies)
        filtered_df = df[
            (df["year"] >= selected_years[0]) & (df["year"] <= selected_years[1]) &
            (df["make"].isin(selected_makes)) &
            (df["body"].isin(selected_bodies))]
        if not filtered_df.empty:
            plot_data = pd.crosstab(filtered_df["year"], filtered_df["make"]).sort_index()
            fig3, ax3 = plt.subplots(figsize=(15, 6))
            plot_data.plot(kind="bar", stacked=False, ax=ax3, width=0.8)
            ax3.set_title(f"Filtrlangan avtomobillarning yillik sotuv ko'rsatkichlari ({selected_years[0]} - {selected_years[1]})",fontsize=13, pad=12)
            ax3.set_xlabel("Ishlab chiqarilgan yili (Year)", fontsize=11)
            ax3.set_ylabel("Sotilgan avtomobillar soni (Ta)", fontsize=11)
            ax3.set_xticklabels(ax3.get_xticklabels(), rotation=0)
            ax3.legend(title="Brendlar", bbox_to_anchor=(1.01, 1), loc="upper left")
            plt.tight_layout()
            st.pyplot(fig3)
            plt.close(fig3)
        else:st.warning("Belgilangan kombinatsiya bo'yicha ma'lumot topilmadi. Filtr parametrlarini o'zgartirib ko'ring.")
    except Exception as exc:st.error(f"Statistikani shakllantirishda xatolik: {exc}")


def page_ml_prediction(df: pd.DataFrame) -> None:
    st.header("Narx Bashorati va Aqlli Maslahatchi")
    tab1, tab2, tab3 = st.tabs([
        "ML Narx Bashorati",
        "Mening Budjetimga Nimalar Keladi?",
        "Shtatlar aro"
    ])
    with tab1:
        st.markdown("Scikit-learn yordamida avtomobil texnik xususiyatlariga qarab uning narxini bashorat qiling.")
        try:
            model_df = df[["year", "condition", "odometer", "mmr", "sellingprice"]].dropna()
            X = model_df[["year", "condition", "odometer", "mmr"]]
            y = model_df["sellingprice"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            st.success(f"Model muvaffaqiyatli o'qitildi! Model aniqligi (R2 Score): {r2:.2%}")
            st.subheader("Yangi avtomobil qiymatlarini kiriting:")
            c1, c2, c3, c4 = st.columns(4)
            with c1:input_year = st.number_input("Ishlab chiqarilgan yili:", min_value=1990, max_value=2026, value=2015)
            with c2:input_cond = st.slider("Texnik holati (1-50):", min_value=1.0, max_value=50.0, value=35.0, step=1.0)
            with c3:input_odometer = st.number_input("Probegi (Mil):", min_value=0, value=50000, step=1000)
            with c4:input_mmr = st.number_input("Bozor bahosi (MMR $):", min_value=100, value=15000, step=500)
            if st.button("Narxni Bashorat Qilish"):
                input_data = pd.DataFrame([[input_year, input_cond, input_odometer, input_mmr]],
                                          columns=["year", "condition", "odometer", "mmr"])
                predicted_price = model.predict(input_data)[0]
                if predicted_price < 0: predicted_price = 100
                st.metric("Tavsiya etilgan sotuv narxi:", f"${predicted_price:,.2f}")
        except Exception as exc:st.error(f"Modelni yuklash yoki ishlatishda xatolik: {exc}")
    with tab2:
        st.markdown("Hamyoningizdagi pul miqdorini kiriting va unga mos keladigan eng ommabop avtomobillar tahlilini ko'ring.")
        budget = st.number_input("Sizning budjetingiz ($):", min_value=500, max_value=200000, value=15000, step=500)
        min_budget = budget * 0.9
        max_budget = budget * 1.1
        budget_df = df[(df["sellingprice"] >= min_budget) & (df["sellingprice"] <= max_budget)]
        if not budget_df.empty:
            st.info( f"Sizning budjetingiz atrofida (${min_budget:,.0f} - ${max_budget:,.0f}) jami {budget_df.shape[0]:,} ta savdo topildi.")
            top_budget_makes = budget_df["make"].value_counts().head(5).index
            filtered_budget_df = budget_df[budget_df["make"].isin(top_budget_makes)]
            analysis_table = (
                filtered_budget_df.groupby("make")
                .agg(
                    Ortacha_Narx=("sellingprice", "mean"),
                    Ortacha_Probeg=("odometer", "mean"),
                    Ortacha_Holat=("condition", "mean"),
                    Sotuvlar_Soni=("sellingprice", "count")
                ).reset_index())
            b_col1, b_col2 = st.columns(2)
            with b_col1:
                st.subheader("Budjetingizga mos top-5 brend ko'rsatkichlari")
                display_table = analysis_table.copy()
                display_table["Ortacha_Narx"] = display_table["Ortacha_Narx"].map("${:,.0f}".format)
                display_table["Ortacha_Probeg"] = display_table["Ortacha_Probeg"].map("{:,.0f} mil".format)
                display_table["Ortacha_Holat"] = display_table["Ortacha_Holat"].map("{:.1f}".format)
                display_table.columns = ["Brend", "O'rtacha Narxi", "O'rtacha Probegi", "O'rtacha Holati", "Savdolar Soni"]
                st.dataframe(display_table, use_container_width=True, hide_index=True)
            with b_col2:
                st.subheader("O'rtacha narxlar taqsimoti (Brendlar bo'yicha)")
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(data=analysis_table, y="make", x="Ortacha_Narx", hue="make", palette="viridis", legend=False, ax=ax)
                ax.set_xlabel("O'rtacha sotilish narxi ($)")
                ax.set_ylabel("Brend")
                st.pyplot(fig)
                plt.close(fig)
        else:st.warning("Bu budjet atrofida ma'lumot topilmadi. Pul miqdorini o'zgartirib ko'ring.")
    with tab3:
        st.markdown("Avtomobil rusumlarining AQSh shtatlari bo'yicha narx farqlari va eng arzon hududlar tahlili.")
        try:
            geo_df = df[["state", "make", "sellingprice"]].dropna().copy()
            geo_df["state"] = geo_df["state"].astype(str).str.upper()
            popular_makes = geo_df["make"].value_counts().head(10).index.tolist()
            selected_make = st.selectbox("Taqqoslash uchun avtomobil brendini tanlang:", popular_makes)
            make_df = geo_df[geo_df["make"] == selected_make]
            make_state_prices = make_df.groupby("state")["sellingprice"].agg(["mean", "count"]).reset_index()
            make_state_prices = make_state_prices[make_state_prices["count"] >= 30].sort_values(by="mean")
            if len(make_state_prices) >= 2:
                cheapest_state = make_state_prices.iloc[0]
                expensive_state = make_state_prices.iloc[-1]
                price_diff = expensive_state["mean"] - cheapest_state["mean"]
                st.warning(f"**Biznes Tahlil:** **{selected_make}** rusumli avtomobillar hozirda eng arzon **{cheapest_state['state']}** shtatida sotilmoqda ")
                st.subheader(f"{selected_make} brendining shtatlar bo'yicha narx ko'rinishi")
                fig_geo, ax_geo = plt.subplots(figsize=(15, 6))
                sns.barplot(data=make_state_prices,x="state",y="mean",hue="state",palette="coolwarm",legend=False,ax=ax_geo)
                ax_geo.set_xlabel("Shtat (State)")
                ax_geo.set_ylabel("O'rtacha Sotilish Narxi ($)")
                plt.xticks(rotation=45)
                st.pyplot(fig_geo)
                plt.close(fig_geo)
            else:st.info("Ushbu brend bo'yicha hududiy taqqoslash o'tkazish uchun yetarli geografik ma'lumot mavjud emas.")
        except Exception as exc:
            st.error(f"Geografik tahlil qismida xatolik: {exc}")


def page_condition_analysis(df: pd.DataFrame) -> None:
    st.header("Texnik Holat va Narx Tahlili")
    st.markdown("Avtomobilning texnik holati, bosib o'tgan masofasi va ishlab chiqarilgan yilining yakuniy narxga ta'siri.")
    try:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Texnik holat va Narx o'rtasidagi bog'liqlik")
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            sample_df = df.sample(n=min(3000, len(df)), random_state=42)
            sns.regplot(data=sample_df, x="condition", y="sellingprice", scatter_kws={"alpha": 0.3}, line_kws={"color": "red"}, ax=ax1)
            ax1.set_xlabel("Texnik holat")
            ax1.set_ylabel("Sotilish narxi")
            st.pyplot(fig1)
            plt.close(fig1)
        with col2:
            st.subheader("Probeg va Narx o'rtasidagi bog'liqlik")
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            sns.scatterplot(data=sample_df, x="odometer", y="sellingprice", alpha=0.4, color="#2c3e50", ax=ax2)
            ax2.set_xlabel("Bosib o'tilgan masofa (Odometer / Mil)")
            ax2.set_ylabel("Sotilish narxi ($)")
            st.pyplot(fig2)
            plt.close(fig2)
        st.divider()
        st.subheader("Yillar davomida o'rtacha narx va probeg tendensiyasi")
        trend_df = df.groupby("year").agg({"sellingprice": "mean", "odometer": "mean"}).reset_index()
        fig3, ax3_1 = plt.subplots(figsize=(15, 6))
        ax3_2 = ax3_1.twinx()
        sns.lineplot(data=trend_df, x="year", y="sellingprice", color="#1f77b4", linewidth=2.5, label="O'rtacha narx", ax=ax3_1)
        sns.lineplot(data=trend_df, x="year", y="odometer", color="#2ca02c", linewidth=2.5, label="O'rtacha probeg", ax=ax3_2)
        ax3_1.set_xlabel("Ishlab chiqarilgan yili")
        ax3_1.set_ylabel("O'rtacha sotilish narxi ($)", color="#1f77b4")
        ax3_2.set_ylabel("O'rtacha bosib o'tilgan masofa (Mil)", color="#2ca02c")
        ax3_1.tick_params(axis='y', labelcolor="#1f77b4")
        ax3_2.tick_params(axis='y', labelcolor="#2ca02c")
        ax3_1.get_legend().remove()
        ax3_2.get_legend().remove()
        lines1, labels1 = ax3_1.get_images_and_labels() if hasattr(ax3_1,'get_images_and_labels') else ax3_1.get_legend_handles_labels()
        lines2, labels2 = ax3_2.get_legend_handles_labels()
        ax3_1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
        st.pyplot(fig3)
        plt.close(fig3)
    except Exception as exc:st.error(f"Texnik holatni tahlil qilishda xatolik: {exc}")


def main() -> None:
    st.title("Avtomobil Auksion Narxlari Tahlili")
    df = load_data(DATA_PATH)
    if df.empty:st.stop()
    st.sidebar.title("Navigatsiya")
    st.sidebar.markdown("Kerakli bo'limni tanlang:")
    page = st.sidebar.radio(
        "Bo'limlar:",
        ["Umumiy Statistika", "Texnik Holat va Narx", "Narx Bashorati (ML)"])
    st.sidebar.divider()
    st.sidebar.info(f"Datasetda jami: {df.shape[0]:,} ta faol qator mavjud.")
    if page == "Umumiy Statistika":page_overview(df)
    elif page == "Narx Bashorati (ML)":page_ml_prediction(df)
    elif page == "Texnik Holat va Narx":page_condition_analysis(df)


if __name__ == "__main__":
    main()