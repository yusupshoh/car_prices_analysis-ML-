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
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=numeric_cols)
        text_cols = ["make", "model", "trim", "body", "transmission", "color", "state"]
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.title()
        df = df[(df["sellingprice"] > 100) & (df["odometer"] > 0) & (df["year"] >= 1990)]
        df = df.reset_index(drop=True)
        return df
    except Exception as exc:
        st.error(f"Faylni o'qishda xatolik: {exc}")
        return pd.DataFrame()


@st.cache_data(show_spinner="Flip ma'lumotlari tayyorlanmoqda...")
def compute_flips(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, on_bad_lines='skip')
        df.columns = df.columns.str.lower()
        df['saledate'] = pd.to_datetime(df['saledate'], errors='coerce', utc=True)
        df['sellingprice'] = pd.to_numeric(df['sellingprice'], errors='coerce')
        df['odometer'] = pd.to_numeric(df['odometer'], errors='coerce')
        df['make'] = df['make'].astype(str).str.strip().str.title()

        resold = df[df.duplicated(subset='vin', keep=False)].copy()
        resold = resold.sort_values(['vin', 'saledate'])
        resold['price_change'] = resold.groupby('vin')['sellingprice'].diff()
        resold['days_held'] = resold.groupby('vin')['saledate'].diff().dt.days
        resold['odo_change'] = resold.groupby('vin')['odometer'].diff()

        flips = resold.dropna(subset=['price_change']).copy()
        return flips
    except Exception as exc:
        st.error(f"Flip ma'lumotlarini hisoblashda xatolik: {exc}")
        return pd.DataFrame()


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
        brand_table = (
            top_10_df.groupby("make")
            .agg(
                Sotuvlar_Soni=("sellingprice", "count"),
                Ortacha_Narx=("sellingprice", "mean"),
                Ortacha_Probeg=("odometer", "mean"),
                Ortacha_Holat=("condition", "mean"),
            )
            .sort_values(by="Sotuvlar_Soni", ascending=False)
        )
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
            for i, v in enumerate(top_makes.values):
                ax1.text(v, i, f" {v:,}", va="center", fontsize=10, weight='bold')
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
            selected_years = st.slider(
                "Yillar oralig'i:", min_value=min_year, max_value=max_year,
                value=(int(df["year"].quantile(0.15)), max_year), step=1,
            )
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
            (df["body"].isin(selected_bodies))
        ]
        if not filtered_df.empty:
            plot_data = pd.crosstab(filtered_df["year"], filtered_df["make"]).sort_index()
            fig3, ax3 = plt.subplots(figsize=(15, 6))
            plot_data.plot(kind="bar", stacked=False, ax=ax3, width=0.8)
            ax3.set_title(
                f"Filtrlangan avtomobillarning yillik sotuv ko'rsatkichlari ({selected_years[0]} - {selected_years[1]})",
                fontsize=13, pad=12,
            )
            ax3.set_xlabel("Ishlab chiqarilgan yili (Year)", fontsize=11)
            ax3.set_ylabel("Sotilgan avtomobillar soni (Ta)", fontsize=11)
            ax3.set_xticklabels(ax3.get_xticklabels(), rotation=0)
            ax3.legend(title="Brendlar", bbox_to_anchor=(1.01, 1), loc="upper left")
            plt.tight_layout()
            st.pyplot(fig3)
            plt.close(fig3)
        else:
            st.warning("Belgilangan kombinatsiya bo'yicha ma'lumot topilmadi. Filtr parametrlarini o'zgartirib ko'ring.")
    except Exception as exc:
        st.error(f"Statistikani shakllantirishda xatolik: {exc}")


def page_ml_prediction(df: pd.DataFrame) -> None:
    st.header("Narx Bashorati va Aqlli Maslahatchi")
    tab1, tab2, tab3 = st.tabs([
        "ML Narx Bashorati",
        "Mening Budjetimga Nimalar Keladi?",
        "Shtatlar aro",
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
            with c1:
                input_year = st.number_input("Ishlab chiqarilgan yili:", min_value=1990, max_value=2026, value=2015)
            with c2:
                input_cond = st.slider("Texnik holati (1-50):", min_value=1.0, max_value=50.0, value=35.0, step=1.0)
            with c3:
                input_odometer = st.number_input("Probegi (Mil):", min_value=0, value=50000, step=1000)
            with c4:
                input_mmr = st.number_input("Bozor bahosi (MMR $):", min_value=100, value=15000, step=500)
            if st.button("Narxni Bashorat Qilish"):
                input_data = pd.DataFrame(
                    [[input_year, input_cond, input_odometer, input_mmr]],
                    columns=["year", "condition", "odometer", "mmr"],
                )
                predicted_price = model.predict(input_data)[0]
                if predicted_price < 0:
                    predicted_price = 100
                st.metric("Tavsiya etilgan sotuv narxi:", f"${predicted_price:,.2f}")
        except Exception as exc:
            st.error(f"Modelni yuklash yoki ishlatishda xatolik: {exc}")
    with tab2:
        st.markdown("Hamyoningizdagi pul miqdorini kiriting va unga mos keladigan eng ommabop avtomobillar tahlilini ko'ring.")
        budget = st.number_input("Sizning budjetingiz ($):", min_value=500, max_value=200000, value=15000, step=500)
        min_budget = budget * 0.9
        max_budget = budget * 1.1
        budget_df = df[(df["sellingprice"] >= min_budget) & (df["sellingprice"] <= max_budget)]
        if not budget_df.empty:
            st.info(f"Sizning budjetingiz atrofida (${min_budget:,.0f} - ${max_budget:,.0f}) jami {budget_df.shape[0]:,} ta savdo topildi.")
            top_budget_makes = budget_df["make"].value_counts().head(5).index
            filtered_budget_df = budget_df[budget_df["make"].isin(top_budget_makes)]
            analysis_table = (
                filtered_budget_df.groupby("make")
                .agg(
                    Ortacha_Narx=("sellingprice", "mean"),
                    Ortacha_Probeg=("odometer", "mean"),
                    Ortacha_Holat=("condition", "mean"),
                    Sotuvlar_Soni=("sellingprice", "count"),
                )
                .reset_index()
            )
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
        else:
            st.warning("Bu budjet atrofida ma'lumot topilmadi. Pul miqdorini o'zgartirib ko'ring.")
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
                st.warning(
                    f"**Biznes Tahlil:** **{selected_make}** rusumli avtomobillarni eng arzonlari "
                    f"**{cheapest_state['state']}** shtatida sotilmoqda"
                )
                st.subheader(f"{selected_make} brendining shtatlar bo'yicha narx ko'rinishi")
                fig_geo, ax_geo = plt.subplots(figsize=(15, 6))
                sns.barplot(data=make_state_prices, x="state", y="mean", hue="state", palette="coolwarm", legend=False, ax=ax_geo)
                ax_geo.set_xlabel("Shtat (State)")
                ax_geo.set_ylabel("O'rtacha Sotilish Narxi ($)")
                plt.xticks(rotation=45)
                st.pyplot(fig_geo)
                plt.close(fig_geo)
            else:
                st.info("Ushbu brend bo'yicha hududiy taqqoslash o'tkazish uchun yetarli geografik ma'lumot mavjud emas.")
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
        lines1, labels1 = ax3_1.get_legend_handles_labels()
        lines2, labels2 = ax3_2.get_legend_handles_labels()
        ax3_1.get_legend().remove()
        ax3_1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
        st.pyplot(fig3)
        plt.close(fig3)
    except Exception as exc:
        st.error(f"Texnik holatni tahlil qilishda xatolik: {exc}")


def page_flip_analysis(flips: pd.DataFrame) -> None:
    st.header("Flip Tahlili — Qayta Sotilgan Mashinalar")
    st.markdown(
        "Bir xil VIN raqamiga ega bo'lib, ikki marta auksiondan o'tgan avtomobillar tahlili. "
        "Har bir 'flip' — bu bir odam sotib olib, keyinchalik qayta sotgan mashina."
    )

    if flips.empty:
        st.error("Flip ma'lumotlari yuklanmadi.")
        return

    try:
        total_flips = len(flips)
        avg_profit = flips["price_change"].mean()
        median_profit = flips["price_change"].median()
        profitable_pct = (flips["price_change"] > 0).mean() * 100
        loss_pct = (flips["price_change"] < 0).mean() * 100
        avg_days = flips["days_held"].dropna().mean()

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Jami flip soni", f"{total_flips:,} ta")
        m2.metric("O'rtacha foyda", f"${avg_profit:,.0f}")
        m3.metric("Mediana foyda", f"${median_profit:,.0f}")
        m4.metric("Foydali fliplar", f"{profitable_pct:.1f}%")
        m5.metric("O'rtacha kutish", f"{avg_days:.0f} kun")

        st.divider()

        # --- Tab tuzilmasi ---
        tab1, tab2, tab3 = st.tabs([
            "Foyda Taqsimoti",
            "Brend bo'yicha Tahlil",
            "Eng Foydali Fliplar",
        ])

        # ---------- TAB 1: Foyda taqsimoti ----------
        with tab1:
            st.subheader("Foyda/Zarar taqsimoti")
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Foyda guruhlari bo'yicha")
                bins = [-999999, -3000, -1000, 0, 1000, 3000, 5000, 999999]
                labels_b = ["< -$3k", "-$3k — -$1k", "-$1k — $0", "$0 — $1k", "$1k — $3k", "$3k — $5k", "> $5k"]
                flips["profit_bucket"] = pd.cut(flips["price_change"], bins=bins, labels=labels_b)
                bucket_counts = flips["profit_bucket"].value_counts().reindex(labels_b)

                colors_bar = ["#A32D2D", "#D85A30", "#E2906A", "#888780", "#1D9E75", "#0F6E56", "#085041"]
                fig1, ax1 = plt.subplots(figsize=(9, 5))
                bars = ax1.bar(labels_b, bucket_counts.values, color=colors_bar, edgecolor="white", linewidth=0.5)
                for bar, val in zip(bars, bucket_counts.values):
                    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                             f"{int(val):,}", ha="center", va="bottom", fontsize=9, fontweight="bold")
                ax1.set_xlabel("Foyda/Zarar oralig'i")
                ax1.set_ylabel("Flip soni")
                ax1.set_xticklabels(labels_b, rotation=25, ha="right")
                plt.tight_layout()
                st.pyplot(fig1)
                plt.close(fig1)

            with col2:
                st.markdown("#### Umumiy natija")
                profitable = (flips["price_change"] > 50).sum()
                loss = (flips["price_change"] < -50).sum()
                breakeven = total_flips - profitable - loss
                pie_labels = [f"Foydali ({profitable:,})", f"Zararli ({loss:,})", f"Neytral ({breakeven:,})"]
                pie_sizes = [profitable, loss, breakeven]
                pie_colors = ["#1D9E75", "#D85A30", "#888780"]
                fig2, ax2 = plt.subplots(figsize=(7, 5))
                wedges, texts, autotexts = ax2.pie(
                    pie_sizes, labels=pie_labels, colors=pie_colors,
                    autopct="%1.1f%%", startangle=140,
                    wedgeprops={"edgecolor": "white", "linewidth": 1.5},
                )
                for at in autotexts:
                    at.set_fontsize(10)
                    at.set_fontweight("bold")
                plt.tight_layout()
                st.pyplot(fig2)
                plt.close(fig2)

            st.divider()
            st.subheader("Kutish vaqti va foyda o'rtasidagi bog'liqlik")
            sample = flips[flips["days_held"].between(1, 365)].sample(n=min(2000, len(flips)), random_state=42)
            sample["profit_color"] = sample["price_change"].apply(lambda x: "#1D9E75" if x > 0 else "#D85A30")
            fig3, ax3 = plt.subplots(figsize=(14, 5))
            ax3.scatter(sample["days_held"], sample["price_change"], c=sample["profit_color"], alpha=0.4, s=15)
            ax3.axhline(0, color="black", linewidth=0.8, linestyle="--")
            ax3.set_xlabel("Ushlab turilgan kunlar soni")
            ax3.set_ylabel("Narx o'zgarishi ($)")
            ax3.set_xlim(0, 365)
            plt.tight_layout()
            st.pyplot(fig3)
            plt.close(fig3)

        # ---------- TAB 2: Brend bo'yicha ----------
        with tab2:
            st.subheader("Brendlar bo'yicha flip statistikasi")
            min_flips = st.slider("Minimal flip soni (filtr):", min_value=10, max_value=200, value=50, step=10)

            brand_stats = (
                flips.groupby("make")
                .agg(
                    Flip_Soni=("price_change", "count"),
                    Ortacha_Foyda=("price_change", "mean"),
                    Mediana_Foyda=("price_change", "median"),
                    Foydali_Pct=("price_change", lambda x: (x > 0).mean() * 100),
                    Ortacha_Kun=("days_held", "mean"),
                )
                .reset_index()
                .query(f"Flip_Soni >= {min_flips}")
                .sort_values("Flip_Soni", ascending=False)
            )

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Flip soni bo'yicha top brendlar")
                fig4, ax4 = plt.subplots(figsize=(9, max(5, len(brand_stats) * 0.45)))
                colors_make = ["#1D9E75" if p > 0 else "#D85A30" for p in brand_stats["Ortacha_Foyda"]]
                bars4 = ax4.barh(brand_stats["make"], brand_stats["Flip_Soni"], color=colors_make)
                for bar, val in zip(bars4, brand_stats["Flip_Soni"]):
                    ax4.text(val + 5, bar.get_y() + bar.get_height() / 2,
                             f"{int(val):,}", va="center", fontsize=9)
                ax4.set_xlabel("Flip soni")
                ax4.invert_yaxis()
                plt.tight_layout()
                st.pyplot(fig4)
                plt.close(fig4)

            with col2:
                st.markdown("#### O'rtacha foyda bo'yicha brendlar")
                brand_profit = brand_stats.sort_values("Ortacha_Foyda", ascending=True)
                colors_profit = ["#1D9E75" if p > 0 else "#D85A30" for p in brand_profit["Ortacha_Foyda"]]
                fig5, ax5 = plt.subplots(figsize=(9, max(5, len(brand_profit) * 0.45)))
                ax5.barh(brand_profit["make"], brand_profit["Ortacha_Foyda"], color=colors_profit)
                ax5.axvline(0, color="black", linewidth=0.8, linestyle="--")
                ax5.set_xlabel("O'rtacha foyda ($)")
                plt.tight_layout()
                st.pyplot(fig5)
                plt.close(fig5)

            st.divider()
            st.subheader("Jadval ko'rinishida")
            display_stats = brand_stats.copy()
            display_stats["Ortacha_Foyda"] = display_stats["Ortacha_Foyda"].map("${:,.0f}".format)
            display_stats["Mediana_Foyda"] = display_stats["Mediana_Foyda"].map("${:,.0f}".format)
            display_stats["Foydali_Pct"] = display_stats["Foydali_Pct"].map("{:.1f}%".format)
            display_stats["Ortacha_Kun"] = display_stats["Ortacha_Kun"].map("{:.0f} kun".format)
            display_stats["Flip_Soni"] = display_stats["Flip_Soni"].map("{:,} ta".format)
            display_stats.columns = [
                "Brend", "Flip soni", "O'rtacha foyda", "Mediana foyda",
                "Foydali fliplar %", "O'rtacha kutish",
            ]
            st.dataframe(display_stats, use_container_width=True, hide_index=True)

        # ---------- TAB 3: Eng foydali fliplar ----------
        with tab3:
            st.subheader("Eng yuqori foydali individual fliplar")
            st.markdown("Bitta sotuv davomida eng ko'p foyda qilingan mashinalar:")
            top_n = st.slider("Nechta ko'rsatilsin:", min_value=5, max_value=50, value=20)
            top_flips = (
                flips[["vin", "make", "model", "year", "sellingprice", "price_change", "days_held", "odometer"]]
                .dropna()
                .sort_values("price_change", ascending=False)
                .head(top_n)
                .copy()
            )
            top_flips["price_change"] = top_flips["price_change"].map("${:,.0f}".format)
            top_flips["sellingprice"] = top_flips["sellingprice"].map("${:,.0f}".format)
            top_flips["odometer"] = top_flips["odometer"].map("{:,.0f} mil".format)
            top_flips["days_held"] = top_flips["days_held"].map("{:.0f} kun".format)
            top_flips["year"] = top_flips["year"].astype(int)
            top_flips.columns = ["VIN", "Brend", "Model", "Yil", "Sotilish narxi", "Foyda", "Kutilgan vaqt", "Probeg"]
            st.dataframe(top_flips, use_container_width=True, hide_index=True)

            st.divider()
            st.subheader("Eng katta zararli fliplar")
            worst_flips = (
                flips[["vin", "make", "model", "year", "sellingprice", "price_change", "days_held"]]
                .dropna()
                .sort_values("price_change", ascending=True)
                .head(top_n)
                .copy()
            )
            worst_flips["price_change"] = worst_flips["price_change"].map("${:,.0f}".format)
            worst_flips["sellingprice"] = worst_flips["sellingprice"].map("${:,.0f}".format)
            worst_flips["days_held"] = worst_flips["days_held"].map("{:.0f} kun".format)
            worst_flips["year"] = worst_flips["year"].astype(int)
            worst_flips.columns = ["VIN", "Brend", "Model", "Yil", "Sotilish narxi", "Zarar", "Kutilgan vaqt"]
            st.dataframe(worst_flips, use_container_width=True, hide_index=True)

    except Exception as exc:
        st.error(f"Flip tahlilida xatolik: {exc}")


def main() -> None:
    st.title("Avtomobil Auksion Narxlari Tahlili")
    df = load_data(DATA_PATH)
    if df.empty:
        st.stop()

    st.sidebar.title("Navigatsiya")
    st.sidebar.markdown("Kerakli bo'limni tanlang:")
    page = st.sidebar.radio(
        "Bo'limlar:",
        [
            "Umumiy Statistika",
            "Texnik Holat va Narx",
            "Narx Bashorati (ML)",
            "Flip Tahlili (VIN)",
        ],
    )
    st.sidebar.divider()
    st.sidebar.info(f"Datasetda jami: {df.shape[0]:,} ta faol qator mavjud.")

    if page == "Umumiy Statistika":
        page_overview(df)
    elif page == "Narx Bashorati (ML)":
        page_ml_prediction(df)
    elif page == "Texnik Holat va Narx":
        page_condition_analysis(df)
    elif page == "Flip Tahlili (VIN)":
        flips = compute_flips(DATA_PATH)
        page_flip_analysis(flips)


if __name__ == "__main__":
    main()