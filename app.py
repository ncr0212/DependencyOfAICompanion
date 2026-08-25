import os
import platform
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------
# 1. 기본 페이지 설정 및 한글 폰트 지정
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI 동반자 의존도 예측 & 분석 시스템",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

import matplotlib.font_manager as fm

def configure_korean_font():
    system_name = platform.system()
    if system_name == 'Darwin':
        font_name = 'AppleGothic'
    elif system_name == 'Windows':
        font_name = 'Malgun Gothic'
    else:
        font_name = 'NanumGothic'

    available_fonts = set([f.name for f in fm.fontManager.ttflist])
    if font_name not in available_fonts:
        candidates = ['Malgun Gothic', 'NanumGothic', 'Dotum', 'Gulim', 'Batang', 'AppleGothic']
        for cand in candidates:
            if cand in available_fonts:
                font_name = cand
                break

    # Seaborn & Matplotlib 글로벌 폰트 설정
    sns.set_theme(style="whitegrid", font=font_name)
    plt.rc('font', family=font_name)
    plt.rcParams['font.family'] = font_name
    plt.rcParams['font.sans-serif'] = [font_name, 'Malgun Gothic', 'NanumGothic', 'AppleGothic', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    return font_name

KOREAN_FONT = configure_korean_font()

def apply_axis_font(ax, title="", xlabel="", ylabel=""):
    configure_korean_font()
    if title:
        ax.set_title(title, fontsize=12, pad=12, fontfamily=KOREAN_FONT)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10, fontfamily=KOREAN_FONT)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10, fontfamily=KOREAN_FONT)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily(KOREAN_FONT)
    if ax.get_legend():
        plt.setp(ax.get_legend().get_texts(), fontfamily=KOREAN_FONT)
        if ax.get_legend().get_title():
            ax.get_legend().get_title().set_fontfamily(KOREAN_FONT)

# Custom CSS (깔끔한 밝은 화이트 테마)
st.markdown("""
<style>
    .main {
        background-color: #ffffff;
        color: #0f172a;
    }
    .stApp {
        background-color: #f8fafc;
    }
    .stMetric {
        background: #ffffff !important;
        border-radius: 12px;
        padding: 15px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    h1, h2, h3, h4 {
        color: #0f172a !important;
        font-weight: 700;
    }
    .stButton>button {
        background: linear-gradient(90deg, #4f46e5 0%, #6366f1 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 24px;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 14px 0 rgba(79, 70, 229, 0.3);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px 0 rgba(79, 70, 229, 0.45);
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# 2. 모델 객체 및 원본 데이터 로드 함수 작성 (@st.cache_resource / @st.cache_data)
# ---------------------------------------------------------
@st.cache_resource
def load_model_artifact(model_path="elasticnet_ai_companion_model.joblib"):
    if not os.path.exists(model_path):
        st.error(f"모델 파일({model_path})을 찾을 수 없습니다.")
        st.stop()
    artifact = joblib.load(model_path)
    model = artifact['model']
    # Prompt Requirement: Safe feature_importances_ attribute setup
    if not hasattr(model, 'feature_importances_'):
        model.feature_importances_ = np.abs(model.coef_)
    return artifact

@st.cache_data
def load_dataset(data_path="data/ai_companion_dependency_dataset.csv"):
    if not os.path.exists(data_path):
        st.error(f"데이터셋 파일({data_path})을 찾을 수 없습니다.")
        st.stop()
    df = pd.read_csv(data_path)
    return df

artifact = load_model_artifact()
df_raw = load_dataset()

encoder = artifact['onehot_encoder']
poly_transformer = artifact['poly_transformer']
scaler = artifact['scaler']
model = artifact['model']
numeric_features = artifact['numeric_features']
categorical_features = artifact['categorical_features']


# ---------------------------------------------------------
# 3. 메인 타이틀 및 탭 구조 설정
# ---------------------------------------------------------
st.title("🤖 AI 동반자 정서적 의존도 예측 & 분석 시스템")
st.caption("ElasticNet 회귀 모델 기반 사용자 맞춤형 의존도 시뮬레이션 및 인사이트 대시보드")

tab_sim, tab_dash = st.tabs(["🎮 AI 의존도 시뮬레이터", "📊 데이터 인사이트 대시보드"])

# ---------------------------------------------------------
# 4. 시뮬레이터 탭: st.columns([2, 1.1]) 컬럼 분할 및 예측
# ---------------------------------------------------------
with tab_sim:
    st.markdown("### 📋 사용자 프로필 및 생활 패턴 입력")
    st.info("좌측 폼(65%)에서 사용자 프로필 수치를 조정하면, 우측(35%)에서 실시간으로 예측 결과가 즉시 갱신됩니다.")

    col_left, col_right = st.columns([2, 1.1])

    with col_left:
        st.subheader("👤 기본 인적사항 & 범주형 특성")
        c1, c2, c3 = st.columns(3)
        
        with c1:
            occ_options = list(encoder.categories_[0])
            user_occ = st.selectbox("직업 상태 (occupation_status)", options=occ_options, index=0)
        with c2:
            rel_options = list(encoder.categories_[1])
            user_rel = st.selectbox("연애/결혼 상태 (relationship_status)", options=rel_options, index=0)
        with c3:
            liv_options = list(encoder.categories_[2])
            user_liv = st.selectbox("주거 형태 (living_situation)", options=liv_options, index=0)

        st.markdown("---")
        st.subheader("⏰ 일상 시간 활용 (시간/일)")
        t1, t2, t3, t4 = st.columns(4)
        with t1:
            user_age = st.slider("나이 (age)", min_value=15, max_value=24, value=20)
        with t2:
            user_ai_chat = st.slider("일간 AI 대화 (daily_ai_chat_hours)", 0.0, 15.0, 3.5, step=0.5)
        with t3:
            user_human_social = st.slider("대인 상호작용 (human_social_interaction_hours)", 0.0, 15.0, 2.0, step=0.5)
        with t4:
            user_social_media = st.slider("SNS 사용 (social_media_hours_daily)", 0.0, 15.0, 4.0, step=0.5)

        s1, s2, s3, s4 = st.columns(4)
        with s1:
            user_screen_time = st.slider("총 스크린 타임 (screen_time_total_hours)", 0.0, 24.0, 8.0, step=0.5)
        with s2:
            user_close_friends = st.slider("친한 친구 수 (number_of_close_friends)", 0, 20, 3)
        with s3:
            user_sleep_hours = st.slider("수면 시간 (sleep_hours)", 0.0, 15.0, 7.0, step=0.5)
        with s4:
            user_exercise = st.slider("주간 운동 시간 (exercise_hours_weekly)", 0.0, 20.0, 2.5, step=0.5)

        st.markdown("---")
        st.subheader("🧠 심리 및 관계 만족도 지표 (1~10 점)")
        m1, m2, m3 = st.columns(3)
        with m1:
            user_rel_sat = st.slider("현실 관계 만족도 (real_relationship_satisfaction)", 1.0, 10.0, 5.0, step=0.5)
            user_sleep_qual = st.slider("수면 품질 점수 (sleep_quality_score)", 1.0, 10.0, 6.0, step=0.5)
        with m2:
            user_loneliness = st.slider("외로움 점수 (loneliness_score)", 1.0, 10.0, 6.5, step=0.5)
            user_anxiety = st.slider("불안 점수 (anxiety_score)", 1.0, 10.0, 5.5, step=0.5)
        with m3:
            user_depression = st.slider("우울 점수 (depression_score)", 1.0, 10.0, 5.0, step=0.5)
            user_stress = st.slider("스트레스 점수 (stress_score)", 1.0, 10.0, 6.0, step=0.5)
            user_self_esteem = st.slider("자존감 점수 (self_esteem_score)", 1.0, 10.0, 5.0, step=0.5)

    with col_right:
        st.subheader("🎯 실시간 예측 결과")
        st.write("좌측 폼의 프로필 입력값을 변경하면 실시간으로 ElasticNet 회귀 모델 예측 결과가 반영됩니다.")
        
        # 입력 데이터를 1행짜리 DataFrame으로 변환
        input_data = {
            'age': user_age,
            'daily_ai_chat_hours': user_ai_chat,
            'human_social_interaction_hours': user_human_social,
            'social_media_hours_daily': user_social_media,
            'screen_time_total_hours': user_screen_time,
            'number_of_close_friends': user_close_friends,
            'real_relationship_satisfaction': user_rel_sat,
            'sleep_hours': user_sleep_hours,
            'sleep_quality_score': user_sleep_qual,
            'exercise_hours_weekly': user_exercise,
            'loneliness_score': user_loneliness,
            'anxiety_score': user_anxiety,
            'depression_score': user_depression,
            'stress_score': user_stress,
            'self_esteem_score': user_self_esteem,
            'occupation_status': user_occ,
            'relationship_status': user_rel,
            'living_situation': user_liv
        }

        input_df = pd.DataFrame([input_data])

        # 입력 데이터 전처리 파이프라인: 원핫 인코딩 -> 수치형과 결합 -> 다항식 변환 -> 스케일링
        encoded_cats = encoder.transform(input_df[categorical_features])
        cat_feature_names = encoder.get_feature_names_out(categorical_features)
        encoded_cat_df = pd.DataFrame(encoded_cats, columns=cat_feature_names, index=input_df.index)

        X_combined = pd.concat([input_df[numeric_features], encoded_cat_df], axis=1)
        X_poly = poly_transformer.transform(X_combined)
        X_scaled = scaler.transform(X_poly)

        # 모델 예측 실행
        pred_value = model.predict(X_scaled)[0]
        clamped_score = max(0.0, min(10.0, float(pred_value)))

        st.markdown("<br>", unsafe_allow_html=True)
        
        # st.metric으로 요약 상태(의존도 수치) 표시
        st.metric(
            label="예측된 AI 정서적 의존도 점수 (Emotional Attachment)",
            value=f"{clamped_score:.2f} / 10.0"
        )

        # 위험도 라벨 및 설명
        if clamped_score < 4.0:
            risk_label = "🟢 낮음 (Low)"
            risk_color = "#10b981"
            risk_desc = "AI 동반자에 대한 정서적 의존도가 낮은 수준입니다."
        elif clamped_score < 7.0:
            risk_label = "🟡 보통 (Moderate)"
            risk_color = "#f59e0b"
            risk_desc = "외로움이나 스트레스 발생 시 AI에 대한 의존성이 높아질 수 있습니다."
        else:
            risk_label = "🔴 높음 (High)"
            risk_color = "#ef4444"
            risk_desc = "AI 동반자에 대한 정서적 의존성이 매우 높습니다. 대인 관계 활동을 권장합니다."

        st.markdown(f"""
        <div style="background-color: #f1f5f9; border-left: 5px solid {risk_color}; padding: 15px; border-radius: 8px; margin-top: 15px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
            <h4 style="margin: 0; color: {risk_color}; font-weight: 700;">{risk_label}</h4>
            <p style="margin: 5px 0 0 0; color: #334155; font-size: 14px; font-weight: 500;">{risk_desc}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### 📌 입력 프로필 요약")
        st.write(f"- **AI 대화 시간**: {user_ai_chat} 시간/일")
        st.write(f"- **외로움 점수**: {user_loneliness} / 10 점")
        st.write(f"- **현실 관계 만족도**: {user_rel_sat} / 10 점")
        st.write(f"- **직업 상태**: {user_occ}")
        st.write(f"- **연애 상태**: {user_rel}")


# ---------------------------------------------------------
# 5. 데이터 인사이트 대시보드 구현 (서브 탭 3개)
# ---------------------------------------------------------
with tab_dash:
    sub_tab1, sub_tab2, sub_tab3 = st.tabs([
        "💡 AI 특성 중요도 분석",
        "🔍 상위 2개 피처 세부 분석",
        "📈 전체 데이터 분포 및 상관관계"
    ])

    cat_cols_encoded = list(encoder.get_feature_names_out(categorical_features))
    all_feature_names = numeric_features + cat_cols_encoded
    poly_feature_names = poly_transformer.get_feature_names_out(all_feature_names)

    # Prompt requirement: model.feature_importances_ 기준 상위 10개 피처
    importances = model.feature_importances_
    
    feat_df = pd.DataFrame({
        'Feature': poly_feature_names,
        'Importance': importances,
        'Raw_Coef': model.coef_
    }).sort_values(by='Importance', ascending=False).reset_index(drop=True)

    # ---------------------------------------------------------
    # 서브 탭 1: AI 특성 중요도 분석 (sns.barplot 상위 10개)
    # ---------------------------------------------------------
    with sub_tab1:
        st.markdown("### 🔝 AI 특성 중요도 분석 (`model.feature_importances_` 기준 상위 10개)")
        st.caption("ElasticNet 모델의 계수 절대값(|Coefficient|) 기준 상위 10개 피처를 시각화합니다.")

        top10_df = feat_df.head(10).copy()
        
        # 가독성을 위한 표기 정제
        top10_df['Clean_Feature'] = top10_df['Feature'].apply(
            lambda x: x.replace(' ', ' × ').replace('^2', '²')
                       .replace('occupation_status_', '직업:')
                       .replace('relationship_status_', '관계:')
                       .replace('living_situation_', '주거:')
        )

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(
            data=top10_df,
            x='Importance',
            y='Clean_Feature',
            palette='mako',
            ax=ax
        )
        apply_axis_font(ax, title='상위 10개 피처 중요도 시각화', xlabel='중요도 (|Coefficient|)', ylabel='피처 이름')
        
        # 값 표시
        for p in ax.patches:
            width = p.get_width()
            ax.annotate(f'{width:.4f}',
                        (width, p.get_y() + p.get_height() / 2.),
                        ha='left', va='center',
                        xytext=(5, 0), textcoords='offset points',
                        fontsize=10, color='black', fontfamily=KOREAN_FONT)

        st.pyplot(fig)
        plt.close(fig)

        st.markdown("#### 📋 상위 10개 피처 상세 데이터")
        st.dataframe(top10_df[['Clean_Feature', 'Importance', 'Raw_Coef']], use_container_width=True)

    # ---------------------------------------------------------
    # 서브 탭 2: 중요도 상위 2개 피처 세부 분석 (sns.boxplot & sns.countplot 나란히 배치)
    # ---------------------------------------------------------
    with sub_tab2:
        st.markdown("### 📊 중요도 상위 2개 피처 세부 분석")
        st.caption("상위 2개 핵심 피처를 기반으로 의존도(emotional_attachment_score)와 위험도를 2개 컬럼(st.columns)으로 나란히 시각화합니다.")

        col1, col2 = st.columns(2)

        # 피처 1 시각화: sns.boxplot (직업 상태별 의존도 점수)
        with col1:
            st.markdown("#### 1️⃣ 피처 1: 직업 상태(occupation_status)별 의존도 (`sns.boxplot`)")
            fig1, ax1 = plt.subplots(figsize=(6, 4.5))
            sns.boxplot(
                data=df_raw,
                x='occupation_status',
                y='emotional_attachment_score',
                palette='Set2',
                ax=ax1
            )
            apply_axis_font(ax1, title='직업 상태에 따른 의존도 시각화 (Boxplot)', xlabel='직업 상태 (occupation_status)', ylabel='의존도 점수 (emotional_attachment_score)')
            plt.xticks(rotation=20)
            st.pyplot(fig1)
            plt.close(fig1)

        # 피처 2 시각화: sns.countplot (의존도 위험등급별 연애 상태)
        with col2:
            st.markdown("#### 2️⃣ 피처 2: 연애 상태(relationship_status) & 위험군 분포 (`sns.countplot`)")
            fig2, ax2 = plt.subplots(figsize=(6, 4.5))
            sns.countplot(
                data=df_raw,
                x='dependency_risk_label',
                hue='relationship_status',
                palette='viridis',
                ax=ax2
            )
            apply_axis_font(ax2, title='의존도 위험군별 연애 상태 시각화 (Countplot)', xlabel='의존도 위험군 (dependency_risk_label)', ylabel='인원 수 (Count)')
            st.pyplot(fig2)
            plt.close(fig2)

    # ---------------------------------------------------------
    # 서브 탭 3: 전체 데이터 분포 및 상관관계
    # ---------------------------------------------------------
    with sub_tab3:
        st.markdown("### 📈 전체 데이터 분포 및 변수 간 상관관계")
        
        c_left, c_right = st.columns(2)
        with c_left:
            st.markdown("#### 외로움 점수 vs AI 의존도 점수 분포 (KDE)")
            fig3, ax3 = plt.subplots(figsize=(6, 4.5))
            sns.kdeplot(df_raw['loneliness_score'], fill=True, label='외로움 점수', color='#8b5cf6', ax=ax3)
            sns.kdeplot(df_raw['emotional_attachment_score'], fill=True, label='AI 의존도 점수', color='#06b6d4', ax=ax3)
            apply_axis_font(ax3, title='외로움 및 의존도 점수 밀도 추정', xlabel='점수', ylabel='밀도')
            ax3.legend()
            st.pyplot(fig3)
            plt.close(fig3)

        with c_right:
            st.markdown("#### 주요 수치형 변수 간 Correlation Heatmap")
            fig4, ax4 = plt.subplots(figsize=(6, 4.5))
            corr_cols = ['emotional_attachment_score', 'daily_ai_chat_hours', 'loneliness_score', 'depression_score', 'stress_score', 'real_relationship_satisfaction']
            sns.heatmap(df_raw[corr_cols].corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=ax4, cbar=False)
            apply_axis_font(ax4, title='주요 변수간 상관계수 히트맵')
            st.pyplot(fig4)
            plt.close(fig4)
