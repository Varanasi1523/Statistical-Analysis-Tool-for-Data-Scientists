import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
import io
import base64

def main():
    st.set_page_config(page_title="Statistical Analysis Tool", layout="wide")
    st.title("Statistical Analysis Tool for Data Scientists")
    
    # File upload
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state['df'] = df
        
        # Sidebar for navigation
        st.sidebar.title("Navigation")
        analysis_type = st.sidebar.radio(
            "Choose Analysis Type",
            ["Data Overview",
             "Descriptive Statistics",
             "Inferential Statistics",
             "Regression Analysis",
             "Time Series Analysis"]
        )
        
        if analysis_type == "Data Overview":
            show_data_overview(df)
        elif analysis_type == "Descriptive Statistics":
            show_descriptive_statistics(df)
        elif analysis_type == "Inferential Statistics":
            show_inferential_statistics(df)
        elif analysis_type == "Regression Analysis":
            show_regression_analysis(df)
        elif analysis_type == "Time Series Analysis":
            show_time_series_analysis(df)

def show_data_overview(df):
    st.header("Data Overview")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("First few rows of the dataset:")
        st.write(df.head())
    
    with col2:
        st.write("Dataset Info:")
        buffer = io.StringIO()
        df.info(buf=buffer)
        st.text(buffer.getvalue())
    
    st.write("Data Types:")
    st.write(pd.DataFrame(df.dtypes, columns=['Data Type']))
    
    st.write("Missing Values:")
    st.write(pd.DataFrame(df.isnull().sum(), columns=['Missing Count']))

def show_descriptive_statistics(df):
    st.header("Descriptive Statistics")
    
    # Select numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numerical_cols) > 0:
        selected_col = st.selectbox("Select column for analysis:", numerical_cols)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Basic Statistics")
            stats_df = pd.DataFrame({
                'Statistic': ['Mean', 'Median', 'Std Dev', 'Variance', 'Minimum', 'Maximum'],
                'Value': [
                    df[selected_col].mean(),
                    df[selected_col].median(),
                    df[selected_col].std(),
                    df[selected_col].var(),
                    df[selected_col].min(),
                    df[selected_col].max()
                ]
            })
            st.write(stats_df)
            
            st.subheader("Quantiles")
            quantiles = pd.DataFrame({
                'Quantile': ['25%', '50%', '75%'],
                'Value': [
                    df[selected_col].quantile(0.25),
                    df[selected_col].quantile(0.50),
                    df[selected_col].quantile(0.75)
                ]
            })
            st.write(quantiles)
        
        with col2:
            st.subheader("Distribution Plot")
            fig = px.histogram(df, x=selected_col, nbins=30, marginal="box")
            st.plotly_chart(fig)
            
            st.subheader("Box Plot")
            fig = px.box(df, y=selected_col)
            st.plotly_chart(fig)
    else:
        st.warning("No numerical columns found in the dataset.")

def show_inferential_statistics(df):
    st.header("Inferential Statistics")
    
    test_type = st.selectbox(
        "Select Statistical Test:",
        ["T-Test (Independent)",
         "T-Test (Paired)",
         "Chi-Square Test",
         "ANOVA",
         "Correlation Analysis"]
    )
    
    if test_type == "T-Test (Independent)":
        perform_independent_ttest(df)
    elif test_type == "T-Test (Paired)":
        perform_paired_ttest(df)
    elif test_type == "Chi-Square Test":
        perform_chi_square_test(df)
    elif test_type == "ANOVA":
        perform_anova(df)
    elif test_type == "Correlation Analysis":
        perform_correlation_analysis(df)

def perform_independent_ttest(df):
    st.subheader("Independent T-Test")
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    if len(numerical_cols) > 0 and len(categorical_cols) > 0:
        num_col = st.selectbox("Select numerical column:", numerical_cols)
        cat_col = st.selectbox("Select categorical column (for grouping):", categorical_cols)
        
        if len(df[cat_col].unique()) == 2:
            groups = df[cat_col].unique()
            group1_data = df[df[cat_col] == groups[0]][num_col]
            group2_data = df[df[cat_col] == groups[1]][num_col]
            
            t_stat, p_value = stats.ttest_ind(group1_data, group2_data)
            
            st.write(f"T-statistic: {t_stat:.4f}")
            st.write(f"P-value: {p_value:.4f}")
            
            # Visualization
            fig = go.Figure()
            fig.add_trace(go.Box(y=group1_data, name=str(groups[0])))
            fig.add_trace(go.Box(y=group2_data, name=str(groups[1])))
            fig.update_layout(title="Distribution Comparison",
                             yaxis_title=num_col)
            st.plotly_chart(fig)
        else:
            st.warning("Please select a categorical column with exactly 2 categories for t-test.")
    else:
        st.warning("Need both numerical and categorical columns for t-test.")

def perform_paired_ttest(df):
    st.subheader("Paired T-Test")
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numerical_cols) >= 2:
        col1 = st.selectbox("Select first variable:", numerical_cols)
        col2 = st.selectbox("Select second variable:", numerical_cols)
        
        if col1 != col2:
            t_stat, p_value = stats.ttest_rel(df[col1], df[col2])
            
            st.write(f"T-statistic: {t_stat:.4f}")
            st.write(f"P-value: {p_value:.4f}")
            
            # Visualization
            fig = go.Figure()
            fig.add_trace(go.Box(y=df[col1], name=col1))
            fig.add_trace(go.Box(y=df[col2], name=col2))
            fig.update_layout(title="Paired Data Comparison")
            st.plotly_chart(fig)
        else:
            st.warning("Please select different columns for comparison.")
    else:
        st.warning("Need at least 2 numerical columns for paired t-test.")

def perform_chi_square_test(df):
    st.subheader("Chi-Square Test of Independence")
    
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    if len(categorical_cols) >= 2:
        col1 = st.selectbox("Select first categorical variable:", categorical_cols)
        col2 = st.selectbox("Select second categorical variable:", categorical_cols)
        
        if col1 != col2:
            contingency_table = pd.crosstab(df[col1], df[col2])
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
            
            st.write("Contingency Table:")
            st.write(contingency_table)
            st.write(f"Chi-square statistic: {chi2:.4f}")
            st.write(f"P-value: {p_value:.4f}")
            st.write(f"Degrees of freedom: {dof}")
            
            # Visualization
            fig = px.imshow(contingency_table,
                           labels=dict(x=col2, y=col1, color="Count"),
                           title="Contingency Table Heatmap")
            st.plotly_chart(fig)
        else:
            st.warning("Please select different columns for comparison.")
    else:
        st.warning("Need at least 2 categorical columns for chi-square test.")

def perform_anova(df):
    st.subheader("One-way ANOVA")
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    if len(numerical_cols) > 0 and len(categorical_cols) > 0:
        num_col = st.selectbox("Select numerical column:", numerical_cols)
        cat_col = st.selectbox("Select categorical column (groups):", categorical_cols)
        
        groups = [group for name, group in df.groupby(cat_col)[num_col]]
        f_stat, p_value = stats.f_oneway(*groups)
        
        st.write(f"F-statistic: {f_stat:.4f}")
        st.write(f"P-value: {p_value:.4f}")
        
        # Visualization
        fig = px.box(df, x=cat_col, y=num_col,
                    title="Distribution Across Groups")
        st.plotly_chart(fig)
    else:
        st.warning("Need both numerical and categorical columns for ANOVA.")

def perform_correlation_analysis(df):
    st.subheader("Correlation Analysis")
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numerical_cols) >= 2:
        correlation_matrix = df[numerical_cols].corr()
        
        st.write("Correlation Matrix:")
        st.write(correlation_matrix)
        
        # Heatmap visualization
        fig = px.imshow(correlation_matrix,
                       labels=dict(color="Correlation Coefficient"),
                       title="Correlation Matrix Heatmap")
        st.plotly_chart(fig)
    else:
        st.warning("Need at least 2 numerical columns for correlation analysis.")

def perform_logistic_regression(X, y, X_col, y_col):
    # Convert y to binary
    y_binary = (y > y.mean()).astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)
    
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    st.write(f"Accuracy score: {accuracy_score(y_test, y_pred):.4f}")
    st.write(f"Coefficient: {model.coef_[0][0]:.4f}")
    st.write(f"Intercept: {model.intercept_[0]:.4f}")
    
    # Visualization
    X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_prob = model.predict_proba(X_line)[:, 1]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X.flatten(), y=y_binary,
                            mode='markers', name='Data Points'))
    fig.add_trace(go.Scatter(x=X_line.flatten(), y=y_prob,
                            mode='lines', name='Logistic Curve'))
    fig.update_layout(title='Logistic Regression',
                      xaxis_title=X_col,
                      yaxis_title=y_col)
    st.plotly_chart(fig)

def show_regression_analysis(df):
    st.header("Regression Analysis")
    
    regression_type = st.selectbox(
        "Select Regression Type:",
        ["Linear Regression",
         "Logistic Regression",
         "Lasso Regression",
         "Ridge Regression"]
    )
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numerical_cols) >= 2:
        X_col = st.selectbox("Select independent variable (X):", numerical_cols)
        y_col = st.selectbox("Select dependent variable (y):", numerical_cols)
        
        if X_col != y_col:
            X = df[X_col].values.reshape(-1, 1)
            y = df[y_col].values
            
            if regression_type == "Linear Regression":
                perform_linear_regression(X, y, X_col, y_col)
            elif regression_type == "Logistic Regression":
                perform_logistic_regression(X, y, X_col, y_col)
            elif regression_type == "Lasso Regression":
                perform_lasso_regression(X, y, X_col, y_col)
            elif regression_type == "Ridge Regression":
                perform_ridge_regression(X, y, X_col, y_col)
    else:
        st.warning("Need at least 2 numerical columns for regression analysis.")

def perform_lasso_regression(X, y, X_col, y_col):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = Lasso(alpha=0.1)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    st.write(f"R-squared score: {r2_score(y_test, y_pred):.4f}")
    st.write(f"Coefficient: {model.coef_[0]:.4f}")
    st.write(f"Intercept: {model.intercept_:.4f}")
    
    # Visualization
    fig = px.scatter(x=X.flatten(), y=y, 
                    labels={'x': X_col, 'y': y_col},
                    title="Lasso Regression Plot")
    
    X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_line = model.predict(X_line)
    
    fig.add_trace(go.Scatter(x=X_line.flatten(), y=y_line,
                            mode='lines', name='Regression Line'))
    st.plotly_chart(fig)
def perform_linear_regression(X, y, X_col, y_col):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    st.write(f"R-squared score: {r2_score(y_test, y_pred):.4f}")
    st.write(f"Coefficient: {model.coef_[0]:.4f}")
    st.write(f"Intercept: {model.intercept_:.4f}")
    
    # Visualization
    fig = px.scatter(x=X.flatten(), y=y, 
                    labels={'x': X_col, 'y': y_col},
                    title="Linear Regression Plot")
    
    X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_line = model.predict(X_line)
    
    fig.add_trace(go.Scatter(x=X_line.flatten(), y=y_line,
                            mode='lines', name='Regression Line'))
    st.plotly_chart(fig)

def perform_ridge_regression(X, y, X_col, y_col):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = Ridge(alpha=0.1)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    st.write(f"R-squared score: {r2_score(y_test, y_pred):.4f}")
    st.write(f"Coefficient: {model.coef_[0]:.4f}")
    st.write(f"Intercept: {model.intercept_:.4f}")
    
    # Visualization
    fig = px.scatter(x=X.flatten(), y=y, 
                    labels={'x': X_col, 'y': y_col},
                    title="Ridge Regression Plot")
    
    X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_line = model.predict(X_line)
    
    fig.add_trace(go.Scatter(x=X_line.flatten(), y=y_line,
                            mode='lines', name='Regression Line'))
    st.plotly_chart(fig)

def show_time_series_analysis(df):
    st.header("Time Series Analysis")
    
    # Add stock market data option
    data_source = st.radio(
        "Select Data Source:",
        ["Upload CSV", "Stock Market Data"]
    )
    
    if data_source == "Stock Market Data":
        # Stock selection
        stock_options = ['A', 'AA', 'AACG', 'AACT', 'AADI', 'AAL', 'AAM', 'AAME', 'AAOI', 'AAON', 'AAP', 'AAPL', 'AAT', 'AB', 'ABAT', 'ABBV', 'ABCB', 'ABCL', 'ABEO', 'ABEV', 'ABG', 'ABL', 'ABLLL', 'ABLLW', 'ABLV', 'ABLVW', 'ABM', 'ABNB', 'ABOS', 'ABR', 'ABSI', 'ABT', 'ABTS', 'ABUS', 'ABVC', 'ABVE', 'ABVEW', 'ABVX', 'AC', 'ACA', 'ACAB', 'ACABU', 'ACAD', 'ACB', 'ACCD', 'ACCO', 'ACDC', 'ACEL', 'ACET', 'ACGL', 'ACGLN', 'ACGLO', 'ACHC', 'ACHL', 'ACHR', 'ACHV', 'ACI', 'ACIC', 'ACIU', 'ACIW', 'ACLS', 'ACLX', 'ACM', 'ACMR', 'ACN', 'ACNB', 'ACNT', 'ACON', 'ACONW', 'ACP', 'ACR', 'ACRE', 'ACRS', 'ACRV', 'ACT', 'ACTG', 'ACTU', 'ACU', 'ACV', 'ACVA', 'ACXP', 'ADAG', 'ADAP', 'ADBE', 'ADC', 'ADCT', 'ADD', 'ADEA', 'ADGM', 'ADI', 'ADIL', 'ADM', 'ADMA', 'ADN', 'ADNT', 'ADNWW', 'ADP', 'ADPT', 'ADSE', 'ADSEW', 'ADSK', 'ADT', 'ADTN', 'ADTX', 'ADUS', 'ADV', 'ADVM', 'ADVWW', 'ADX', 'ADXN', 'AE', 'AEAE', 'AEE', 'AEF', 'AEFC', 'AEG', 'AEHL', 'AEHR', 'AEI', 'AEIS', 'AEM', 'AEMD', 'AENT', 'AENTW', 'AEO', 'AEON', 'AEP', 'AER', 'AERT', 'AES', 'AESI', 'AEVA', 'AEYE', 'AFB', 'AFBI', 'AFCG', 'AFG', 'AFGB', 'AFGC', 'AFGD', 'AFGE', 'AFJK', 'AFJKR', 'AFJKU', 'AFL', 'AFMD', 'AFRI', 'AFRIW', 'AFRM', 'AFYA', 'AG', 'AGAE', 'AGCO', 'AGD', 'AGEN', 'AGFY', 'AGI', 'AGIO', 'AGL', 'AGM', 'AGMH', 'AGNC', 'AGNCL', 'AGNCM', 'AGNCN', 'AGNCO', 'AGNCP', 'AGO', 'AGR', 'AGRI', 'AGRO', 'AGS', 'AGX', 'AGYS', 'AHCO', 'AHG', 'AHH', 'AHL', 'AHR', 'AHT', 'AI', 'AIEV', 'AIFF', 'AIFU', 'AIG', 'AIHS', 'AILE', 'AILEW', 'AIM', 'AIMAU', 'AIMAW', 'AIMBU', 'AIMD', 'AIMDW', 'AIN', 'AIO', 'AIOT', 'AIP', 'AIR', 'AIRE', 'AIRG', 'AIRI', 'AIRJ', 'AIRJW', 'AIRS', 'AIRT', 'AIRTP', 'AISP', 'AISPW', 'AIT', 'AITR', 'AITRR', 'AITRU', 'AIV', 'AIXI', 'AIZ', 'AIZN', 'AJG', 'AJX', 'AKA', 'AKAM', 'AKAN', 'AKBA', 'AKO', 'AKR', 'AKRO', 'AKTS', 'AKTX', 'AKYA', 'AL', 'ALAB', 'ALAR', 'ALB', 'ALBT', 'ALC', 'ALCE', 'ALCO', 'ALCY', 'ALCYW', 'ALDFU', 'ALDX', 'ALE', 'ALEC', 'ALEX', 'ALF', 'ALFUU', 'ALFUW', 'ALG', 'ALGM', 'ALGN', 'ALGS', 'ALGT', 'ALHC', 'ALIT', 'ALK', 'ALKS', 'ALKT', 'ALL', 'ALLE', 'ALLK', 'ALLO', 'ALLR', 'ALLT', 'ALLY', 'ALMS', 'ALNT', 'ALNY', 'ALOT', 'ALRM', 'ALRN', 'ALRS', 'ALSA', 'ALSAR', 'ALSAU', 'ALSAW', 'ALSN', 'ALT', 'ALTG', 'ALTI', 'ALTM', 'ALTO', 'ALTR', 'ALTS', 'ALUR', 'ALV', 'ALVO', 'ALVOW', 'ALVR', 'ALX', 'ALXO', 'ALZN', 'AM', 'AMAL', 'AMAT', 'AMBA', 'AMBC', 'AMBI', 'AMBO', 'AMBP', 'AMC', 'AMCR', 'AMCX', 'AMD', 'AME', 'AMED', 'AMG', 'AMGN', 'AMH', 'AMIX', 'AMKR', 'AMLI', 'AMLX', 'AMN', 'AMP', 'AMPG', 'AMPGW', 'AMPH', 'AMPL', 'AMPS', 'AMPX', 'AMPY', 'AMR', 'AMRC', 'AMRK', 'AMRN', 'AMRX', 'AMS', 'AMSC', 'AMSF', 'AMST', 'AMT', 'AMTB', 'AMTD', 'AMTM', 'AMTX', 'AMWD', 'AMWL', 'AMX', 'AMZN', 'AN', 'ANAB', 'ANDE', 'ANEB', 'ANET', 'ANF', 'ANG', 'ANGH', 'ANGHW', 'ANGI', 'ANGO', 'ANIK', 'ANIP', 'ANIX', 'ANL', 'ANNX', 'ANRO', 'ANSC', 'ANSCW', 'ANSS', 'ANTE', 'ANTX', 'ANVS', 'ANY', 'AOD', 'AOMN', 'AOMR', 'AON', 'AORT', 'AOS', 'AOSL', 'AOUT', 'AP', 'APA', 'APAM', 'APCX', 'APCXW', 'APD', 'APDN', 'APEI', 'APG', 'APGE', 'APH', 'API', 'APLD', 'APLE', 'APLM', 'APLS', 'APLT', 'APM', 'APO', 'APOG', 'APOS', 'APP', 'APPF', 'APPN', 'APPS', 'APRE', 'APT', 'APTO', 'APTV', 'APVO', 'APWC', 'APXI', 'APXIW', 'APYX', 'AQB', 'AQMS', 'AQN', 'AQNB', 'AQST', 'AQU', 'AQUNR', 'AR', 'ARAY', 'ARBB', 'ARBE', 'ARBEW', 'ARBK', 'ARBKL', 'ARC', 'ARCB', 'ARCC', 'ARCH', 'ARCO', 'ARCT', 'ARDC', 'ARDT', 'ARDX', 'ARE', 'AREB', 'AREBW', 'AREC', 'AREN', 'ARES', 'ARGD', 'ARGO', 'ARGX', 'ARHS', 'ARI', 'ARIS', 'ARKO', 'ARKOW', 'ARKR', 'ARL', 'ARLO', 'ARLP', 'ARM', 'ARMK', 'ARMN', 'ARMP', 'AROC', 'AROW', 'ARQ', 'ARQQ', 'ARQQW', 'ARQT', 'ARR', 'ARRY', 'ARTL', 'ARTNA', 'ARTV', 'ARTW', 'ARVN', 'ARW', 'ARWR', 'AS', 'ASA', 'ASAI', 'ASAN', 'ASB', 'ASBA', 'ASC', 'ASG', 'ASGI', 'ASGN', 'ASH', 'ASIX', 'ASLE', 'ASM', 'ASMB', 'ASML', 'ASND', 'ASNS', 'ASO', 'ASPI', 'ASPN', 'ASPS', 'ASR', 'ASRT', 'ASRV', 'ASST', 'ASTC', 'ASTE', 'ASTH', 'ASTI', 'ASTL', 'ASTLW', 'ASTS', 'ASUR', 'ASX', 'ASYS', 'ATAI', 'ATAT', 'ATCH', 'ATCO', 'ATCOL', 'ATEC', 'ATEK', 'ATEN', 'ATER', 'ATEX', 'ATGE', 'ATGL', 'ATH', 'ATHA', 'ATHE', 'ATHM', 'ATHS', 'ATI', 'ATIF', 'ATIP', 'ATKR', 'ATLC', 'ATLCL', 'ATLCP', 'ATLCZ', 'ATLO', 'ATLX', 'ATMC', 'ATMCR', 'ATMCU', 'ATMCW', 'ATMU', 'ATMV', 'ATMVR', 'ATNF', 'ATNFW', 'ATNI', 'ATNM', 'ATO', 'ATOM', 'ATOS', 'ATPC', 'ATR', 'ATRA', 'ATRC', 'ATRO', 'ATS', 'ATSG', 'ATUS', 'ATXG', 'ATXI', 'ATXS', 'ATYR', 'AU', 'AUB', 'AUBN', 'AUDC', 'AUID', 'AUMN', 'AUNA', 'AUPH', 'AUR', 'AURA', 'AUROW', 'AUST', 'AUTL', 'AUUD', 'AUUDW', 'AVA', 'AVAH', 'AVAL', 'AVAV', 'AVB', 'AVBP', 'AVD', 'AVDL', 'AVDX', 'AVGO', 'AVGR', 'AVIR', 'AVK', 'AVNS', 'AVNT', 'AVNW', 'AVO', 'AVPT', 'AVPTW', 'AVT', 'AVTE', 'AVTR', 'AVTX', 'AVXL', 'AVY', 'AWF', 'AWH', 'AWI', 'AWK', 'AWP', 'AWR', 'AWRE', 'AWX', 'AX', 'AXDX', 'AXGN', 'AXIL', 'AXL', 'AXNX', 'AXON', 'AXP', 'AXR', 'AXS', 'AXSM', 'AXTA', 'AXTI', 'AY', 'AYI', 'AYRO', 'AYTU', 'AZ', 'AZEK', 'AZI', 'AZN', 'AZO', 'AZPN', 'AZTA', 'AZTR', 'AZUL', 'AZZ', 'B', 'BA', 'BABA', 'BAC', 'BACK', 'BACQU', 'BAER', 'BAERW', 'BAFN', 'BAH', 'BAK', 'BALL', 'BALY', 'BAM', 'BANC', 'BAND', 'BANF', 'BANFP', 'BANL', 'BANR', 'BANX', 'BAOS', 'BAP', 'BARK', 'BASE', 'BATL', 'BATRA', 'BATRK', 'BAX', 'BAYA', 'BAYAR', 'BB', 'BBAI', 'BBAR', 'BBCP', 'BBD', 'BBDC', 'BBDO', 'BBGI', 'BBIO', 'BBLG', 'BBLGW', 'BBN', 'BBSI', 'BBU', 'BBUC', 'BBVA', 'BBW', 'BBWI', 'BBY', 'BC', 'BCAB', 'BCAL', 'BCAN', 'BCAT', 'BCAX', 'BCBP', 'BCC', 'BCDA', 'BCE', 'BCG', 'BCGWW', 'BCH', 'BCLI', 'BCML', 'BCO', 'BCOV', 'BCOW', 'BCPC', 'BCRX', 'BCS', 'BCSA', 'BCSAU', 'BCSAW', 'BCSF', 'BCTX', 'BCTXW', 'BCV', 'BCX', 'BCYC', 'BDC', 'BDJ', 'BDL', 'BDMD', 'BDMDW', 'BDN', 'BDRX', 'BDSX', 'BDTX', 'BDX', 'BE', 'BEAGU', 'BEAM', 'BEAT', 'BEATW', 'BECN', 'BEDU', 'BEEM', 'BEEP', 'BEKE', 'BELFA', 'BELFB', 'BEN', 'BENF', 'BENFW', 'BEP', 'BEPC', 'BEPH', 'BEPI', 'BEPJ', 'BERY', 'BEST', 'BETR', 'BETRW', 'BF', 'BFAC', 'BFAM', 'BFC', 'BFH', 'BFIN', 'BFK', 'BFLY', 'BFRG', 'BFRGW', 'BFRI', 'BFRIW', 'BFS', 'BFST', 'BFZ', 'BG', 'BGB', 'BGC', 'BGFV', 'BGH', 'BGI', 'BGLC', 'BGM', 'BGNE', 'BGR', 'BGS', 'BGSF', 'BGT', 'BGX', 'BGY', 'BH', 'BHAT', 'BHB', 'BHC', 'BHE', 'BHF', 'BHFAL', 'BHFAM', 'BHFAN', 'BHFAO', 'BHFAP', 'BHIL', 'BHK', 'BHLB', 'BHM', 'BHP', 'BHR', 'BHRB', 'BHV', 'BHVN', 'BIAF', 'BIAFW', 'BIDU', 'BIGC', 'BIGZ', 'BIIB', 'BILI', 'BILL', 'BIO', 'BIOA', 'BIOR', 'BIOX', 'BIP', 'BIPC', 'BIPH', 'BIPI', 'BIPJ', 'BIRD', 'BIRK', 'BIT', 'BITF', 'BIVI', 'BJ', 'BJDX', 'BJRI', 'BK', 'BKD', 'BKDT', 'BKE', 'BKH', 'BKHAR', 'BKHAU', 'BKKT', 'BKN', 'BKNG', 'BKR', 'BKSY', 'BKT', 'BKTI', 'BKU', 'BKV', 'BKYI', 'BL', 'BLAC', 'BLACR', 'BLACU', 'BLACW', 'BLBD', 'BLBX', 'BLCO', 'BLD', 'BLDE', 'BLDEW', 'BLDP', 'BLDR', 'BLE', 'BLEU', 'BLEUR', 'BLEUW', 'BLFS', 'BLFY', 'BLIN', 'BLK', 'BLKB', 'BLMN', 'BLMZ', 'BLND', 'BLNK', 'BLRX', 'BLTE', 'BLUE', 'BLW', 'BLX', 'BLZE', 'BMA', 'BMBL', 'BME', 'BMEA', 'BMEZ', 'BMI', 'BML', 'BMN', 'BMO', 'BMR', 'BMRA', 'BMRC', 'BMRN', 'BMTX', 'BMY', 'BN', 'BNAI', 'BNAIW', 'BNED', 'BNGO', 'BNH', 'BNIX', 'BNIXR', 'BNIXW', 'BNJ', 'BNL', 'BNOX', 'BNR', 'BNRG', 'BNS', 'BNT', 'BNTC', 'BNTX', 'BNY', 'BNZI', 'BNZIW', 'BOC', 'BOCN', 'BOCNW', 'BODI', 'BOE', 'BOF', 'BOH', 'BOKF', 'BOLD', 'BOLT', 'BON', 'BOOM', 'BOOT', 'BORR', 'BOSC', 'BOTJ', 'BOW', 'BOWL', 'BOWN', 'BOWNR', 'BOX', 'BOXL', 'BP', 'BPMC', 'BPOP', 'BPOPM', 'BPRN', 'BPT', 'BPTH', 'BPYPM', 'BPYPN', 'BPYPO', 'BPYPP', 'BQ', 'BR', 'BRAC', 'BRAG', 'BRBR', 'BRBS', 'BRC', 'BRCC', 'BRDG', 'BREA', 'BRFH', 'BRFS', 'BRID', 'BRK', 'BRKH', 'BRKHU', 'BRKHW', 'BRKL', 'BRKR', 'BRLS', 'BRLSW', 'BRLT', 'BRN', 'BRNS', 'BRO', 'BROG', 'BROGW', 'BROS', 'BRSP', 'BRT', 'BRTX', 'BRW', 'BRX', 'BRY', 'BRZE', 'BSAC', 'BSBK', 'BSBR', 'BSET', 'BSFC', 'BSGM', 'BSIG', 'BSII', 'BSIIU', 'BSIIW', 'BSL', 'BSLK', 'BSLKW', 'BSM', 'BSRR', 'BST', 'BSTZ', 'BSVN', 'BSX', 'BSY', 'BTA', 'BTAI', 'BTBD', 'BTBDW', 'BTBT', 'BTCM', 'BTCS', 'BTCT', 'BTCTW', 'BTDR', 'BTE', 'BTG', 'BTI', 'BTM', 'BTMD', 'BTMWW', 'BTO', 'BTOC', 'BTOG', 'BTSG', 'BTSGU', 'BTT', 'BTTR', 'BTU', 'BTZ', 'BUD', 'BUI', 'BUJA', 'BUJAW', 'BUR', 'BURL', 'BURU', 'BUSE', 'BV', 'BVFL', 'BVN', 'BVS', 'BW', 'BWA', 'BWAY', 'BWB', 'BWBBP', 'BWEN', 'BWFG', 'BWG', 'BWIN', 'BWLP', 'BWMN', 'BWMX', 'BWNB', 'BWSN', 'BWXT', 'BX', 'BXC', 'BXMT', 'BXMX', 'BXP', 'BXSL', 'BY', 'BYD', 'BYFC', 'BYM', 'BYND', 'BYNO', 'BYON', 'BYRN', 'BYSI', 'BYU', 'BZ', 'BZFD', 'BZFDW', 'BZH', 'BZUN', 'C', 'CAAP', 'CAAS', 'CABA', 'CABO', 'CAC', 'CACC', 'CACI', 'CADE', 'CADL', 'CAE', 'CAF', 'CAG', 'CAH', 'CAKE', 'CAL', 'CALC', 'CALM', 'CALX', 'CAMP', 'CAMT', 'CAN', 'CANF', 'CANG', 'CAPL', 'CAPN', 'CAPNR', 'CAPNU', 'CAPR', 'CAPT', 'CAR', 'CARA', 'CARE', 'CARG', 'CARM', 'CARR', 'CARS', 'CART', 'CARV', 'CASH', 'CASI', 'CASS', 'CASY', 'CAT', 'CATO', 'CATX', 'CATY', 'CAVA', 'CB', 'CBAN', 'CBAT', 'CBFV', 'CBL', 'CBLL', 'CBNA', 'CBNK', 'CBRE', 'CBRG', 'CBRL', 'CBSH', 'CBT', 'CBU', 'CBUS', 'CBZ', 'CC', 'CCAP', 'CCB', 'CCBG', 'CCCC', 'CCCS', 'CCD', 'CCEC', 'CCEL', 'CCEP', 'CCG', 'CCI', 'CCIA', 'CCIF', 'CCIRU', 'CCIX', 'CCIXU', 'CCIXW', 'CCJ', 'CCK', 'CCL', 'CCLD', 'CCLDO', 'CCLDP', 'CCM', 'CCNE', 'CCNEP', 'CCO', 'CCOI', 'CCRD', 'CCRN', 'CCS', 'CCSI', 'CCTG', 'CCTS', 'CCTSU', 'CCU', 'CCZ', 'CDE', 'CDIO', 'CDIOW', 'CDLR', 'CDLX', 'CDMO', 'CDNA', 'CDNS', 'CDP', 'CDR', 'CDRE', 'CDRO', 'CDROW', 'CDT', 'CDTG', 'CDTTW', 'CDTX', 'CDW', 'CDXC', 'CDXS', 'CDZI', 'CDZIP', 'CE', 'CEAD', 'CECO', 'CEE', 'CEG', 'CEIX', 'CELC', 'CELH', 'CELU', 'CELUW', 'CELZ', 'CENN', 'CENT', 'CENTA', 'CENX', 'CEP', 'CEPU', 'CERO', 'CEROW', 'CERS', 'CERT', 'CET', 'CETX', 'CETY', 'CEV', 'CEVA', 'CF', 'CFB', 'CFBK', 'CFFI', 'CFFN', 'CFFS', 'CFFSW', 'CFG', 'CFLT', 'CFR', 'CFSB', 'CG', 'CGA', 'CGABL', 'CGAU', 'CGBD', 'CGBDL', 'CGBS', 'CGBSW', 'CGC', 'CGEM', 'CGEN', 'CGNT', 'CGNX', 'CGO', 'CGON', 'CGTX', 'CHARU', 'CHCI', 'CHCO', 'CHCT', 'CHD', 'CHDN', 'CHE', 'CHEB', 'CHEF', 'CHEK', 'CHGG', 'CHH', 'CHI', 'CHKP', 'CHMG', 'CHMI', 'CHN', 'CHNR', 'CHPT', 'CHR', 'CHRD', 'CHRO', 'CHRS', 'CHRW', 'CHSCL', 'CHSCM', 'CHSCN', 'CHSCO', 'CHSCP', 'CHSN', 'CHT', 'CHTR', 'CHW', 'CHWY', 'CHX', 'CHY', 'CI', 'CIA', 'CIB', 'CICB', 'CIEN', 'CIF', 'CIFR', 'CIFRW', 'CIG', 'CIGI', 'CII', 'CIK', 'CIM', 'CIMN', 'CIMO', 'CINF', 'CING', 'CINGW', 'CINT', 'CIO', 'CION', 'CISO', 'CISS', 'CITE', 'CITEW', 'CIVB', 'CIVI', 'CIX', 'CJET', 'CJJD', 'CKPT', 'CKX', 'CL', 'CLAR', 'CLB', 'CLBK', 'CLBR', 'CLBT', 'CLCO', 'CLDI', 'CLDT', 'CLDX', 'CLEU', 'CLF', 'CLFD', 'CLGN', 'CLH', 'CLIK', 'CLIR', 'CLLS', 'CLM', 'CLMB', 'CLMT', 'CLNE', 'CLNN', 'CLOV', 'CLPR', 'CLPS', 'CLPT', 'CLRB', 'CLRO', 'CLS', 'CLSD', 'CLSK', 'CLST', 'CLVT', 'CLW', 'CLWT', 'CLX', 'CLYM', 'CM', 'CMA', 'CMAX', 'CMAXW', 'CMBM', 'CMBT', 'CMC', 'CMCL', 'CMCM', 'CMCO', 'CMCSA', 'CMCT', 'CME', 'CMG', 'CMI', 'CMLS', 'CMMB', 'CMND', 'CMP', 'CMPO', 'CMPOW', 'CMPR', 'CMPS', 'CMPX', 'CMRE', 'CMRX', 'CMS', 'CMSA', 'CMSC', 'CMSD', 'CMT', 'CMTG', 'CMTL', 'CMU', 'CNA', 'CNC', 'CNDT', 'CNET', 'CNEY', 'CNF', 'CNFR', 'CNFRZ', 'CNH', 'CNI', 'CNK', 'CNL', 'CNM', 'CNMD', 'CNNE', 'CNO', 'CNOB', 'CNOBP', 'CNP', 'CNQ', 'CNS', 'CNSL', 'CNSP', 'CNTA', 'CNTB', 'CNTM', 'CNTX', 'CNTY', 'CNVS', 'CNX', 'CNXC', 'CNXN', 'COCH', 'COCHW', 'COCO', 'COCP', 'CODA', 'CODI', 'CODX', 'COE', 'COEP', 'COEPW', 'COF', 'COFS', 'COGT', 'COHN', 'COHR', 'COHU', 'COIN', 'COKE', 'COLB', 'COLD', 'COLL', 'COLM', 'COMM', 'COMP', 'CON', 'COO', 'COOK', 'COOP', 'COOT', 'COOTW', 'COP', 'COR', 'CORT', 'CORZ', 'CORZW', 'CORZZ', 'COSM', 'COST', 'COTY', 'COUR', 'COYA', 'CP', 'CPA', 'CPAC', 'CPAY', 'CPB', 'CPBI', 'CPF', 'CPHC', 'CPHI', 'CPIX', 'CPK', 'CPNG', 'CPOP', 'CPRI', 'CPRT', 'CPRX', 'CPS', 'CPSH', 'CPSS', 'CPT', 'CPTN', 'CPTNW', 'CPZ', 'CQP', 'CR', 'CRAI', 'CRBG', 'CRBP', 'CRBU', 'CRC', 'CRCT', 'CRD', 'CRDF', 'CRDL', 'CRDO', 'CREG', 'CRESW', 'CRESY', 'CREV', 'CREX', 'CRF', 'CRGO', 'CRGOW', 'CRGX', 'CRGY', 'CRH', 'CRI', 'CRIS', 'CRK', 'CRKN', 'CRL', 'CRM', 'CRMD', 'CRML', 'CRMLW', 'CRMT', 'CRNC', 'CRNT', 'CRNX', 'CRON', 'CROX', 'CRS', 'CRSP', 'CRSR', 'CRT', 'CRTO', 'CRUS', 'CRVL', 'CRVO', 'CRVS', 'CRWD', 'CRWS', 'CSAN', 'CSBR', 'CSCI', 'CSCO', 'CSGP', 'CSGS', 'CSIQ', 'CSL', 'CSLM', 'CSLR', 'CSLRW', 'CSPI', 'CSQ', 'CSR', 'CSTE', 'CSTL', 'CSTM', 'CSV', 'CSWC', 'CSWCZ', 'CSWI', 'CSX', 'CTA', 'CTAS', 'CTBB', 'CTBI', 'CTCX', 'CTDD', 'CTGO', 'CTHR', 'CTKB', 'CTLP', 'CTLT', 'CTM', 'CTMX', 'CTNM', 'CTNT', 'CTO', 'CTOR', 'CTOS', 'CTRA', 'CTRE', 'CTRI', 'CTRM', 'CTRN', 'CTS', 'CTSH', 'CTSO', 'CTV', 'CTVA', 'CTXR', 'CUB', 'CUBA', 'CUBB', 'CUBE', 'CUBI', 'CUBWU', 'CUBWW', 'CUE', 'CUK', 'CULP', 'CURB', 'CURI', 'CURIW', 'CURR', 'CURV', 'CUTR', 'CUZ', 'CVAC', 'CVBF', 'CVCO', 'CVE', 'CVEO', 'CVGI', 'CVGW', 'CVI', 'CVKD', 'CVLG', 'CVLT', 'CVM', 'CVNA', 'CVR', 'CVRX', 'CVS', 'CVU', 'CVV', 'CVX', 'CW', 'CWAN', 'CWBC', 'CWCO', 'CWD', 'CWEN', 'CWH', 'CWK', 'CWST', 'CWT', 'CX', 'CXAI', 'CXAIW', 'CXDO', 'CXE', 'CXH', 'CXM', 'CXT', 'CXW', 'CYBN', 'CYBR', 'CYCC', 'CYCCP', 'CYCN', 'CYD', 'CYH', 'CYN', 'CYRX', 'CYTH', 'CYTHW', 'CYTK', 'CYTO', 'CZFS', 'CZNC', 'CZR', 'CZWI', 'D', 'DAC', 'DADA', 'DAIO', 'DAKT', 'DAL', 'DALN', 'DAN', 'DAO', 'DAR', 'DARE', 'DASH', 'DATS', 'DATSW', 'DAVA', 'DAVE', 'DAVEW', 'DAWN', 'DAY', 'DB', 'DBD', 'DBGI', 'DBGIW', 'DBI', 'DBL', 'DBRG', 'DBVT', 'DBX', 'DC', 'DCBO', 'DCF', 'DCGO', 'DCI', 'DCO', 'DCOM', 'DCOMG', 'DCOMP', 'DCTH', 'DD', 'DDC', 'DDD', 'DDI', 'DDL', 'DDOG', 'DDS', 'DDT', 'DE', 'DEA', 'DEC', 'DECA', 'DECAW', 'DECK', 'DEI', 'DELL', 'DENN', 'DEO', 'DERM', 'DESP', 'DFH', 'DFIN', 'DFLI', 'DFLIW', 'DFP', 'DFS', 'DG', 'DGHI', 'DGICA', 'DGICB', 'DGII', 'DGLY', 'DGX', 'DH', 'DHAI', 'DHAIW', 'DHC', 'DHCNI', 'DHCNL', 'DHF', 'DHI', 'DHIL', 'DHR', 'DHT', 'DHX', 'DHY', 'DIAX', 'DIBS', 'DIN', 'DINO', 'DIOD', 'DIS', 'DIST', 'DISTW', 'DIT', 'DJCO', 'DJT', 'DJTWW', 'DK', 'DKL', 'DKNG', 'DKS', 'DLB', 'DLHC', 'DLNG', 'DLO', 'DLPN', 'DLR', 'DLTH', 'DLTR', 'DLX', 'DLY', 'DM', 'DMA', 'DMAC', 'DMB', 'DMF', 'DMLP', 'DMO', 'DMRC', 'DMYY', 'DNA', 'DNB', 'DNLI', 'DNMR', 'DNN', 'DNOW', 'DNP', 'DNTH', 'DNUT', 'DOC', 'DOCN', 'DOCS', 'DOCU', 'DOGZ', 'DOLE', 'DOMH', 'DOMO', 'DOOO', 'DORM', 'DOUG', 'DOV', 'DOW', 'DOX', 'DOYU', 'DPCS', 'DPCSW', 'DPG', 'DPRO', 'DPZ', 'DQ', 'DRCT', 'DRD', 'DRH', 'DRI', 'DRIO', 'DRMA', 'DRMAW', 'DRRX', 'DRS', 'DRTS', 'DRTSW', 'DRUG', 'DRVN', 'DSGN', 'DSGR', 'DSGX', 'DSL', 'DSM', 'DSP', 'DSS', 'DSU', 'DSWL', 'DSX', 'DSY', 'DSYWW', 'DT', 'DTB', 'DTC', 'DTCK', 'DTE', 'DTF', 'DTG', 'DTI', 'DTIL', 'DTM', 'DTSQ', 'DTSQU', 'DTSS', 'DTST', 'DTSTW', 'DTW', 'DUETW', 'DUK', 'DUKB', 'DUO', 'DUOL', 'DUOT', 'DV', 'DVA', 'DVAX', 'DVN', 'DWSN', 'DWTX', 'DX', 'DXC', 'DXCM', 'DXLG', 'DXPE', 'DXR', 'DXYZ', 'DY', 'DYAI', 'DYCQ', 'DYCQR', 'DYN', 'E', 'EA', 'EAD', 'EAF', 'EAI', 'EARN', 'EAST', 'EAT', 'EB', 'EBAY', 'EBC', 'EBF', 'EBMT', 'EBON', 'EBR', 'EBS', 'EBTC', 'EC', 'ECAT', 'ECBK', 'ECC', 'ECCC', 'ECCF', 'ECCV', 'ECCW', 'ECCX', 'ECDA', 'ECF', 'ECG', 'ECL', 'ECO', 'ECOR', 'ECPG', 'ECVT', 'ECX', 'ECXWW', 'ED', 'EDAP', 'EDBL', 'EDBLW', 'EDD', 'EDF', 'EDIT', 'EDN', 'EDR', 'EDRY', 'EDSA', 'EDTK', 'EDU', 'EDUC', 'EE', 'EEA', 'EEFT', 'EEIQ', 'EEX', 'EFC', 'EFOI', 'EFR', 'EFSC', 'EFSCP', 'EFSH', 'EFT', 'EFX', 'EFXT', 'EG', 'EGAN', 'EGBN', 'EGF', 'EGY', 'EH', 'EHAB', 'EHC', 'EHGO', 'EHI', 'EHTH', 'EIC', 'EICA', 'EICB', 'EICC', 'EIG', 'EIIA', 'EIM', 'EIX', 'EJH', 'EKSO', 'EL', 'ELA', 'ELAB', 'ELAN', 'ELBM', 'ELC', 'ELDN', 'ELEV', 'ELF', 'ELLO', 'ELMD', 'ELME', 'ELP', 'ELPC', 'ELS', 'ELSE', 'ELTK', 'ELTX', 'ELUT', 'ELV', 'ELVA', 'ELVN', 'ELWS', 'EM', 'EMBC', 'EMCG', 'EMCGU', 'EMD', 'EME', 'EMF', 'EMKR', 'EML', 'EMN', 'EMO', 'EMP', 'EMR', 'EMX', 'ENB', 'ENFN', 'ENG', 'ENGN', 'ENGNW', 'ENIC', 'ENJ', 'ENLC', 'ENLT', 'ENLV', 'ENO', 'ENOV', 'ENPH', 'ENR', 'ENS', 'ENSC', 'ENSG', 'ENSV', 'ENTA', 'ENTG', 'ENTO', 'ENTX', 'ENV', 'ENVA', 'ENVB', 'ENVX', 'ENX', 'ENZ', 'EOD', 'EOG', 'EOI', 'EOLS', 'EONR', 'EOS', 'EOSE', 'EOSEW', 'EOT', 'EP', 'EPAC', 'EPAM', 'EPC', 'EPD', 'EPIX', 'EPM', 'EPOW', 'EPR', 'EPRT', 'EPRX', 'EPSN', 'EQ', 'EQBK', 'EQC', 'EQH', 'EQIX', 'EQNR', 'EQR', 'EQS', 'EQT', 'EQV', 'EQX', 'ERAS', 'ERC', 'ERH', 'ERIC', 'ERIE', 'ERII', 'ERJ', 'ERNA', 'ERO', 'ES', 'ESAB', 'ESCA', 'ESE', 'ESEA', 'ESGL', 'ESGLW', 'ESGR', 'ESGRO', 'ESGRP', 'ESHA', 'ESHAR', 'ESI', 'ESLA', 'ESLT', 'ESNT', 'ESOA', 'ESP', 'ESPR', 'ESQ', 'ESRT', 'ESS', 'ESSA', 'ESTA', 'ESTC', 'ET', 'ETB', 'ETD', 'ETG', 'ETI', 'ETJ', 'ETN', 'ETNB', 'ETO', 'ETON', 'ETR', 'ETSY', 'ETV', 'ETW', 'ETWO', 'ETX', 'ETY', 'EU', 'EUDA', 'EUDAW', 'EURKR', 'EVAX', 'EVBN', 'EVC', 'EVCM', 'EVE', 'EVER', 'EVEX', 'EVF', 'EVG', 'EVGN', 'EVGO', 'EVGOW', 'EVGR', 'EVGRU', 'EVH', 'EVI', 'EVLV', 'EVLVW', 'EVM', 'EVN', 'EVO', 'EVOK', 'EVR', 'EVRG', 'EVRI', 'EVT', 'EVTC', 'EVTL', 'EVTV', 'EVV', 'EW', 'EWBC', 'EWCZ', 'EWTX', 'EXAI', 'EXAS', 'EXC', 'EXE', 'EXEEL', 'EXEEZ', 'EXEL', 'EXFY', 'EXG', 'EXK', 'EXLS', 'EXP', 'EXPD', 'EXPE', 'EXPI', 'EXPO', 'EXR', 'EXTO', 'EXTR', 'EYE', 'EYEN', 'EYPT', 'EZFL', 'EZGO', 'EZPW', 'F', 'FA', 'FAAS', 'FAASW', 'FAF', 'FAMI', 'FANG', 'FARM', 'FARO', 'FAST', 'FAT', 'FATBB', 'FATBP', 'FATBW', 'FATE', 'FAX', 'FBIN', 'FBIO', 'FBIOP', 'FBIZ', 'FBK', 'FBLA', 'FBLG', 'FBMS', 'FBNC', 'FBP', 'FBRT', 'FBRX', 'FBYD', 'FBYDW', 'FC', 'FCAP', 'FCBC', 'FCCO', 'FCEL', 'FCF', 'FCFS', 'FCN', 'FCNCA', 'FCNCO', 'FCNCP', 'FCO', 'FCPT', 'FCRX', 'FCT', 'FCUV', 'FCX', 'FDBC', 'FDMT', 'FDP', 'FDSB', 'FDUS', 'FDX', 'FE', 'FEAM', 'FEBO', 'FEDU', 'FEIM', 'FELE', 'FEMY', 'FENC', 'FENG', 'FER', 'FERG', 'FET', 'FF', 'FFA', 'FFBC', 'FFC', 'FFIC', 'FFIE', 'FFIEW', 'FFIN', 'FFIV', 'FFNW', 'FFWM', 'FG', 'FGB', 'FGBI', 'FGBIP', 'FGEN', 'FGF', 'FGFPP', 'FGI', 'FGL', 'FGN', 'FHB', 'FHI', 'FHN', 'FHTX', 'FI', 'FIAC', 'FIACU', 'FIBK', 'FICO', 'FIGS', 'FIHL', 'FINS', 'FINV', 'FINW', 'FIP', 'FIS', 'FISI', 'FITB', 'FITBI', 'FITBO', 'FITBP', 'FIVE', 'FIVN', 'FIX', 'FIZZ', 'FKWL', 'FL', 'FLC', 'FLD', 'FLDDU', 'FLDDW', 'FLEX', 'FLG', 'FLGC', 'FLGT', 'FLIC', 'FLL', 'FLNC', 'FLNG', 'FLNT', 'FLO', 'FLR', 'FLS', 'FLUT', 'FLUX', 'FLWS', 'FLX', 'FLXS', 'FLYE', 'FLYW', 'FLYX', 'FMAO', 'FMBH', 'FMC', 'FMN', 'FMNB', 'FMS', 'FMST', 'FMSTW', 'FMX', 'FMY', 'FN', 'FNA', 'FNB', 'FND', 'FNF', 'FNGR', 'FNKO', 'FNLC', 'FNV', 'FNVTW', 'FNWB', 'FNWD', 'FOA', 'FOF', 'FOLD', 'FONR', 'FOR', 'FORA', 'FORD', 'FORL', 'FORLW', 'FORM', 'FORR', 'FORTY', 'FOSL', 'FOSLL', 'FOUR', 'FOX', 'FOXA', 'FOXF', 'FOXO', 'FOXX', 'FOXXW', 'FPAY', 'FPF', 'FPH', 'FPI', 'FR', 'FRA', 'FRAF', 'FRBA', 'FRD', 'FRES', 'FREY', 'FRGE', 'FRGT', 'FRHC', 'FRLA', 'FRLAW', 'FRME', 'FRMEP', 'FRO', 'FROG', 'FRPH', 'FRPT', 'FRSH', 'FRST', 'FRSX', 'FRT', 'FSBC', 'FSBW', 'FSCO', 'FSEA', 'FSFG', 'FSHPR', 'FSHPU', 'FSI', 'FSK', 'FSLR', 'FSLY', 'FSM', 'FSP', 'FSS', 'FSTR', 'FSUN', 'FSV', 'FT', 'FTAI', 'FTAIM', 'FTAIN', 'FTAIO', 'FTCI', 'FTDR', 'FTEK', 'FTEL', 'FTF', 'FTFT', 'FTHM', 'FTHY', 'FTI', 'FTII', 'FTK', 'FTLF', 'FTNT', 'FTRE', 'FTS', 'FTV', 'FUBO', 'FUFU', 'FUFUW', 'FUL', 'FULC', 'FULT', 'FULTP', 'FUN', 'FUNC', 'FUND', 'FURY', 'FUSB', 'FUTU', 'FVCB', 'FVNNU', 'FVR', 'FVRR', 'FWONA', 'FWONK', 'FWRD', 'FWRG', 'FXNC', 'FYBR', 'G', 'GAB', 'GABC', 'GAIA', 'GAIN', 'GAINL', 'GAINN', 'GAINZ', 'GALT', 'GAM', 'GAMB', 'GAME', 'GAN', 'GANX', 'GAP', 'GAQ', 'GASS', 'GATE', 'GATO', 'GATX', 'GAU', 'GAUZ', 'GB', 'GBAB', 'GBBK', 'GBBKR', 'GBCI', 'GBDC', 'GBIO', 'GBLI', 'GBR', 'GBTG', 'GBX', 'GCBC', 'GCI', 'GCMG', 'GCMGW', 'GCO', 'GCT', 'GCTK', 'GCTS', 'GCV', 'GD', 'GDC', 'GDDY', 'GDEN', 'GDEV', 'GDEVW', 'GDHG', 'GDL', 'GDO', 'GDOT', 'GDRX', 'GDS', 'GDTC', 'GDV', 'GDYN', 'GE', 'GECC', 'GECCH', 'GECCI', 'GECCO', 'GECCZ', 'GEF', 'GEG', 'GEGGL', 'GEHC', 'GEL', 'GELS', 'GEN', 'GENC', 'GENI', 'GENK', 'GEO', 'GEOS', 'GERN', 'GES', 'GETY', 'GEV', 'GEVO', 'GF', 'GFAI', 'GFAIW', 'GFF', 'GFI', 'GFL', 'GFR', 'GFS', 'GGAL', 'GGB', 'GGG', 'GGN', 'GGR', 'GGROW', 'GGT', 'GGZ', 'GH', 'GHC', 'GHG', 'GHI', 'GHIX', 'GHIXU', 'GHLD', 'GHM', 'GHRS', 'GHY', 'GIB', 'GIC', 'GIFI', 'GIFT', 'GIG', 'GIGGU', 'GIGGW', 'GIGM', 'GIII', 'GIL', 'GILD', 'GILT', 'GIPR', 'GIPRW', 'GIS', 'GJH', 'GJP', 'GJR', 'GJS', 'GJT', 'GKOS', 'GL', 'GLAC', 'GLACR', 'GLAD', 'GLADZ', 'GLBE', 'GLBS', 'GLBZ', 'GLDD', 'GLDG', 'GLE', 'GLLI', 'GLLIR', 'GLMD', 'GLNG', 'GLO', 'GLOB', 'GLOP', 'GLP', 'GLPG', 'GLPI', 'GLQ', 'GLRE', 'GLSI', 'GLST', 'GLSTR', 'GLSTU', 'GLT', 'GLTO', 'GLU', 'GLUE', 'GLV', 'GLW', 'GLXG', 'GLYC', 'GM', 'GMAB', 'GME', 'GMED', 'GMGI', 'GMM', 'GMRE', 'GMS', 'GNE', 'GNFT', 'GNK', 'GNL', 'GNLN', 'GNLX', 'GNPX', 'GNRC', 'GNS', 'GNSS', 'GNT', 'GNTA', 'GNTX', 'GNTY', 'GNW', 'GO', 'GOCO', 'GODN', 'GODNR', 'GODNU', 'GOEV', 'GOEVW', 'GOF', 'GOGL', 'GOGO', 'GOLD', 'GOLF', 'GOOD', 'GOODN', 'GOODO', 'GOOG', 'GOOGL', 'GOOS', 'GORO', 'GORV', 'GOSS', 'GOTU', 'GOVX', 'GOVXW', 'GP', 'GPAT', 'GPATU', 'GPATW', 'GPC', 'GPCR', 'GPI', 'GPJA', 'GPK', 'GPMT', 'GPN', 'GPOR', 'GPRE', 'GPRK', 'GPRO', 'GPUS', 'GRAB', 'GRABW', 'GRAF', 'GRAL', 'GRBK', 'GRC', 'GRCE', 'GRDN', 'GREE', 'GREEL', 'GRF', 'GRFS', 'GRFX', 'GRI', 'GRMN', 'GRND', 'GRNQ', 'GRNT', 'GROV', 'GROW', 'GROY', 'GRPN', 'GRRR', 'GRRRW', 'GRVY', 'GRWG', 'GRX', 'GRYP', 'GS', 'GSAT', 'GSBC', 'GSBD', 'GSHD', 'GSIT', 'GSIW', 'GSK', 'GSL', 'GSM', 'GSMGW', 'GSUN', 'GT', 'GTBP', 'GTE', 'GTEC', 'GTES', 'GTI', 'GTIM', 'GTLB', 'GTLS', 'GTN', 'GTX', 'GTY', 'GUG', 'GURE', 'GUT', 'GUTS', 'GV', 'GVA', 'GVH', 'GVP', 'GWAV', 'GWH', 'GWRE', 'GWRS', 'GWW', 'GXAI', 'GXO', 'GYRE', 'GYRO', 'H', 'HAE', 'HAFC', 'HAFN', 'HAIA', 'HAIN', 'HAL', 'HALO', 'HAO', 'HAS', 'HASI', 'HAYN', 'HAYW', 'HBAN', 'HBANL', 'HBANM', 'HBANP', 'HBB', 'HBCP', 'HBI', 'HBIO', 'HBM', 'HBNC', 'HBT', 'HCA', 'HCAT', 'HCC', 'HCI', 'HCKT', 'HCM', 'HCP', 'HCSG', 'HCTI', 'HCVI', 'HCWB', 'HCWC', 'HCXY', 'HD', 'HDB', 'HDL', 'HDSN', 'HE', 'HEAR', 'HEES', 'HEI', 'HELE', 'HEPA', 'HEPS', 'HEQ', 'HES', 'HESM', 'HFBL', 'HFFG', 'HFRO', 'HFWA', 'HG', 'HGBL', 'HGLB', 'HGTY', 'HGV', 'HHH', 'HHS', 'HI', 'HIE', 'HIFS', 'HIG', 'HIHO', 'HII', 'HIMS', 'HIMX', 'HIO', 'HIPO', 'HITI', 'HIVE', 'HIW', 'HIX', 'HKD', 'HKIT', 'HL', 'HLF', 'HLI', 'HLIO', 'HLIT', 'HLLY', 'HLMN', 'HLN', 'HLNE', 'HLP', 'HLT', 'HLVX', 'HLX', 'HLXB', 'HMC', 'HMN', 'HMST', 'HMY', 'HNI', 'HNNA', 'HNNAZ', 'HNRG', 'HNST', 'HNVR', 'HNW', 'HOFT', 'HOFV', 'HOFVW', 'HOG', 'HOLO', 'HOLOW', 'HOLX', 'HOMB', 'HON', 'HOND', 'HONDU', 'HONDW', 'HONE', 'HOOD', 'HOOK', 'HOPE', 'HOTH', 'HOUR', 'HOUS', 'HOV', 'HOVNP', 'HOVR', 'HOVRW', 'HOWL', 'HP', 'HPAI', 'HPAIW', 'HPE', 'HPF', 'HPH', 'HPI', 'HPK', 'HPKEW', 'HPP', 'HPQ', 'HPS', 'HQH', 'HQI', 'HQL', 'HQY', 'HR', 'HRB', 'HRI', 'HRL', 'HRMY', 'HROW', 'HROWL', 'HROWM', 'HRTG', 'HRTX', 'HRYU', 'HRZN', 'HSAI', 'HSBC', 'HSCS', 'HSDT', 'HSHP', 'HSIC', 'HSII', 'HSON', 'HSPO', 'HSPOU', 'HST', 'HSTM', 'HSY', 'HTBI', 'HTBK', 'HTCO', 'HTCR', 'HTD', 'HTFB', 'HTFC', 'HTGC', 'HTH', 'HTHT', 'HTIA', 'HTIBP', 'HTLD', 'HTLF', 'HTLFP', 'HTLM', 'HTOO', 'HTOOW', 'HTZ', 'HTZWW', 'HUBB', 'HUBC', 'HUBCW', 'HUBCZ', 'HUBG', 'HUBS', 'HUDI', 'HUHU', 'HUIZ', 'HUM', 'HUMA', 'HUMAW', 'HUN', 'HURA', 'HURC', 'HURN', 'HUSA', 'HUT', 'HUYA', 'HVT', 'HWBK', 'HWC', 'HWCPZ', 'HWH', 'HWKN', 'HWM', 'HXL', 'HY', 'HYAC', 'HYB', 'HYFM', 'HYI', 'HYLN', 'HYMC', 'HYMCL', 'HYMCW', 'HYPR', 'HYT', 'HYZN', 'HYZNW', 'HZO', 'I', 'IAC', 'IAE', 'IAF', 'IAG', 'IART', 'IAS', 'IAUX', 'IBAC', 'IBACR', 'IBCP', 'IBEX', 'IBG', 'IBIO', 'IBKR', 'IBM', 'IBN', 'IBO', 'IBOC', 'IBP', 'IBRX', 'IBTA', 'IBTX', 'ICAD', 'ICCC', 'ICCH', 'ICCM', 'ICCT', 'ICE', 'ICFI', 'ICG', 'ICHR', 'ICL', 'ICLK', 'ICLR', 'ICMB', 'ICON', 'ICR', 'ICU', 'ICUCW', 'ICUI', 'IDA', 'IDAI', 'IDCC', 'IDE', 'IDN', 'IDR', 'IDT', 'IDXX', 'IDYA', 'IE', 'IEP', 'IESC', 'IEX', 'IFBD', 'IFF', 'IFN', 'IFRX', 'IFS', 'IGA', 'IGC', 'IGD', 'IGI', 'IGIC', 'IGMS', 'IGR', 'IGT', 'IGTA', 'IH', 'IHD', 'IHG', 'IHRT', 'IHS', 'IHT', 'IHTA', 'IIF', 'III', 'IIIN', 'IIIV', 'IIM', 'IINN', 'IINNW', 'IIPR', 'IKNA', 'IKT', 'ILAG', 'ILLR', 'ILLRW', 'ILMN', 'ILPT', 'IMAB', 'IMAX', 'IMCC', 'IMCR', 'IMG', 'IMKTA', 'IMMP', 'IMMR', 'IMMX', 'IMNM', 'IMNN', 'IMO', 'IMOS', 'IMPP', 'IMPPP', 'IMRN', 'IMRX', 'IMTE', 'IMTX', 'IMTXW', 'IMUX', 'IMVT', 'IMXI', 'INAB', 'INAQ', 'INAQU', 'INAQW', 'INBK', 'INBKZ', 'INBS', 'INBX', 'INCR', 'INCY', 'INDB', 'INDI', 'INDO', 'INDP', 'INDV', 'INFA', 'INFN', 'INFU', 'INFY', 'ING', 'INGM', 'INGN', 'INGR', 'INHD', 'INKT', 'INLX', 'INM', 'INMB', 'INMD', 'INN', 'INNV', 'INO', 'INOD', 'INSE', 'INSG', 'INSI', 'INSM', 'INSP', 'INST', 'INSW', 'INTA', 'INTC', 'INTE', 'INTG', 'INTJ', 'INTR', 'INTS', 'INTT', 'INTU', 'INTZ', 'INUV', 'INV', 'INVA', 'INVE', 'INVH', 'INVX', 'INVZ', 'INVZW', 'INZY', 'IOBT', 'IONQ', 'IONR', 'IONS', 'IOR', 'IOSP', 'IOT', 'IOVA', 'IP', 'IPA', 'IPAR', 'IPDN', 'IPG', 'IPGP', 'IPHA', 'IPI', 'IPSC', 'IPW', 'IPWR', 'IPX', 'IPXX', 'IPXXU', 'IPXXW', 'IQ', 'IQI', 'IQV', 'IR', 'IRBT', 'IRD', 'IRDM', 'IREN', 'IRIX', 'IRM', 'IRMD', 'IROH', 'IRON', 'IROQ', 'IRS', 'IRT', 'IRTC', 'IRWD', 'ISD', 'ISDR', 'ISPC', 'ISPO', 'ISPOW', 'ISPR', 'ISRG', 'ISRL', 'ISRLU', 'ISRLW', 'ISSC', 'ISTR', 'IT', 'ITCI', 'ITGR', 'ITI', 'ITIC', 'ITOS', 'ITP', 'ITRG', 'ITRI', 'ITRM', 'ITRN', 'ITT', 'ITUB', 'ITW', 'IVA', 'IVAC', 'IVCA', 'IVCAU', 'IVCAW', 'IVCB', 'IVCBW', 'IVCP', 'IVDA', 'IVDAW', 'IVP', 'IVR', 'IVT', 'IVVD', 'IVZ', 'IX', 'IXHL', 'IZEA', 'IZM', 'J', 'JACK', 'JAGX', 'JAKK', 'JAMF', 'JANX', 'JAZZ', 'JBDI', 'JBGS', 'JBHT', 'JBI', 'JBL', 'JBLU', 'JBSS', 'JBT', 'JCE', 'JCI', 'JCSE', 'JCTC', 'JD', 'JDZG', 'JEF', 'JELD', 'JEQ', 'JFBR', 'JFBRW', 'JFIN', 'JFR', 'JFU', 'JG', 'JGH', 'JHG', 'JHI', 'JHS', 'JHX', 'JILL', 'JJSF', 'JKHY', 'JKS', 'JL', 'JLL', 'JLS', 'JMIA', 'JMM', 'JMSB', 'JNJ', 'JNPR', 'JNVR', 'JOB', 'JOBY', 'JOE', 'JOF', 'JOUT', 'JPC', 'JPI', 'JPM', 'JQC', 'JRI', 'JRS', 'JRSH', 'JRVR', 'JSM', 'JSPR', 'JSPRW', 'JTAI', 'JUNE', 'JVA', 'JVSA', 'JVSAU', 'JWEL', 'JWN', 'JWSM', 'JXJT', 'JXN', 'JYD', 'JYNT', 'JZ', 'JZXN', 'K', 'KACL', 'KACLR', 'KACLW', 'KAI', 'KALA', 'KALU', 'KALV', 'KAPA', 'KAR', 'KARO', 'KAVL', 'KB', 'KBDC', 'KBH', 'KBR', 'KC', 'KD', 'KDLY', 'KDLYW', 'KDP', 'KE', 'KELYA', 'KELYB', 'KEN', 'KEP', 'KEQU', 'KEX', 'KEY', 'KEYS', 'KF', 'KFFB', 'KFRC', 'KFS', 'KFY', 'KGC', 'KGEI', 'KGS', 'KHC', 'KIDS', 'KIM', 'KIND', 'KINS', 'KIO', 'KIRK', 'KITT', 'KITTW', 'KKR', 'KKRS', 'KLAC', 'KLC', 'KLG', 'KLIC', 'KLTO', 'KLTOW', 'KLTR', 'KLXE', 'KMB', 'KMDA', 'KMI', 'KMPB', 'KMPR', 'KMT', 'KMX', 'KN', 'KNDI', 'KNF', 'KNOP', 'KNSA', 'KNSL', 'KNTK', 'KNW', 'KNX', 'KO', 'KOD', 'KODK', 'KOF', 'KOP', 'KOPN', 'KORE', 'KOS', 'KOSS', 'KPLT', 'KPLTW', 'KPRX', 'KPTI', 'KR', 'KRC', 'KREF', 'KRG', 'KRKR', 'KRMD', 'KRNT', 'KRNY', 'KRO', 'KRON', 'KROS', 'KRP', 'KRRO', 'KRT', 'KRUS', 'KRYS', 'KSCP', 'KSM', 'KSPI', 'KSS', 'KT', 'KTB', 'KTCC', 'KTF', 'KTH', 'KTN', 'KTOS', 'KTTA', 'KTTAW', 'KUKE', 'KULR', 'KURA', 'KVAC', 'KVACU', 'KVACW', 'KVHI', 'KVUE', 'KVYO', 'KW', 'KWE', 'KWESW', 'KWR', 'KXIN', 'KYMR', 'KYN', 'KYTX', 'KZIA', 'KZR', 'L', 'LAAC', 'LAB', 'LAC', 'LAD', 'LADR', 'LAES', 'LAKE', 'LAMR', 'LANC', 'LAND', 'LANDM', 'LANDO', 'LANDP', 'LANV', 'LARK', 'LASE', 'LASR', 'LATG', 'LAUR', 'LAW', 'LAZ', 'LAZR', 'LB', 'LBGJ', 'LBPH', 'LBRDA', 'LBRDK', 'LBRDP', 'LBRT', 'LBTYA', 'LBTYB', 'LBTYK', 'LC', 'LCFY', 'LCFYW', 'LCID', 'LCII', 'LCNB', 'LCTX', 'LCUT', 'LDI', 'LDOS', 'LDP', 'LDTC', 'LDTCW', 'LDWY', 'LE', 'LEA', 'LECO', 'LEDS', 'LEE', 'LEG', 'LEGH', 'LEGN', 'LEGT', 'LEN', 'LENZ', 'LEO', 'LESL', 'LEU', 'LEV', 'LEVI', 'LEXX', 'LEXXW', 'LFCR', 'LFLY', 'LFLYW', 'LFMD', 'LFMDP', 'LFST', 'LFT', 'LFUS', 'LFVN', 'LFWD', 'LGCB', 'LGCL', 'LGCY', 'LGHL', 'LGHLW', 'LGI', 'LGIH', 'LGL', 'LGMK', 'LGND', 'LGO', 'LGTY', 'LGVN', 'LH', 'LHX', 'LI', 'LICN', 'LICY', 'LIDR', 'LIDRW', 'LIEN', 'LIF', 'LIFW', 'LIFWW', 'LIFWZ', 'LII', 'LILA', 'LILAK', 'LILM', 'LILMW', 'LIN', 'LINC', 'LIND', 'LINE', 'LINK', 'LION', 'LIPO', 'LIQT', 'LITB', 'LITE', 'LITM', 'LIVE', 'LIVN', 'LIXT', 'LIXTW', 'LKCO', 'LKFN', 'LKQ', 'LLY', 'LLYVA', 'LLYVK', 'LMAT', 'LMB', 'LMFA', 'LMND', 'LMNR', 'LMT', 'LNC', 'LND', 'LNG', 'LNKB', 'LNN', 'LNSR', 'LNT', 'LNTH', 'LNW', 'LNZA', 'LOAN', 'LOAR', 'LOB', 'LOBO', 'LOCL', 'LOCO', 'LODE', 'LOGC', 'LOGI', 'LOMA', 'LOOP', 'LOPE', 'LOT', 'LOTWW', 'LOVE', 'LOW', 'LPA', 'LPAA', 'LPAAU', 'LPAAW', 'LPBBU', 'LPCN', 'LPG', 'LPL', 'LPLA', 'LPRO', 'LPSN', 'LPTH', 'LPTX', 'LPX', 'LQDA', 'LQDT', 'LQR', 'LRCX', 'LRE', 'LRFC', 'LRHC', 'LRMR', 'LRN', 'LSAK', 'LSB', 'LSBK', 'LSCC', 'LSEA', 'LSEAW', 'LSF', 'LSH', 'LSPD', 'LSTA', 'LSTR', 'LTBR', 'LTC', 'LTH', 'LTM', 'LTRN', 'LTRX', 'LTRY', 'LTRYW', 'LU', 'LUCD', 'LUCY', 'LUCYW', 'LULU', 'LUMN', 'LUMO', 'LUNA', 'LUNG', 'LUNR', 'LUNRW', 'LUV', 'LUXH', 'LUXHP', 'LVLU', 'LVO', 'LVRO', 'LVROW', 'LVS', 'LVTX', 'LVWR', 'LW', 'LWAY', 'LWLG', 'LX', 'LXEH', 'LXEO', 'LXFR', 'LXP', 'LXRX', 'LXU', 'LYB', 'LYEL', 'LYFT', 'LYG', 'LYRA', 'LYT', 'LYTS', 'LYV', 'LZ', 'LZB', 'LZM', 'M', 'MA', 'MAA', 'MAC', 'MACI', 'MACIW', 'MAG', 'MAIA', 'MAIN', 'MAMA', 'MAMO', 'MAN', 'MANH', 'MANU', 'MAPS', 'MAPSW', 'MAR', 'MARA', 'MARPS', 'MARX', 'MARXU', 'MAS', 'MASI', 'MASS', 'MAT', 'MATH', 'MATV', 'MATW', 'MATX', 'MAV', 'MAX', 'MAXN', 'MAYS', 'MBAV', 'MBAVU', 'MBAVW', 'MBC', 'MBCN', 'MBI', 'MBIN', 'MBINM', 'MBINN', 'MBINO', 'MBIO', 'MBLY', 'MBNKP', 'MBOT', 'MBRX', 'MBUU', 'MBWM', 'MBX', 'MC', 'MCAA', 'MCAG', 'MCAGR', 'MCB', 'MCBS', 'MCD', 'MCFT', 'MCHP', 'MCHX', 'MCI', 'MCK', 'MCN', 'MCO', 'MCR', 'MCRB', 'MCRI', 'MCS', 'MCVT', 'MCW', 'MCY', 'MD', 'MDAI', 'MDAIW', 'MDB', 'MDBH', 'MDGL', 'MDIA', 'MDJH', 'MDLZ', 'MDRR', 'MDRRP', 'MDT', 'MDU', 'MDV', 'MDWD', 'MDXG', 'MDXH', 'ME', 'MEC', 'MED', 'MEDP', 'MEG', 'MEGI', 'MEGL', 'MEI', 'MEIP', 'MELI', 'MEOH', 'MER', 'MERC', 'MESA', 'MESO', 'MET', 'META', 'METC', 'METCB', 'METCL', 'MFA', 'MFAN', 'MFAO', 'MFC', 'MFG', 'MFH', 'MFI', 'MFIC', 'MFICL', 'MFIN', 'MFM', 'MG', 'MGA', 'MGEE', 'MGF', 'MGIC', 'MGIH', 'MGLD', 'MGM', 'MGNI', 'MGNX', 'MGOL', 'MGPI', 'MGR', 'MGRB', 'MGRC', 'MGRD', 'MGRE', 'MGRM', 'MGRX', 'MGTX', 'MGX', 'MGY', 'MGYR', 'MHD', 'MHF', 'MHH', 'MHI', 'MHK', 'MHLA', 'MHLD', 'MHN', 'MHNC', 'MHO', 'MHUA', 'MI', 'MIDD', 'MIGI', 'MIN', 'MIND', 'MIO', 'MIR', 'MIRA', 'MIRM', 'MIST', 'MITA', 'MITK', 'MITN', 'MITP', 'MITQ', 'MITT', 'MIY', 'MKC', 'MKDW', 'MKDWW', 'MKFG', 'MKL', 'MKSI', 'MKTW', 'MKTX', 'ML', 'MLAB', 'MLCO', 'MLEC', 'MLECW', 'MLGO', 'MLI', 'MLKN', 'MLM', 'MLNK', 'MLP', 'MLR', 'MLSS', 'MLTX', 'MLYS', 'MMA', 'MMC', 'MMD', 'MMI', 'MMLP', 'MMM', 'MMS', 'MMSI', 'MMT', 'MMU', 'MMV', 'MMYT', 'MNDO', 'MNDR', 'MNDY', 'MNKD', 'MNMD', 'MNOV', 'MNPR', 'MNR', 'MNRO', 'MNSB', 'MNSBP', 'MNSO', 'MNST', 'MNTK', 'MNTN', 'MNTS', 'MNTSW', 'MNTX', 'MNY', 'MNYWW', 'MO', 'MOB', 'MOBX', 'MOBXW', 'MOD', 'MODD', 'MODG', 'MODV', 'MOFG', 'MOGO', 'MOGU', 'MOH', 'MOLN', 'MOMO', 'MOND', 'MORN', 'MOS', 'MOV', 'MOVE', 'MP', 'MPA', 'MPAA', 'MPB', 'MPC', 'MPLN', 'MPLX', 'MPTI', 'MPU', 'MPV', 'MPW', 'MPWR', 'MPX', 'MQ', 'MQT', 'MQY', 'MRAM', 'MRBK', 'MRC', 'MRCC', 'MRCY', 'MREO', 'MRIN', 'MRK', 'MRKR', 'MRM', 'MRNA', 'MRNO', 'MRNOW', 'MRNS', 'MRO', 'MRSN', 'MRT', 'MRTN', 'MRUS', 'MRVI', 'MRVL', 'MRX', 'MS', 'MSA', 'MSAI', 'MSAIW', 'MSB', 'MSBI', 'MSBIP', 'MSC', 'MSCI', 'MSD', 'MSDL', 'MSEX', 'MSFT', 'MSGE', 'MSGM', 'MSGS', 'MSI', 'MSM', 'MSN', 'MSS', 'MSSA', 'MSSAR', 'MSTR', 'MT', 'MTA', 'MTAL', 'MTB', 'MTC', 'MTCH', 'MTD', 'MTDR', 'MTEK', 'MTEKW', 'MTEM', 'MTEN', 'MTEX', 'MTG', 'MTH', 'MTLS', 'MTN', 'MTNB', 'MTR', 'MTRN', 'MTRX', 'MTSI', 'MTTR', 'MTUS', 'MTW', 'MTX', 'MTZ', 'MU', 'MUA', 'MUC', 'MUE', 'MUFG', 'MUI', 'MUJ', 'MULN', 'MUR', 'MURA', 'MUSA', 'MUX', 'MVBF', 'MVF', 'MVIS', 'MVO', 'MVST', 'MVSTW', 'MVT', 'MWA', 'MWG', 'MX', 'MXC', 'MXCT', 'MXE', 'MXF', 'MXL', 'MYD', 'MYE', 'MYFW', 'MYGN', 'MYI', 'MYN', 'MYNA', 'MYND', 'MYNZ', 'MYO', 'MYPS', 'MYRG', 'MYSZ', 'MYTE', 'N', 'NA', 'NAAS', 'NABL', 'NAC', 'NAD', 'NAII', 'NAK', 'NAMS', 'NAMSW', 'NAN', 'NAOV', 'NAPA', 'NARI', 'NAT', 'NATH', 'NATL', 'NATR', 'NAUT', 'NAVI', 'NAYA', 'NAZ', 'NB', 'NBB', 'NBBK', 'NBH', 'NBHC', 'NBIS', 'NBIX', 'NBN', 'NBR', 'NBTB', 'NBTX', 'NBXG', 'NBY', 'NC', 'NCA', 'NCDL', 'NCI', 'NCL', 'NCLH', 'NCMI', 'NCNA', 'NCNC', 'NCNCW', 'NCNO', 'NCPL', 'NCPLW', 'NCRA', 'NCSM', 'NCTY', 'NCV', 'NCZ', 'NDAQ', 'NDLS', 'NDMO', 'NDP', 'NDRA', 'NDSN', 'NE', 'NEA', 'NECB', 'NEE', 'NEGG', 'NEM', 'NEN', 'NEO', 'NEOG', 'NEON', 'NEOV', 'NEOVW', 'NEP', 'NEPH', 'NERV', 'NESR', 'NESRW', 'NET', 'NETD', 'NETDW', 'NEU', 'NEUE', 'NEWP', 'NEWT', 'NEWTG', 'NEWTH', 'NEWTI', 'NEWTZ', 'NEXA', 'NEXN', 'NEXT', 'NFBK', 'NFE', 'NFG', 'NFGC', 'NFJ', 'NFLX', 'NG', 'NGD', 'NGG', 'NGL', 'NGNE', 'NGS', 'NGVC', 'NGVT', 'NHC', 'NHI', 'NHS', 'NHTC', 'NI', 'NIC', 'NICE', 'NIE', 'NIM', 'NINE', 'NIO', 'NIOBW', 'NIPG', 'NISN', 'NITO', 'NIU', 'NIVF', 'NIVFW', 'NIXX', 'NIXXW', 'NJR', 'NKE', 'NKGN', 'NKGNW', 'NKLA', 'NKSH', 'NKTR', 'NKTX', 'NKX', 'NL', 'NLOP', 'NLSP', 'NLY', 'NMAI', 'NMCO', 'NMFC', 'NMFCZ', 'NMG', 'NMHI', 'NMHIW', 'NMI', 'NMIH', 'NML', 'NMM', 'NMR', 'NMRA', 'NMRK', 'NMS', 'NMT', 'NMTC', 'NMZ', 'NN', 'NNAVW', 'NNBR', 'NNDM', 'NNE', 'NNI', 'NNN', 'NNOX', 'NNVC', 'NNY', 'NOA', 'NOAH', 'NOC', 'NODK', 'NOG', 'NOK', 'NOM', 'NOMD', 'NOTE', 'NOTV', 'NOV', 'NOVA', 'NOVT', 'NOW', 'NPABU', 'NPCE', 'NPCT', 'NPFD', 'NPK', 'NPO', 'NPV', 'NPWR', 'NQP', 'NR', 'NRBO', 'NRC', 'NRDS', 'NRDY', 'NREF', 'NRG', 'NRGV', 'NRIM', 'NRIX', 'NRK', 'NRO', 'NRP', 'NRSN', 'NRSNW', 'NRT', 'NRUC', 'NRXP', 'NRXPW', 'NRXS', 'NSA', 'NSC', 'NSIT', 'NSP', 'NSPR', 'NSSC', 'NSTS', 'NSYS', 'NTAP', 'NTB', 'NTCT', 'NTES', 'NTG', 'NTGR', 'NTIC', 'NTIP', 'NTLA', 'NTNX', 'NTR', 'NTRA', 'NTRB', 'NTRBW', 'NTRP', 'NTRS', 'NTRSO', 'NTST', 'NTWK', 'NTWOU', 'NTZ', 'NU', 'NUE', 'NUKK', 'NUKKW', 'NURO', 'NUS', 'NUTX', 'NUV', 'NUVB', 'NUVL', 'NUW', 'NUWE', 'NVA', 'NVAC', 'NVAWW', 'NVAX', 'NVCR', 'NVCT', 'NVDA', 'NVEC', 'NVEE', 'NVEI', 'NVFY', 'NVG', 'NVGS', 'NVMI', 'NVNI', 'NVNIW', 'NVNO', 'NVO', 'NVOS', 'NVR', 'NVRI', 'NVRO', 'NVS', 'NVST', 'NVT', 'NVTS', 'NVVE', 'NVVEW', 'NVX', 'NWBI', 'NWE', 'NWFL', 'NWG', 'NWGL', 'NWL', 'NWN', 'NWPX', 'NWS', 'NWSA', 'NWTN', 'NWTNW', 'NX', 'NXC', 'NXDT', 'NXE', 'NXG', 'NXGL', 'NXGLW', 'NXJ', 'NXL', 'NXLIW', 'NXN', 'NXP', 'NXPI', 'NXPL', 'NXPLW', 'NXRT', 'NXST', 'NXT', 'NXTC', 'NXTT', 'NXU', 'NYAX', 'NYC', 'NYMT', 'NYMTI', 'NYMTL', 'NYMTM', 'NYMTN', 'NYMTZ', 'NYT', 'NYXH', 'NZF', 'O', 'OABI', 'OABIW', 'OACCU', 'OAK', 'OAKU', 'OAKUR', 'OAKUW', 'OB', 'OBDC', 'OBDE', 'OBE', 'OBIO', 'OBK', 'OBLG', 'OBT', 'OC', 'OCC', 'OCCI', 'OCCIM', 'OCCIN', 'OCCIO', 'OCEA', 'OCEAW', 'OCFC', 'OCFCP', 'OCFT', 'OCG', 'OCGN', 'OCS', 'OCSAW', 'OCSL', 'OCTO', 'OCUL', 'OCX', 'ODC', 'ODD', 'ODFL', 'ODP', 'ODV', 'ODVWZ', 'OEC', 'OESX', 'OFG', 'OFIX', 'OFLX', 'OFS', 'OFSSH', 'OGE', 'OGEN', 'OGI', 'OGN', 'OGS', 'OHI', 'OI', 'OIA', 'OII', 'OIS', 'OKE', 'OKLO', 'OKTA', 'OKUR', 'OKYO', 'OLB', 'OLED', 'OLLI', 'OLMA', 'OLN', 'OLO', 'OLP', 'OLPX', 'OM', 'OMAB', 'OMC', 'OMCC', 'OMCL', 'OMER', 'OMEX', 'OMF', 'OMGA', 'OMH', 'OMI', 'OMIC', 'ON', 'ONB', 'ONBPO', 'ONBPP', 'ONCO', 'ONCT', 'ONCY', 'ONDS', 'ONEW', 'ONFO', 'ONFOW', 'ONIT', 'ONL', 'ONMD', 'ONMDW', 'ONON', 'ONTF', 'ONTO', 'ONVO', 'ONYX', 'ONYXW', 'OOMA', 'OP', 'OPAD', 'OPAL', 'OPBK', 'OPCH', 'OPEN', 'OPFI', 'OPHC', 'OPI', 'OPINL', 'OPK', 'OPOF', 'OPP', 'OPRA', 'OPRT', 'OPRX', 'OPT', 'OPTN', 'OPTT', 'OPTX', 'OPTXW', 'OPXS', 'OPY', 'OR', 'ORA', 'ORC', 'ORCL', 'ORGN', 'ORGNW', 'ORGO', 'ORI', 'ORIC', 'ORIS', 'ORKA', 'ORKT', 'ORLA', 'ORLY', 'ORMP', 'ORN', 'ORRF', 'OS', 'OSBC', 'OSCR', 'OSIS', 'OSK', 'OSPN', 'OSS', 'OST', 'OSTX', 'OSUR', 'OSW', 'OTEX', 'OTIS', 'OTLK', 'OTLY', 'OTRK', 'OTTR', 'OUST', 'OUT', 'OVBC', 'OVID', 'OVLY', 'OVV', 'OWL', 'OWLT', 'OXBR', 'OXBRW', 'OXLC', 'OXLCI', 'OXLCL', 'OXLCN', 'OXLCO', 'OXLCP', 'OXLCZ', 'OXM', 'OXSQ', 'OXSQG', 'OXSQZ', 'OXY', 'OZ', 'OZK', 'OZKAP', 'P', 'PAA', 'PAAS', 'PAC', 'PACB', 'PACK', 'PACS', 'PAG', 'PAGP', 'PAGS', 'PAHC', 'PAI', 'PAL', 'PALI', 'PALT', 'PAM', 'PANL', 'PANW', 'PAPL', 'PAR', 'PARA', 'PARAA', 'PARR', 'PASG', 'PATH', 'PATK', 'PAVM', 'PAVMZ', 'PAVS', 'PAX', 'PAXS', 'PAY', 'PAYC', 'PAYO', 'PAYS', 'PAYX', 'PB', 'PBA', 'PBBK', 'PBF', 'PBFS', 'PBH', 'PBHC', 'PBI', 'PBM', 'PBMWW', 'PBPB', 'PBR', 'PBT', 'PBYI', 'PC', 'PCAR', 'PCB', 'PCF', 'PCG', 'PCH', 'PCK', 'PCM', 'PCN', 'PCOR', 'PCQ', 'PCRX', 'PCSA', 'PCT', 'PCTTU', 'PCTTW', 'PCTY', 'PCVX', 'PCYO', 'PD', 'PDCC', 'PDCO', 'PDD', 'PDEX', 'PDFS', 'PDI', 'PDLB', 'PDM', 'PDO', 'PDS', 'PDSB', 'PDT', 'PDX', 'PDYN', 'PDYNW', 'PEB', 'PEBK', 'PEBO', 'PECO', 'PED', 'PEG', 'PEGA', 'PEGY', 'PEN', 'PENG', 'PENN', 'PEO', 'PEP', 'PEPG', 'PERF', 'PERI', 'PESI', 'PET', 'PETS', 'PETWW', 'PETZ', 'PEV', 'PFBC', 'PFC', 'PFD', 'PFE', 'PFG', 'PFGC', 'PFH', 'PFIE', 'PFIS', 'PFL', 'PFLT', 'PFMT', 'PFN', 'PFO', 'PFS', 'PFSI', 'PFTA', 'PFTAU', 'PFTAW', 'PFX', 'PFXNZ', 'PG', 'PGC', 'PGEN', 'PGHL', 'PGNY', 'PGP', 'PGR', 'PGRE', 'PGRU', 'PGY', 'PGYWW', 'PGZ', 'PH', 'PHAR', 'PHAT', 'PHD', 'PHG', 'PHGE', 'PHI', 'PHIN', 'PHIO', 'PHK', 'PHM', 'PHR', 'PHT', 'PHUN', 'PHVS', 'PHX', 'PI', 'PII', 'PIII', 'PIIIW', 'PIK', 'PIM', 'PINC', 'PINE', 'PINS', 'PIPR', 'PIRS', 'PITA', 'PITAW', 'PJT', 'PK', 'PKBK', 'PKE', 'PKG', 'PKOH', 'PKST', 'PKX', 'PL', 'PLAB', 'PLAG', 'PLAO', 'PLAY', 'PLBC', 'PLBY', 'PLCE', 'PLD', 'PLG', 'PLL', 'PLMJ', 'PLMR', 'PLNT', 'PLOW', 'PLPC', 'PLRX', 'PLRZ', 'PLSE', 'PLTK', 'PLTR', 'PLUG', 'PLUR', 'PLUS', 'PLX', 'PLXS', 'PLYA', 'PLYM', 'PM', 'PMAX', 'PMCB', 'PMD', 'PMEC', 'PMF', 'PML', 'PMM', 'PMN', 'PMNT', 'PMO', 'PMT', 'PMTS', 'PMTU', 'PMVP', 'PMX', 'PNBK', 'PNC', 'PNF', 'PNFP', 'PNFPP', 'PNI', 'PNNT', 'PNR', 'PNRG', 'PNST', 'PNTG', 'PNW', 'POAI', 'POCI', 'PODC', 'PODD', 'POET', 'POLA', 'POLE', 'POLEU', 'POLEW', 'POOL', 'POR', 'POST', 'POWI', 'POWL', 'POWW', 'POWWP', 'PPBI', 'PPBT', 'PPC', 'PPG', 'PPIH', 'PPL', 'PPSI', 'PPT', 'PPTA', 'PPYA', 'PPYAU', 'PR', 'PRA', 'PRAA', 'PRAX', 'PRCH', 'PRCT', 'PRDO', 'PRE', 'PRENW', 'PRFX', 'PRG', 'PRGO', 'PRGS', 'PRH', 'PRI', 'PRIF', 'PRIM', 'PRK', 'PRKS', 'PRLB', 'PRLD', 'PRLH', 'PRLHW', 'PRM', 'PRME', 'PRMW', 'PRO', 'PROC', 'PROF', 'PROK', 'PROP', 'PROV', 'PRPH', 'PRPL', 'PRPO', 'PRQR', 'PRS', 'PRSO', 'PRT', 'PRTA', 'PRTC', 'PRTG', 'PRTH', 'PRTS', 'PRU', 'PRVA', 'PRZO', 'PSA', 'PSBD', 'PSEC', 'PSF', 'PSFE', 'PSHG', 'PSIG', 'PSMT', 'PSN', 'PSNL', 'PSNY', 'PSNYW', 'PSO', 'PSQH', 'PSTG', 'PSTL', 'PSTV', 'PSTX', 'PSX', 'PT', 'PTA', 'PTC', 'PTCT', 'PTEN', 'PTGX', 'PTHL', 'PTIX', 'PTLE', 'PTLO', 'PTMN', 'PTN', 'PTON', 'PTPI', 'PTSI', 'PTVE', 'PTY', 'PUBM', 'PUK', 'PULM', 'PUMP', 'PVBC', 'PVH', 'PVL', 'PW', 'PWM', 'PWOD', 'PWP', 'PWR', 'PWUP', 'PWUPU', 'PWUPW', 'PX', 'PXDT', 'PXLW', 'PXS', 'PXSAW', 'PYCR', 'PYN', 'PYPD', 'PYPL', 'PYT', 'PYXS', 'PZC', 'PZG', 'PZZA', 'Q', 'QBTS', 'QCOM', 'QCRH', 'QD', 'QDEL', 'QETA', 'QETAR', 'QETAU', 'QFIN', 'QGEN', 'QH', 'QIPT', 'QLGN', 'QLYS', 'QMCO', 'QMMM', 'QNCX', 'QNRX', 'QNST', 'QNTM', 'QQQX', 'QRHC', 'QRTEA', 'QRTEB', 'QRTEP', 'QRVO', 'QS', 'QSG', 'QSI', 'QSIAW', 'QSR', 'QTI', 'QTRX', 'QTTB', 'QTWO', 'QUAD', 'QUBT', 'QUIK', 'QURE', 'QVCC', 'QVCD', 'QXO', 'R', 'RA', 'RACE', 'RAIL', 'RAMP', 'RAND', 'RANI', 'RAPP', 'RAPT', 'RARE', 'RAVE', 'RAY', 'RAYA', 'RBA', 'RBB', 'RBBN', 'RBC', 'RBCAA', 'RBKB', 'RBLX', 'RBOT', 'RBRK', 'RC', 'RCAT', 'RCB', 'RCC', 'RCEL', 'RCFA', 'RCG', 'RCI', 'RCKT', 'RCKTW', 'RCKY', 'RCL', 'RCM', 'RCMT', 'RCON', 'RCS', 'RCUS', 'RDACU', 'RDCM', 'RDDT', 'RDFN', 'RDHL', 'RDI', 'RDIB', 'RDN', 'RDNT', 'RDUS', 'RDVT', 'RDW', 'RDWR', 'RDY', 'RDZN', 'RDZNW', 'REAL', 'REAX', 'REBN', 'RECT', 'REE', 'REFI', 'REFR', 'REG', 'REGCO', 'REGCP', 'REGN', 'REI', 'REKR', 'RELI', 'RELIW', 'RELL', 'RELX', 'RELY', 'RENB', 'RENE', 'RENEW', 'RENT', 'REPL', 'REPX', 'RERE', 'RES', 'RETO', 'REVB', 'REVBW', 'REVG', 'REX', 'REXR', 'REYN', 'REZI', 'RF', 'RFAC', 'RFACW', 'RFAI', 'RFAIR', 'RFAIU', 'RFI', 'RFIL', 'RFL', 'RFM', 'RFMZ', 'RGA', 'RGC', 'RGCO', 'RGEN', 'RGF', 'RGLD', 'RGLS', 'RGNX', 'RGP', 'RGR', 'RGS', 'RGT', 'RGTI', 'RGTIW', 'RH', 'RHE', 'RHI', 'RHP', 'RICK', 'RIG', 'RIGL', 'RILY', 'RILYG', 'RILYK', 'RILYL', 'RILYM', 'RILYN', 'RILYP', 'RILYT', 'RILYZ', 'RIME', 'RIO', 'RIOT', 'RITM', 'RITR', 'RIV', 'RIVN', 'RJF', 'RKDA', 'RKLB', 'RKT', 'RL', 'RLAY', 'RLGT', 'RLI', 'RLJ', 'RLMD', 'RLTY', 'RLX', 'RLYB', 'RM', 'RMAX', 'RMBI', 'RMBL', 'RMBS', 'RMCF', 'RMCO', 'RMCOW', 'RMD', 'RMI', 'RMM', 'RMMZ', 'RMNI', 'RMR', 'RMT', 'RMTI', 'RNA', 'RNAC', 'RNAZ', 'RNG', 'RNGR', 'RNP', 'RNR', 'RNST', 'RNW', 'RNWWW', 'RNXT', 'ROAD', 'ROCK', 'ROCL', 'ROCLU', 'ROCLW', 'ROG', 'ROIC', 'ROIV', 'ROK', 'ROKU', 'ROL', 'ROLR', 'ROMA', 'ROOT', 'ROP', 'ROST', 'RPAY', 'RPD', 'RPID', 'RPM', 'RPRX', 'RPTX', 'RQI', 'RR', 'RRAC', 'RRBI', 'RRC', 'RRGB', 'RRR', 'RRX', 'RS', 'RSF', 'RSG', 'RSI', 'RSKD', 'RSLS', 'RSSS', 'RSVR', 'RSVRW', 'RTC', 'RTO', 'RTX', 'RUM', 'RUMBW', 'RUN', 'RUSHA', 'RUSHB', 'RVLV', 'RVMD', 'RVMDW', 'RVNC', 'RVP', 'RVPH', 'RVPHW', 'RVSB', 'RVSN', 'RVSNW', 'RVT', 'RVTY', 'RVYL', 'RWAY', 'RWAYL', 'RWAYZ', 'RWT', 'RWTN', 'RWTO', 'RXO', 'RXRX', 'RXST', 'RXT', 'RY', 'RYAAY', 'RYAM', 'RYAN', 'RYDE', 'RYI', 'RYN', 'RYTM', 'RZB', 'RZC', 'RZLT', 'RZLV', 'RZLVW', 'S', 'SA', 'SABA', 'SABR', 'SABS', 'SABSW', 'SACC', 'SACH', 'SAFE', 'SAFT', 'SAG', 'SAGE', 'SAH', 'SAIA', 'SAIC', 'SAIH', 'SAIHW', 'SAJ', 'SAM', 'SAMG', 'SAN', 'SANA', 'SAND', 'SANG', 'SANM', 'SANW', 'SAP', 'SAR', 'SARO', 'SASR', 'SAT', 'SATL', 'SATLW', 'SATS', 'SATX', 'SAVA', 'SAVE', 'SAY', 'SAZ', 'SB', 'SBAC', 'SBBA', 'SBC', 'SBCF', 'SBCWW', 'SBET', 'SBEV', 'SBFG', 'SBFM', 'SBFMW', 'SBGI', 'SBH', 'SBI', 'SBLK', 'SBR', 'SBRA', 'SBS', 'SBSI', 'SBSW', 'SBT', 'SBUX', 'SBXD', 'SCCC', 'SCCD', 'SCCE', 'SCCF', 'SCCG', 'SCCO', 'SCD', 'SCE', 'SCHL', 'SCHW', 'SCI', 'SCKT', 'SCL', 'SCLX', 'SCLXW', 'SCM', 'SCNI', 'SCNX', 'SCOR', 'SCPH', 'SCPX', 'SCS', 'SCSC', 'SCVL', 'SCWO', 'SCWX', 'SCYX', 'SD', 'SDA', 'SDAWW', 'SDGR', 'SDHC', 'SDHY', 'SDIG', 'SDOT', 'SDRL', 'SDST', 'SDSTW', 'SE', 'SEAL', 'SEAT', 'SEATW', 'SEB', 'SEDA', 'SEDG', 'SEE', 'SEED', 'SEER', 'SEG', 'SEI', 'SEIC', 'SELF', 'SELX', 'SEM', 'SEMR', 'SENEA', 'SENEB', 'SENS', 'SEPN', 'SER', 'SERA', 'SERV', 'SES', 'SEVN', 'SEZL', 'SF', 'SFB', 'SFBC', 'SFBS', 'SFHG', 'SFIX', 'SFL', 'SFM', 'SFNC', 'SFST', 'SFWL', 'SG', 'SGA', 'SGBX', 'SGC', 'SGD', 'SGHC', 'SGHT', 'SGLY', 'SGMA', 'SGML', 'SGMO', 'SGMT', 'SGN', 'SGRP', 'SGRY', 'SGU', 'SHAK', 'SHBI', 'SHC', 'SHCO', 'SHEL', 'SHEN', 'SHFS', 'SHFSW', 'SHG', 'SHIM', 'SHIP', 'SHLS', 'SHLT', 'SHMD', 'SHMDW', 'SHO', 'SHOO', 'SHOP', 'SHOT', 'SHOTW', 'SHPH', 'SHW', 'SHYF', 'SIBN', 'SID', 'SIDU', 'SIEB', 'SIF', 'SIFY', 'SIG', 'SIGA', 'SIGI', 'SIGIP', 'SII', 'SILA', 'SILC', 'SILO', 'SILV', 'SIM', 'SIMA', 'SIMAU', 'SIMAW', 'SIMO', 'SINT', 'SIRI', 'SISI', 'SITC', 'SITE', 'SITM', 'SJ', 'SJM', 'SJT', 'SJW', 'SKE', 'SKGR', 'SKGRW', 'SKIL', 'SKIN', 'SKK', 'SKLZ', 'SKM', 'SKT', 'SKWD', 'SKX', 'SKY', 'SKYE', 'SKYH', 'SKYQ', 'SKYT', 'SKYW', 'SKYX', 'SLAB', 'SLB', 'SLDB', 'SLDP', 'SLDPW', 'SLE', 'SLF', 'SLG', 'SLGL', 'SLGN', 'SLI', 'SLM', 'SLMBP', 'SLN', 'SLND', 'SLNG', 'SLNH', 'SLNHP', 'SLNO', 'SLP', 'SLQT', 'SLRC', 'SLRN', 'SLRX', 'SLS', 'SLSR', 'SLVM', 'SLXN', 'SLXNW', 'SM', 'SMAR', 'SMBC', 'SMBK', 'SMC', 'SMCI', 'SMFG', 'SMG', 'SMHI', 'SMID', 'SMLR', 'SMMT', 'SMP', 'SMPL', 'SMR', 'SMRT', 'SMSI', 'SMTC', 'SMTI', 'SMTK', 'SMWB', 'SMX', 'SMXT', 'SMXWW', 'SN', 'SNA', 'SNAL', 'SNAP', 'SNAX', 'SNAXW', 'SNBR', 'SNCR', 'SNCRL', 'SNCY', 'SND', 'SNDA', 'SNDL', 'SNDR', 'SNDX', 'SNES', 'SNEX', 'SNFCA', 'SNGX', 'SNN', 'SNOA', 'SNOW', 'SNPS', 'SNPX', 'SNSE', 'SNT', 'SNTG', 'SNTI', 'SNV', 'SNX', 'SNY', 'SNYR', 'SO', 'SOAR', 'SOBO', 'SOBR', 'SOC', 'SOFI', 'SOGP', 'SOHO', 'SOHOB', 'SOHON', 'SOHOO', 'SOHU', 'SOJC', 'SOJD', 'SOJE', 'SOL', 'SOLV', 'SON', 'SOND', 'SONDW', 'SONM', 'SONN', 'SONO', 'SONY', 'SOPA', 'SOPH', 'SOR', 'SOS', 'SOTK', 'SOUN', 'SOUNW', 'SOWG', 'SPAI', 'SPB', 'SPCB', 'SPCE', 'SPE', 'SPFI', 'SPG', 'SPGC', 'SPGI', 'SPH', 'SPHL', 'SPHR', 'SPI', 'SPIR', 'SPKL', 'SPKLU', 'SPKLW', 'SPLP', 'SPMC', 'SPNS', 'SPNT', 'SPOK', 'SPOT', 'SPPL', 'SPR', 'SPRB', 'SPRC', 'SPRO', 'SPRU', 'SPRY', 'SPSC', 'SPT', 'SPTN', 'SPWH', 'SPXC', 'SPXX', 'SQ', 'SQFT', 'SQFTP', 'SQFTW', 'SQM', 'SQNS', 'SR', 'SRAD', 'SRBK', 'SRCE', 'SRCL', 'SRDX', 'SRE', 'SREA', 'SRFM', 'SRG', 'SRI', 'SRL', 'SRM', 'SRPT', 'SRRK', 'SRTS', 'SRV', 'SRZN', 'SRZNW', 'SSB', 'SSBI', 'SSBK', 'SSD', 'SSKN', 'SSL', 'SSNC', 'SSP', 'SSRM', 'SSSS', 'SSSSL', 'SST', 'SSTI', 'SSTK', 'SSY', 'SSYS', 'ST', 'STAA', 'STAF', 'STAG', 'STBA', 'STBX', 'STC', 'STCN', 'STE', 'STEC', 'STEL', 'STEM', 'STEP', 'STEW', 'STFS', 'STG', 'STGW', 'STHO', 'STI', 'STIM', 'STK', 'STKH', 'STKL', 'STKS', 'STLA', 'STLD', 'STM', 'STN', 'STNE', 'STNG', 'STOK', 'STR', 'STRA', 'STRL', 'STRM', 'STRO', 'STRR', 'STRRP', 'STRS', 'STRT', 'STRW', 'STSS', 'STSSW', 'STT', 'STTK', 'STVN', 'STWD', 'STX', 'STXS', 'STZ', 'SU', 'SUGP', 'SUI', 'SUM', 'SUN', 'SUNS', 'SUP', 'SUPN', 'SUPV', 'SURG', 'SURGW', 'SUUN', 'SUZ', 'SVC', 'SVCO', 'SVII', 'SVIIU', 'SVIIW', 'SVM', 'SVMH', 'SVMHW', 'SVRA', 'SVRE', 'SVREW', 'SVT', 'SVV', 'SW', 'SWAG', 'SWAGW', 'SWBI', 'SWI', 'SWIM', 'SWIN', 'SWK', 'SWKH', 'SWKHL', 'SWKS', 'SWTX', 'SWVL', 'SWVLW', 'SWX', 'SWZ', 'SXC', 'SXI', 'SXT', 'SXTC', 'SXTP', 'SXTPW', 'SY', 'SYBT', 'SYBX', 'SYF', 'SYK', 'SYM', 'SYNA', 'SYNX', 'SYPR', 'SYRA', 'SYRE', 'SYRS', 'SYT', 'SYTA', 'SYTAW', 'SYY', 'T', 'TAC', 'TACT', 'TAIT', 'TAK', 'TAL', 'TALK', 'TALKW', 'TALO', 'TANH', 'TAOP', 'TAP', 'TARA', 'TARS', 'TASK', 'TATT', 'TAYD', 'TBB', 'TBBB', 'TBBK', 'TBC', 'TBI', 'TBLA', 'TBLAW', 'TBLD', 'TBMC', 'TBMCR', 'TBN', 'TBNK', 'TBPH', 'TBRG', 'TC', 'TCBI', 'TCBIO', 'TCBK', 'TCBP', 'TCBPW', 'TCBS', 'TCBX', 'TCI', 'TCMD', 'TCOM', 'TCPC', 'TCRT', 'TCRX', 'TCS', 'TCTM', 'TCX', 'TD', 'TDC', 'TDF', 'TDG', 'TDOC', 'TDS', 'TDTH', 'TDUP', 'TDW', 'TDY', 'TEAF', 'TEAM', 'TECH', 'TECK', 'TECTP', 'TECX', 'TEF', 'TEI', 'TEL', 'TELA', 'TELO', 'TEM', 'TEN', 'TENB', 'TENX', 'TEO', 'TER', 'TERN', 'TETE', 'TETEU', 'TEVA', 'TEX', 'TFC', 'TFFP', 'TFII', 'TFIN', 'TFINP', 'TFPM', 'TFSA', 'TFSL', 'TFX', 'TG', 'TGAA', 'TGAAW', 'TGB', 'TGI', 'TGL', 'TGLS', 'TGNA', 'TGS', 'TGT', 'TGTX', 'TH', 'THAR', 'THC', 'THCH', 'THCP', 'THCPW', 'THFF', 'THG', 'THM', 'THO', 'THQ', 'THR', 'THRD', 'THRM', 'THRY', 'THS', 'THTX', 'THW', 'TIGO', 'TIGR', 'TIL', 'TILE', 'TIMB', 'TIPT', 'TIRX', 'TISI', 'TITN', 'TIVC', 'TIXT', 'TJX', 'TK', 'TKC', 'TKLF', 'TKNO', 'TKO', 'TKR', 'TLF', 'TLGY', 'TLGYU', 'TLK', 'TLN', 'TLPH', 'TLRY', 'TLS', 'TLSA', 'TLSI', 'TLSIW', 'TLYS', 'TM', 'TMC', 'TMCI', 'TMCWW', 'TMDX', 'TME', 'TMHC', 'TMO', 'TMP', 'TMQ', 'TMTC', 'TMTCR', 'TMTCU', 'TMUS', 'TNC', 'TNDM', 'TNET', 'TNFA', 'TNGX', 'TNK', 'TNL', 'TNON', 'TNONW', 'TNXP', 'TNYA', 'TOI', 'TOIIW', 'TOL', 'TOMZ', 'TOON', 'TOP', 'TOPS', 'TORO', 'TOST', 'TOUR', 'TOVX', 'TOWN', 'TOYO', 'TPB', 'TPC', 'TPCS', 'TPET', 'TPG', 'TPGXL', 'TPH', 'TPIC', 'TPL', 'TPR', 'TPST', 'TPTA', 'TPVG', 'TPX', 'TPZ', 'TR', 'TRAK', 'TRAW', 'TRC', 'TRDA', 'TREE', 'TREX', 'TRGP', 'TRI', 'TRIB', 'TRIN', 'TRINI', 'TRINL', 'TRINZ', 'TRIP', 'TRMB', 'TRMD', 'TRMK', 'TRML', 'TRN', 'TRNO', 'TRNR', 'TRNS', 'TROO', 'TROW', 'TROX', 'TRP', 'TRS', 'TRSG', 'TRST', 'TRT', 'TRTN', 'TRTX', 'TRU', 'TRUE', 'TRUG', 'TRUP', 'TRV', 'TRVG', 'TRVI', 'TRX', 'TS', 'TSAT', 'TSBK', 'TSBX', 'TSCO', 'TSE', 'TSEM', 'TSHA', 'TSI', 'TSLA', 'TSLX', 'TSM', 'TSN', 'TSQ', 'TSVT', 'TT', 'TTC', 'TTD', 'TTE', 'TTEC', 'TTEK', 'TTGT', 'TTI', 'TTMI', 'TTNP', 'TTOO', 'TTP', 'TTSH', 'TTWO', 'TU', 'TURB', 'TURN', 'TUSK', 'TUYA', 'TV', 'TVC', 'TVE', 'TVGN', 'TVGNW', 'TVTX', 'TW', 'TWFG', 'TWG', 'TWI', 'TWIN', 'TWKS', 'TWLO', 'TWN', 'TWO', 'TWST', 'TX', 'TXG', 'TXMD', 'TXN', 'TXNM', 'TXO', 'TXRH', 'TXT', 'TY', 'TYG', 'TYGO', 'TYL', 'TYRA', 'TZOO', 'TZUP', 'U', 'UA', 'UAA', 'UAL', 'UAMY', 'UAN', 'UAVS', 'UBCP', 'UBER', 'UBFO', 'UBS', 'UBSI', 'UBX', 'UBXG', 'UCAR', 'UCB', 'UCL', 'UCTT', 'UDMY', 'UDR', 'UE', 'UEC', 'UEIC', 'UFCS', 'UFI', 'UFPI', 'UFPT', 'UG', 'UGI', 'UGP', 'UGRO', 'UHAL', 'UHG', 'UHGWW', 'UHS', 'UHT', 'UI', 'UIS', 'UK', 'UKOMW', 'UL', 'ULBI', 'ULCC', 'ULH', 'ULS', 'ULTA', 'ULY', 'UMAC', 'UMBF', 'UMC', 'UMH', 'UNB', 'UNCY', 'UNF', 'UNFI', 'UNH', 'UNIT', 'UNM', 'UNMA', 'UNP', 'UNTY', 'UONE', 'UONEK', 'UP', 'UPB', 'UPBD', 'UPC', 'UPLD', 'UPS', 'UPST', 'UPWK', 'UPXI', 'URBN', 'URG', 'URGN', 'URI', 'UROY', 'USA', 'USAC', 'USAP', 'USAS', 'USAU', 'USB', 'USCB', 'USEA', 'USEG', 'USFD', 'USGO', 'USGOW', 'USIO', 'USLM', 'USM', 'USNA', 'USPH', 'UTF', 'UTG', 'UTHR', 'UTI', 'UTL', 'UTMD', 'UTSI', 'UTZ', 'UUU', 'UUUU', 'UVE', 'UVSP', 'UVV', 'UWMC', 'UXIN', 'UZD', 'UZE', 'UZF', 'V', 'VABK', 'VAC', 'VACH', 'VACHU', 'VACHW', 'VAL', 'VALE', 'VALN', 'VALU', 'VANI', 'VATE', 'VBF', 'VBFC', 'VBNK', 'VBTX', 'VC', 'VCEL', 'VCIC', 'VCICU', 'VCICW', 'VCIG', 'VCNX', 'VCSA', 'VCTR', 'VCV', 'VCYT', 'VECO', 'VEEA', 'VEEE', 'VEEV', 'VEL', 'VEON', 'VERA', 'VERB', 'VERI', 'VERO', 'VERU', 'VERV', 'VERX', 'VET', 'VFC', 'VFF', 'VFL', 'VFS', 'VFSWW', 'VGAS', 'VGASW', 'VGI', 'VGM', 'VGZ', 'VHC', 'VHI', 'VIASP', 'VIAV', 'VICI', 'VICR', 'VIGL', 'VIK', 'VINC', 'VINE', 'VINO', 'VINP', 'VIOT', 'VIPS', 'VIR', 'VIRC', 'VIRT', 'VIRX', 'VISL', 'VIST', 'VITL', 'VIV', 'VIVK', 'VKI', 'VKQ', 'VKTX', 'VLCN', 'VLGEA', 'VLN', 'VLO', 'VLRS', 'VLT', 'VLTO', 'VLY', 'VLYPN', 'VLYPO', 'VLYPP', 'VMAR', 'VMC', 'VMCA', 'VMCAW', 'VMD', 'VMEO', 'VMI', 'VMO', 'VNCE', 'VNDA', 'VNET', 'VNO', 'VNOM', 'VNRX', 'VNT', 'VOC', 'VOD', 'VOR', 'VOXR', 'VOXX', 'VOYA', 'VPG', 'VPV', 'VRA', 'VRAR', 'VRAX', 'VRCA', 'VRDN', 'VRE', 'VREX', 'VRM', 'VRME', 'VRMEW', 'VRN', 'VRNA', 'VRNS', 'VRNT', 'VRPX', 'VRRM', 'VRSK', 'VRSN', 'VRT', 'VRTS', 'VRTX', 'VS', 'VSAT', 'VSCO', 'VSEC', 'VSEE', 'VSEEW', 'VSH', 'VSME', 'VSSYW', 'VST', 'VSTA', 'VSTE', 'VSTEW', 'VSTM', 'VSTO', 'VSTS', 'VTAK', 'VTEX', 'VTGN', 'VTLE', 'VTMX', 'VTN', 'VTOL', 'VTR', 'VTRS', 'VTS', 'VTSI', 'VTVT', 'VTYX', 'VUZI', 'VVI', 'VVOS', 'VVPR', 'VVR', 'VVV', 'VVX', 'VXRT', 'VYGR', 'VYNE', 'VYX', 'VZIO', 'VZLA', 'W', 'WAB', 'WABC', 'WAFD', 'WAFDP', 'WAFU', 'WAI', 'WAL', 'WALD', 'WALDW', 'WASH', 'WAT', 'WATT', 'WAVE', 'WAVS', 'WAY', 'WB', 'WBA', 'WBD', 'WBS', 'WBTN', 'WBUY', 'WBX', 'WCC', 'WCN', 'WCT', 'WD', 'WDAY', 'WDC', 'WDFC', 'WDH', 'WDI', 'WDS', 'WEA', 'WEAV', 'WEC', 'WEL', 'WELL', 'WEN', 'WERN', 'WES', 'WEST', 'WETH', 'WEX', 'WEYS', 'WF', 'WFC', 'WFCF', 'WFG', 'WFRD', 'WGO', 'WGS', 'WGSWW', 'WH', 'WHD', 'WHF', 'WHFCL', 'WHG', 'WHLM', 'WHLR', 'WHLRD', 'WHLRP', 'WHR', 'WIA', 'WILC', 'WIMI', 'WINA', 'WING', 'WINT', 'WINVR', 'WINVW', 'WISA', 'WIT', 'WIW', 'WIX', 'WK', 'WKC', 'WKEY', 'WKHS', 'WKSP', 'WLDN', 'WLDS', 'WLDSW', 'WLFC', 'WLGS', 'WLK', 'WLKP', 'WLY', 'WLYB', 'WM', 'WMB', 'WMG', 'WMK', 'WMPN', 'WMS', 'WMT', 'WNC', 'WNEB', 'WNS', 'WNW', 'WOK', 'WOLF', 'WOOF', 'WOR', 'WORX', 'WOW', 'WPC', 'WPM', 'WPP', 'WPRT', 'WRAP', 'WRB', 'WRBY', 'WRD', 'WRLD', 'WRN', 'WS', 'WSBC', 'WSBCP', 'WSBF', 'WSC', 'WSFS', 'WSM', 'WSO', 'WSR', 'WST', 'WT', 'WTBA', 'WTFC', 'WTFCM', 'WTFCP', 'WTI', 'WTM', 'WTMA', 'WTO', 'WTRG', 'WTS', 'WTTR', 'WTW', 'WU', 'WULF', 'WVE', 'WVVI', 'WVVIP', 'WW', 'WWD', 'WWR', 'WWW', 'WY', 'WYNN', 'WYY', 'X', 'XAIR', 'XBIO', 'XBIT', 'XBP', 'XBPEW', 'XCH', 'XCUR', 'XEL', 'XELA', 'XELAP', 'XELB', 'XENE', 'XERS', 'XFLT', 'XFOR', 'XGN', 'XHG', 'XHR', 'XIN', 'XLO', 'XMTR', 'XNCR', 'XNET', 'XOM', 'XOMA', 'XOMAO', 'XOMAP', 'XOS', 'XOSWW', 'XP', 'XPEL', 'XPER', 'XPEV', 'XPL', 'XPO', 'XPOF', 'XPON', 'XPRO', 'XRAY', 'XRTX', 'XRX', 'XTIA', 'XTKG', 'XTLB', 'XTNT', 'XWEL', 'XXII', 'XYF', 'XYL', 'XYLO', 'Y', 'YALA', 'YCBD', 'YELP', 'YETI', 'YEXT', 'YGMZ', 'YHGJ', 'YHNAU', 'YI', 'YIBO', 'YJ', 'YMAB', 'YMM', 'YORW', 'YOSH', 'YOTA', 'YOTAR', 'YOTAU', 'YOU', 'YPF', 'YQ', 'YRD', 'YSG', 'YTRA', 'YUM', 'YUMC', 'YXT', 'YY', 'YYAI', 'YYGH', 'Z', 'ZAPP', 'ZAPPW', 'ZBAO', 'ZBH', 'ZBIO', 'ZBRA', 'ZCAR', 'ZCARW', 'ZCMD', 'ZD', 'ZDGE', 'ZENA', 'ZENV', 'ZEO', 'ZEOWW', 'ZEPP', 'ZETA', 'ZEUS', 'ZG', 'ZGN', 'ZH', 'ZI', 'ZIM', 'ZIMV', 'ZION', 'ZIONL', 'ZIONO', 'ZIONP', 'ZIP', 'ZJK', 'ZJYL', 'ZK', 'ZKH', 'ZKIN', 'ZLAB', 'ZM', 'ZNTL', 'ZOM', 'ZONE', 'ZOOZ', 'ZOOZW', 'ZS', 'ZTEK', 'ZTO', 'ZTR', 'ZTS', 'ZUMZ', 'ZUO', 'ZURA', 'ZVIA', 'ZVRA', 'ZVSA', 'ZWS', 'ZYME', 'ZYXI']
        selected_stock = st.selectbox("Select Stock:", stock_options)
        
        # Date range selection
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", 
                                     pd.to_datetime('2023-01-01'))
        with col2:
            end_date = st.date_input("End Date", 
                                    pd.to_datetime('today'))
        
        try:
            # Fetch stock data
            df = yf.download(selected_stock, start=start_date, end=end_date)
            if df.empty:
                st.error("No data available for the selected stock and date range.")
                return
            df.index = pd.to_datetime(df.index)
            value_col = 'Close'  # Use closing price by default
        except Exception as e:
            st.error(f"Error fetching stock data: {str(e)}")
            return
    else:
        # For uploaded data
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) == 0:
            st.warning("No numerical columns found in the dataset.")
            return
        value_col = st.selectbox("Select column for analysis:", numerical_cols)
    
    # Create tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs([
        "Time Series Components",
        "Stationarity Analysis",
        "Correlation Analysis",
        "Forecasting"
    ])
    
    with tab1:
        show_time_series_components(df, value_col)
    
    with tab2:
        show_stationarity_analysis(df, value_col)
        
    with tab3:
        show_correlation_analysis_ts(df, value_col)
        
    with tab4:
        show_forecasting(df, value_col)

def show_time_series_components(df, value_col):
    st.subheader("Time Series Components")
    
    # Original series plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df[value_col],
                            mode='lines', name='Original Series'))
    fig.update_layout(title='Original Time Series',
                     xaxis_title='Date',
                     yaxis_title='Value')
    st.plotly_chart(fig)
    
    try:
        # Decomposition
        decomposition = seasonal_decompose(df[value_col], period=30)
        
        # Trend
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(x=df.index,
                                     y=decomposition.trend,
                                     mode='lines', name='Trend'))
        fig_trend.update_layout(title='Trend Component')
        st.plotly_chart(fig_trend)
        
        # Seasonal
        fig_seasonal = go.Figure()
        fig_seasonal.add_trace(go.Scatter(x=df.index,
                                        y=decomposition.seasonal,
                                        mode='lines', name='Seasonal'))
        fig_seasonal.update_layout(title='Seasonal Component')
        st.plotly_chart(fig_seasonal)
        
        # Residual
        fig_residual = go.Figure()
        fig_residual.add_trace(go.Scatter(x=df.index,
                                        y=decomposition.resid,
                                        mode='lines', name='Residual'))
        fig_residual.update_layout(title='Residual Component')
        st.plotly_chart(fig_residual)
    except Exception as e:
        st.warning(f"Could not perform decomposition: {str(e)}")

def show_stationarity_analysis(df, value_col):
    st.subheader("Stationarity Analysis")
    
    # Perform ADF test
    series = df[value_col].dropna()
    result = adfuller(series)
    
    # Display ADF test results
    st.write("Augmented Dickey-Fuller Test Results:")
    st.write(f'ADF Statistic: {result[0]:.4f}')
    st.write(f'p-value: {result[1]:.4f}')
    st.write("Critical Values:")
    for key, value in result[4].items():
        st.write(f'\t{key}: {value:.4f}')
        
    # Interpretation
    if result[1] < 0.05:
        st.success("The series is stationary (p-value < 0.05)")
    else:
        st.warning("The series is non-stationary (p-value > 0.05)")
    
    # Rolling statistics
    window_size = st.slider("Select window size for rolling statistics:", 
                          min_value=5, max_value=50, value=20)
    
    rolling_mean = df[value_col].rolling(window=window_size).mean()
    rolling_std = df[value_col].rolling(window=window_size).std()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df[value_col],
                            mode='lines', name='Original'))
    fig.add_trace(go.Scatter(x=df.index, y=rolling_mean,
                            mode='lines', name='Rolling Mean'))
    fig.add_trace(go.Scatter(x=df.index, y=rolling_std,
                            mode='lines', name='Rolling Std'))
    fig.update_layout(title='Rolling Statistics',
                     xaxis_title='Date',
                     yaxis_title='Value')
    st.plotly_chart(fig)

def show_correlation_analysis_ts(df, value_col):
    st.subheader("Time Series Correlation Analysis")
    
    # Calculate ACF and PACF
    series = df[value_col].dropna()
    max_lags = st.slider("Select maximum lags:", 
                        min_value=10, max_value=50, value=40)
    
    acf_values = acf(series, nlags=max_lags)
    pacf_values = pacf(series, nlags=max_lags)
    
    # Plot ACF
    fig_acf = go.Figure()
    fig_acf.add_trace(go.Bar(x=list(range(len(acf_values))),
                            y=acf_values,
                            name='ACF'))
    fig_acf.add_hline(y=1.96/np.sqrt(len(series)), line_dash="dash", line_color="red")
    fig_acf.add_hline(y=-1.96/np.sqrt(len(series)), line_dash="dash", line_color="red")
    fig_acf.update_layout(title='Autocorrelation Function (ACF)',
                         xaxis_title='Lag',
                         yaxis_title='Correlation')
    st.plotly_chart(fig_acf)
    
    # Plot PACF
    fig_pacf = go.Figure()
    fig_pacf.add_trace(go.Bar(x=list(range(len(pacf_values))),
                             y=pacf_values,
                             name='PACF'))
    fig_pacf.add_hline(y=1.96/np.sqrt(len(series)), line_dash="dash", line_color="red")
    fig_pacf.add_hline(y=-1.96/np.sqrt(len(series)), line_dash="dash", line_color="red")
    fig_pacf.update_layout(title='Partial Autocorrelation Function (PACF)',
                          xaxis_title='Lag',
                          yaxis_title='Correlation')
    st.plotly_chart(fig_pacf)

def show_forecasting(df, value_col):
    st.subheader("ARIMA Forecasting")
    
    # ARIMA parameters selection
    col1, col2, col3 = st.columns(3)
    with col1:
        p = st.slider("Select p (AR order)", 0, 5, 1)
    with col2:
        d = st.slider("Select d (Difference order)", 0, 2, 1)
    with col3:
        q = st.slider("Select q (MA order)", 0, 5, 1)
    
    # Forecast horizon
    forecast_steps = st.slider("Select forecast horizon (days)", 
                             1, 30, 10)
    
    try:
        # Fit ARIMA model
        model = ARIMA(df[value_col], order=(p, d, q))
        model_fit = model.fit()
        
        # Make forecast
        forecast = model_fit.forecast(steps=forecast_steps)
        forecast_index = pd.date_range(df.index[-1], 
                                     periods=forecast_steps+1)[1:]
        
        # Plot results
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df[value_col],
                                mode='lines', name='Actual'))
        fig.add_trace(go.Scatter(x=forecast_index, y=forecast,
                                mode='lines', name='Forecast',
                                line=dict(dash='dash')))
        fig.update_layout(title='ARIMA Forecast',
                         xaxis_title='Date',
                         yaxis_title='Value')
        st.plotly_chart(fig)
        
        # Show model summary
        with st.expander("Show Model Summary"):
            st.text(str(model_fit.summary()))
            
        # Show forecast values
        st.subheader("Forecast Values")
        forecast_df = pd.DataFrame({
            'Date': forecast_index,
            'Forecast': forecast
        })
        st.write(forecast_df)
        
    except Exception as e:
        st.error(f"Error in forecasting: {str(e)}")

if __name__ == "__main__":
    main()

