import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import anthropic
import base64
from io import BytesIO
import os
import json

# ==================== CONFIG ====================
st.set_page_config(
    page_title="FORD 7 Dashboard",
    page_icon="🚛",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "data_df" not in st.session_state:
    st.session_state.data_df = None
if "refresh_data" not in st.session_state:
    st.session_state.refresh_data = True

# ==================== STYLES ====================
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .income-color { color: #2ecc71; }
    .expense-color { color: #e74c3c; }
    .neutral-color { color: #3498db; }
    </style>
""", unsafe_allow_html=True)

# ==================== DATA MANAGEMENT ====================
def load_data():
    """Load data from CSV file or create new one"""
    if os.path.exists("ford7_expenses.csv"):
        df = pd.read_csv("ford7_expenses.csv")
        df['fecha'] = pd.to_datetime(df['fecha'])
        return df.sort_values('fecha')
    else:
        return pd.DataFrame(columns=[
            'fecha', 'camion', 'categoria', 'descripcion', 
            'monto', 'tipo', 'conductor', 'saldo', 'foto_path'
        ])

def save_data(df):
    """Save data to CSV"""
    df_save = df.copy()
    df_save['fecha'] = df_save['fecha'].dt.strftime('%Y-%m-%d')
    df_save.to_csv("ford7_expenses.csv", index=False)

def calculate_balance(df):
    """Calculate running balance"""
    df = df.sort_values('fecha').reset_index(drop=True)
    balance = 0
    saldos = []
    
    for idx, row in df.iterrows():
        if row['tipo'] == 'ingreso':
            balance += row['monto']
        else:
            balance -= row['monto']
        saldos.append(balance)
    
    df['saldo'] = saldos
    return df

def image_to_base64(image_file):
    """Convert image to base64 for API"""
    return base64.standard_b64encode(image_file.read()).decode("utf-8")

# ==================== OCR WITH CLAUDE ====================
def extract_receipt_data(image_file):
    """Use Claude vision to extract receipt data"""
    
    client = anthropic.Anthropic(api_key=st.secrets.get("ANTHROPIC_API_KEY"))
    
    # Determine media type
    media_type_map = {
        'jpg': 'image/jpeg',
        'jpeg': 'image/jpeg',
        'png': 'image/png',
        'gif': 'image/gif',
        'webp': 'image/webp'
    }
    
    file_ext = image_file.name.split('.')[-1].lower()
    media_type = media_type_map.get(file_ext, 'image/jpeg')
    
    # Convert to base64
    image_data = image_to_base64(image_file)
    
    # Claude prompt for receipt analysis
    prompt = """Analyze this receipt image and extract the following information in JSON format:

{
    "fecha": "YYYY-MM-DD (if visible, otherwise use today's date)",
    "categoria": "one of: Gas, Nómina, Mantenimiento, Equipo, Tolls, Comida, Otro",
    "descripcion": "brief description of what was purchased",
    "monto": numeric amount only (no currency symbol),
    "tipo": "ingreso or egreso (income or expense)",
    "conductor": "driver name if visible, otherwise 'Desconocido'"
}

Rules:
- Be accurate with amounts and dates
- If handwritten, do your best to interpret
- If date is not visible, use today's date
- If no driver name, use "Desconocido"
- Always respond ONLY with valid JSON, no markdown or extra text"""

    message = client.messages.create(
        model="claude-opus-4-1",
        max_tokens=500,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_data,
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ],
            }
        ],
    )
    
    # Parse response
    response_text = message.content[0].text.strip()
    
    # Remove markdown if present
    if response_text.startswith("```"):
        response_text = response_text.split("```")[1]
        if response_text.startswith("json"):
            response_text = response_text[4:]
        response_text = response_text.strip()
    
    data = json.loads(response_text)
    return data

# ==================== DASHBOARD METRICS ====================
def get_today_summary(df):
    """Get summary for today"""
    today = pd.Timestamp.now().date()
    today_df = df[df['fecha'].dt.date == today]
    
    ingresos = today_df[today_df['tipo'] == 'ingreso']['monto'].sum()
    egresos = today_df[today_df['tipo'] == 'egreso']['monto'].sum()
    neto = ingresos - egresos
    
    return ingresos, egresos, neto, len(today_df)

def get_weekly_summary(df):
    """Get weekly breakdown"""
    last_7_days = df[df['fecha'] >= pd.Timestamp.now() - timedelta(days=7)]
    weekly = last_7_days.groupby('fecha').agg({
        'monto': 'sum',
        'tipo': lambda x: 'ingreso' if (x == 'ingreso').any() else 'egreso'
    }).reset_index()
    
    return last_7_days

def get_expense_breakdown(df, days=30):
    """Get expense breakdown by category"""
    last_n_days = df[df['fecha'] >= pd.Timestamp.now() - timedelta(days=days)]
    expenses = last_n_days[last_n_days['tipo'] == 'egreso'].groupby('categoria')['monto'].sum().sort_values(ascending=False)
    return expenses

def get_driver_performance(df, days=30):
    """Get driver performance metrics"""
    last_n_days = df[df['fecha'] >= pd.Timestamp.now() - timedelta(days=days)]
    
    driver_stats = last_n_days[last_n_days['tipo'] == 'egreso'].groupby('conductor').agg({
        'monto': ['sum', 'count', 'mean']
    }).round(2)
    
    driver_stats.columns = ['Total_Gasto', 'Transacciones', 'Promedio']
    return driver_stats.sort_values('Total_Gasto', ascending=False)

def forecast_cash_flow(df, days=30):
    """Simple linear regression forecast"""
    if len(df) < 3:
        return None
    
    df_sorted = df.sort_values('fecha')
    
    # Get daily net flow
    daily_flow = df_sorted.groupby(df_sorted['fecha'].dt.date).apply(
        lambda x: (x[x['tipo'] == 'ingreso']['monto'].sum() - 
                  x[x['tipo'] == 'egreso']['monto'].sum())
    ).reset_index()
    daily_flow.columns = ['fecha', 'flujo']
    
    if len(daily_flow) < 3:
        return None
    
    # Simple linear trend
    x = np.arange(len(daily_flow))
    y = daily_flow['flujo'].values
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    
    # Generate forecast
    last_date = df_sorted['fecha'].max()
    current_balance = df_sorted['saldo'].iloc[-1] if len(df_sorted) > 0 else 0
    
    forecast_dates = [last_date + timedelta(days=i) for i in range(1, days+1)]
    forecast_values = []
    
    balance = current_balance
    for i in range(len(daily_flow), len(daily_flow) + days):
        daily_change = p(i)
        balance += daily_change
        forecast_values.append(balance)
    
    return pd.DataFrame({
        'fecha': forecast_dates,
        'saldo_proyectado': forecast_values
    })

# ==================== VISUALIZATIONS ====================
def plot_daily_balance(df):
    """Plot daily balance trend"""
    if len(df) == 0:
        st.info("No data available")
        return
    
    df_sorted = df.sort_values('fecha')
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_sorted['fecha'],
        y=df_sorted['saldo'],
        mode='lines+markers',
        name='Saldo',
        line=dict(color='#3498db', width=2),
        fill='tozeroy'
    ))
    
    fig.update_layout(
        title="Saldo Diario - FORD 7",
        xaxis_title="Fecha",
        yaxis_title="Saldo ($)",
        hovermode='x unified',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_income_vs_expenses(df, days=30):
    """Plot income vs expenses"""
    last_n_days = df[df['fecha'] >= pd.Timestamp.now() - timedelta(days=days)]
    
    daily = last_n_days.groupby(last_n_days['fecha'].dt.date).apply(
        lambda x: pd.Series({
            'ingresos': x[x['tipo'] == 'ingreso']['monto'].sum(),
            'egresos': x[x['tipo'] == 'egreso']['monto'].sum()
        })
    ).reset_index()
    daily.columns = ['fecha', 'ingresos', 'egresos']
    
    fig = go.Figure(data=[
        go.Bar(name='Ingresos', x=daily['fecha'], y=daily['ingresos'], marker_color='#2ecc71'),
        go.Bar(name='Egresos', x=daily['fecha'], y=daily['egresos'], marker_color='#e74c3c')
    ])
    
    fig.update_layout(
        title=f"Ingresos vs Egresos - Últimos {days} días",
        barmode='group',
        xaxis_title="Fecha",
        yaxis_title="Monto ($)",
        hovermode='x unified',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_expense_breakdown(expenses):
    """Plot expense categories"""
    if len(expenses) == 0:
        st.info("No expenses in this period")
        return
    
    fig = go.Figure(data=[
        go.Pie(labels=expenses.index, values=expenses.values, hole=0.3)
    ])
    
    fig.update_layout(
        title="Desglose de Gastos por Categoría",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_forecast(df, forecast_df):
    """Plot cash flow forecast"""
    if forecast_df is None or len(forecast_df) == 0:
        st.info("No hay datos suficientes para pronóstico")
        return
    
    df_sorted = df.sort_values('fecha')
    
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=df_sorted['fecha'],
        y=df_sorted['saldo'],
        mode='lines',
        name='Histórico',
        line=dict(color='#3498db', width=2)
    ))
    
    # Forecast
    fig.add_trace(go.Scatter(
        x=forecast_df['fecha'],
        y=forecast_df['saldo_proyectado'],
        mode='lines',
        name='Pronóstico',
        line=dict(color='#e74c3c', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title="Pronóstico de Saldo (30 días)",
        xaxis_title="Fecha",
        yaxis_title="Saldo ($)",
        hovermode='x unified',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

# ==================== MAIN APP ====================
def main():
    
    # Header
    st.markdown("# 🚛 FORD 7 Expense Dashboard")
    st.markdown("**Skaai Logistics** - Automated Receipt-to-Dashboard System")
    
    # Sidebar Navigation
    with st.sidebar:
        st.title("📊 Navegación")
        page = st.radio("Selecciona una sección:", [
            "📸 Procesar Recibos",
            "📈 Dashboard",
            "📋 Transacciones",
            "👨‍💼 Desempeño Conductores",
            "🔮 Pronóstico",
            "⚙️ Configuración"
        ])
    
    # Load data
    if st.session_state.refresh_data:
        st.session_state.data_df = load_data()
        st.session_state.refresh_data = False
    
    df = st.session_state.data_df
    
    # ================== PAGE: PROCESAR RECIBOS ==================
    if page == "📸 Procesar Recibos":
        st.header("Procesar Recibos")
        st.markdown("Sube fotos de recibos y el sistema extraerá automáticamente los datos")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_files = st.file_uploader(
                "Sube uno o más recibos",
                type=['jpg', 'jpeg', 'png', 'webp'],
                accept_multiple_files=True
            )
        
        with col2:
            auto_save = st.checkbox("Guardar automáticamente", value=True)
        
        if uploaded_files:
            st.markdown("---")
            
            if st.button("🔍 Procesar Recibos", use_container_width=True):
                progress_bar = st.progress(0)
                
                new_entries = []
                
                for idx, uploaded_file in enumerate(uploaded_files):
                    try:
                        st.info(f"Procesando: {uploaded_file.name}...")
                        
                        # Extract data using Claude
                        extracted_data = extract_receipt_data(uploaded_file)
                        
                        # Convert date string to datetime
                        extracted_data['fecha'] = pd.to_datetime(extracted_data['fecha'])
                        extracted_data['camion'] = 'FORD 7'
                        extracted_data['foto_path'] = uploaded_file.name
                        
                        new_entries.append(extracted_data)
                        
                        progress_bar.progress((idx + 1) / len(uploaded_files))
                        
                    except Exception as e:
                        st.error(f"Error procesando {uploaded_file.name}: {str(e)}")
                
                if new_entries:
                    # Add to dataframe
                    new_df = pd.DataFrame(new_entries)
                    df = pd.concat([df, new_df], ignore_index=True)
                    df = df.drop_duplicates(subset=['fecha', 'monto', 'descripcion'], keep='last')
                    df = calculate_balance(df)
                    
                    # Save
                    save_data(df)
                    st.session_state.data_df = df
                    
                    st.success(f"✅ Se agregaron {len(new_entries)} transacciones")
                    
                    # Show extracted data
                    st.markdown("### Datos Extraídos:")
                    for idx, entry in enumerate(new_entries, 1):
                        with st.expander(f"Transacción {idx}: ${entry['monto']} - {entry['categoria']}"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"**Fecha:** {entry['fecha'].strftime('%Y-%m-%d')}")
                                st.write(f"**Monto:** ${entry['monto']}")
                                st.write(f"**Tipo:** {entry['tipo'].upper()}")
                            with col2:
                                st.write(f"**Categoría:** {entry['categoria']}")
                                st.write(f"**Conductor:** {entry['conductor']}")
                                st.write(f"**Descripción:** {entry['descripcion']}")
    
    # ================== PAGE: DASHBOARD ==================
    elif page == "📈 Dashboard":
        st.header("Dashboard Diario")
        
        if len(df) == 0:
            st.warning("No hay datos. Por favor, agrega transacciones primero.")
        else:
            # Top metrics
            ingresos_hoy, egresos_hoy, neto_hoy, trans_hoy = get_today_summary(df)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("💰 Ingresos Hoy", f"${ingresos_hoy:,.0f}", "↑" if ingresos_hoy > 0 else "")
            
            with col2:
                st.metric("📊 Egresos Hoy", f"${egresos_hoy:,.0f}", "↓" if egresos_hoy > 0 else "")
            
            with col3:
                st.metric("💵 Neto Hoy", f"${neto_hoy:,.0f}", 
                         delta_color="inverse" if neto_hoy < 0 else "normal")
            
            with col4:
                current_balance = df['saldo'].iloc[-1] if len(df) > 0 else 0
                st.metric("🏦 Saldo Actual", f"${current_balance:,.0f}")
            
            st.markdown("---")
            
            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                plot_daily_balance(df)
            
            with col2:
                plot_income_vs_expenses(df, days=30)
            
            st.markdown("---")
            
            # Expense breakdown
            expenses = get_expense_breakdown(df, days=30)
            if len(expenses) > 0:
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    plot_expense_breakdown(expenses)
                
                with col2:
                    st.markdown("### Gastos por Categoría (30 días)")
                    for cat, amount in expenses.items():
                        st.write(f"**{cat}**: ${amount:,.0f}")
    
    # ================== PAGE: TRANSACCIONES ==================
    elif page == "📋 Transacciones":
        st.header("Registro de Transacciones")
        
        if len(df) == 0:
            st.warning("No hay transacciones registradas")
        else:
            # Filters
            col1, col2, col3 = st.columns(3)
            
            with col1:
                tipo_filter = st.selectbox("Tipo", ["Todos", "ingreso", "egreso"])
            
            with col2:
                cat_filter = st.multiselect("Categoría", df['categoria'].unique(), default=df['categoria'].unique())
            
            with col3:
                days_back = st.slider("Últimos X días", 1, 365, 30)
            
            # Filter data
            filtered_df = df[df['fecha'] >= pd.Timestamp.now() - timedelta(days=days_back)].copy()
            
            if tipo_filter != "Todos":
                filtered_df = filtered_df[filtered_df['tipo'] == tipo_filter]
            
            if cat_filter:
                filtered_df = filtered_df[filtered_df['categoria'].isin(cat_filter)]
            
            # Display table
            display_df = filtered_df[['fecha', 'categoria', 'descripcion', 'monto', 'tipo', 'conductor', 'saldo']].copy()
            display_df['fecha'] = display_df['fecha'].dt.strftime('%Y-%m-%d')
            
            st.dataframe(display_df.sort_values('fecha', ascending=False), use_container_width=True)
            
            # Summary
            st.markdown("---")
            total_ingresos = filtered_df[filtered_df['tipo'] == 'ingreso']['monto'].sum()
            total_egresos = filtered_df[filtered_df['tipo'] == 'egreso']['monto'].sum()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Ingresos", f"${total_ingresos:,.0f}")
            with col2:
                st.metric("Egresos", f"${total_egresos:,.0f}")
            with col3:
                st.metric("Neto", f"${total_ingresos - total_egresos:,.0f}")
            
            # Download
            csv = display_df.to_csv(index=False)
            st.download_button(
                label="📥 Descargar CSV",
                data=csv,
                file_name=f"ford7_transacciones_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    # ================== PAGE: DESEMPEÑO CONDUCTORES ==================
    elif page == "👨‍💼 Desempeño Conductores":
        st.header("Desempeño de Conductores")
        
        if len(df) == 0:
            st.warning("No hay datos de conductores")
        else:
            days = st.slider("Analizar últimos X días", 1, 365, 30)
            
            driver_perf = get_driver_performance(df, days=days)
            
            if len(driver_perf) == 0:
                st.info("Sin datos de gastos para este período")
            else:
                # Bar chart
                fig = go.Figure(data=[
                    go.Bar(x=driver_perf.index, y=driver_perf['Total_Gasto'], marker_color='#e74c3c')
                ])
                
                fig.update_layout(
                    title=f"Gasto por Conductor - Últimos {days} días",
                    xaxis_title="Conductor",
                    yaxis_title="Gasto Total ($)",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Table
                st.markdown("### Métricas Detalladas")
                st.dataframe(driver_perf, use_container_width=True)
    
    # ================== PAGE: PRONÓSTICO ==================
    elif page == "🔮 Pronóstico":
        st.header("Pronóstico de Flujo de Efectivo")
        
        if len(df) < 3:
            st.warning("Se requieren al menos 3 transacciones para generar pronóstico")
        else:
            forecast_df = forecast_cash_flow(df, days=30)
            
            if forecast_df is not None:
                plot_forecast(df, forecast_df)
                
                st.markdown("### Resumen del Pronóstico")
                current_balance = df['saldo'].iloc[-1]
                forecast_final = forecast_df['saldo_proyectado'].iloc[-1]
                change = forecast_final - current_balance
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Saldo Actual", f"${current_balance:,.0f}")
                with col2:
                    st.metric("Saldo Proyectado (30 días)", f"${forecast_final:,.0f}")
                with col3:
                    st.metric("Cambio Esperado", f"${change:,.0f}", 
                             delta_color="inverse" if change < 0 else "normal")
                
                # Daily breakdown
                st.markdown("---")
                st.markdown("### Proyección Diaria")
                st.dataframe(forecast_df.rename(columns={'fecha': 'Fecha', 'saldo_proyectado': 'Saldo Proyectado'}), use_container_width=True)
            else:
                st.error("No se pudo generar el pronóstico")
    
    # ================== PAGE: CONFIGURACIÓN ==================
    elif page == "⚙️ Configuración":
        st.header("Configuración")
        
        st.markdown("### API Key")
        st.info("La API Key debe estar configurada en los secrets de Streamlit Cloud")
        
        st.markdown("### Datos")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Recargar Datos"):
                st.session_state.refresh_data = True
                st.rerun()
        
        with col2:
            if st.button("🗑️ Limpiar Datos", type="secondary"):
                if st.checkbox("Confirmar eliminación de todos los datos"):
                    os.remove("ford7_expenses.csv")
                    st.session_state.data_df = pd.DataFrame()
                    st.success("Datos eliminados")
                    st.rerun()
        
        st.markdown("---")
        st.markdown("### Información del Sistema")
        st.write(f"**Total de Transacciones:** {len(df)}")
        st.write(f"**Fecha de Inicio:** {df['fecha'].min().strftime('%Y-%m-%d') if len(df) > 0 else 'N/A'}")
        st.write(f"**Saldo Actual:** ${df['saldo'].iloc[-1]:,.0f}" if len(df) > 0 else "N/A")

if __name__ == "__main__":
    # Check for API key
    if "ANTHROPIC_API_KEY" not in st.secrets:
        st.error("⚠️ ANTHROPIC_API_KEY no configurada. Por favor, añade tu API key en los secrets.")
        st.stop()
    
    main()
