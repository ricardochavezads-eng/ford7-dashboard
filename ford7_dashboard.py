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
import time

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

def load_last_entry_tracker():
    """Load the last entry tracker from file"""
    if os.path.exists("last_entry_tracker.json"):
        with open("last_entry_tracker.json", "r") as f:
            return json.load(f)
    return None

def save_last_entry_tracker(entry_data):
    """Save the last entry tracker"""
    with open("last_entry_tracker.json", "w") as f:
        json.dump(entry_data, f)

def find_new_entries(extracted_entries, last_entry_tracker):
    """Find only new entries that aren't already in the system"""
    if last_entry_tracker is None:
        return extracted_entries, []
    
    new_entries = []
    duplicate_entries = []
    last_date = pd.to_datetime(last_entry_tracker.get('fecha', '2020-01-01'))
    
    for entry in extracted_entries:
        entry_date = pd.to_datetime(entry['fecha'])
        
        # Check if entry is after last known entry
        if entry_date > last_date:
            new_entries.append(entry)
        else:
            # Check for exact duplicates (same date + amount)
            if (entry_date == last_date and 
                abs(float(entry['monto']) - float(last_entry_tracker.get('monto', 0))) < 0.01):
                duplicate_entries.append(entry)
    
    return new_entries, duplicate_entries

def get_entry_summary(entry):
    """Create a readable summary of an entry"""
    return {
        'Fecha': entry['fecha'],
        'Concepto': entry['categoria'],
        'Descripción': entry['descripcion'][:50] if entry['descripcion'] else '',
        'Monto': f"${entry['monto']:,.2f}",
        'Tipo': entry['tipo'].upper()
    }
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
    
    # Custom CSS for animations
    st.markdown("""
    <style>
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes slideInLeft {
        from {
            opacity: 0;
            transform: translateX(-30px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(30px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes bounceIn {
        0% {
            opacity: 0;
            transform: scale(0.85);
        }
        50% {
            opacity: 1;
        }
        100% {
            transform: scale(1);
        }
    }
    
    @keyframes pulse {
        0%, 100% {
            opacity: 1;
        }
        50% {
            opacity: 0.7;
        }
    }
    
    @keyframes spinRotate {
        from {
            transform: rotate(0deg);
        }
        to {
            transform: rotate(360deg);
        }
    }
    
    .header-animated {
        animation: fadeInDown 0.6s ease-out;
    }
    
    .metric-card-animated {
        animation: bounceIn 0.5s cubic-bezier(0.34, 1.56, 0.64, 1);
    }
    
    .chart-animated {
        animation: slideInLeft 0.8s ease-out;
    }
    
    .category-animated {
        animation: slideInRight 0.8s ease-out;
    }
    
    .icon-rotating {
        display: inline-block;
        animation: spinRotate 2s linear infinite;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header with animation
    st.markdown('<div class="header-animated">', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 4])
    with col1:
        st.markdown("## 🚛")
    with col2:
        st.markdown("# FORD 7 Dashboard")
        st.markdown("**Skaai Logistics** - Real-time Operations | Automated Receipt Processing")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Sidebar Navigation
    with st.sidebar:
        st.title("📊 Navegación")
        page = st.radio("Selecciona una sección:", [
            "📸 Procesar Recibos",
            "📈 Dashboard",
            "📋 Transacciones",
            "👨‍💼 Desempeño Conductores",
            "👩‍💼 Personal Administrativo",
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
        st.markdown('<div class="header-animated">', unsafe_allow_html=True)
        st.markdown("## 📸 Procesar Recibos", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("Sube fotos de recibos y el sistema extraerá automáticamente los datos")
        
        # Load last entry tracker
        last_entry_tracker = load_last_entry_tracker()
        
        # Show last entry info
        if last_entry_tracker:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.info(f"""
                ✓ **Última entrada registrada:** {last_entry_tracker['fecha']}  
                **{last_entry_tracker['concepto']}** - ${last_entry_tracker['monto']:,.2f}  
                *{last_entry_tracker['descripcion'][:60]}*
                """)
            with col2:
                if st.button("🔄 Resetear", use_container_width=True):
                    os.remove("last_entry_tracker.json")
                    st.rerun()
        else:
            st.warning("⚠️ No hay última entrada registrada. Se agregarán todas las nuevas entradas.")
        
        st.markdown("---")
        
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
            
            if st.button("🔍 Procesar Recibos", use_container_width=True, type="primary"):
                progress_bar = st.progress(0)
                status_container = st.container()
                
                extracted_entries = []
                extraction_errors = []
                
                for idx, uploaded_file in enumerate(uploaded_files):
                    try:
                        with status_container:
                            st.info(f"Procesando: {uploaded_file.name}... ⏳")
                        
                        # Extract data using Claude
                        extracted_data = extract_receipt_data(uploaded_file)
                        
                        # Convert date string to datetime
                        extracted_data['fecha'] = pd.to_datetime(extracted_data['fecha']).strftime('%Y-%m-%d')
                        extracted_data['camion'] = 'FORD 7'
                        extracted_data['foto_path'] = uploaded_file.name
                        
                        extracted_entries.append(extracted_data)
                        progress_bar.progress((idx + 1) / len(uploaded_files))
                        
                    except Exception as e:
                        extraction_errors.append(f"{uploaded_file.name}: {str(e)}")
                
                # Find new vs duplicate entries
                new_entries, duplicate_entries = find_new_entries(extracted_entries, last_entry_tracker)
                
                # Show results
                st.markdown("---")
                st.markdown("### 📊 Análisis de Recibos")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("✅ Nuevas Entradas", len(new_entries))
                with col2:
                    st.metric("⚠️ Duplicadas", len(duplicate_entries))
                with col3:
                    st.metric("❌ Errores", len(extraction_errors))
                
                st.markdown("---")
                
                # Show duplicates warning
                if duplicate_entries:
                    with st.expander(f"⚠️ Duplicadas ({len(duplicate_entries)}) - Ya están en el sistema"):
                        dup_df = pd.DataFrame([get_entry_summary(e) for e in duplicate_entries])
                        st.dataframe(dup_df, use_container_width=True)
                
                # Show errors
                if extraction_errors:
                    with st.expander(f"❌ Errores ({len(extraction_errors)})"):
                        for error in extraction_errors:
                            st.error(error)
                
                # MAIN: Show new entries for approval
                if new_entries:
                    st.success(f"✅ Se encontraron {len(new_entries)} nuevas entradas")
                    
                    with st.expander("📋 Vista Previa de Nuevas Entradas", expanded=True):
                        preview_df = pd.DataFrame([get_entry_summary(e) for e in new_entries])
                        st.dataframe(preview_df, use_container_width=True)
                        
                        # Show detailed preview
                        st.markdown("### 🔍 Detalles de Cada Entrada")
                        for idx, entry in enumerate(new_entries, 1):
                            with st.expander(f"Entrada {idx}: {entry['categoria']} - ${entry['monto']:,.2f}"):
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.write(f"**Fecha:** {entry['fecha']}")
                                    st.write(f"**Monto:** ${entry['monto']:,.2f}")
                                    st.write(f"**Tipo:** {entry['tipo'].upper()}")
                                with col2:
                                    st.write(f"**Categoría:** {entry['categoria']}")
                                    st.write(f"**Conductor:** {entry['conductor']}")
                                    st.write(f"**Descripción:** {entry['descripcion']}")
                    
                    # Confirmation button
                    st.markdown("---")
                    
                    confirm_col1, confirm_col2, confirm_col3 = st.columns([1, 1, 2])
                    
                    with confirm_col1:
                        if st.button("✅ Guardar Todas", use_container_width=True, type="primary"):
                            # Add to dataframe
                            new_df = pd.DataFrame(new_entries)
                            df = pd.concat([df, new_df], ignore_index=True)
                            df = df.drop_duplicates(subset=['fecha', 'monto', 'descripcion'], keep='last')
                            df = calculate_balance(df)
                            
                            # Save
                            save_data(df)
                            st.session_state.data_df = df
                            
                            # Update tracker with last entry
                            if len(new_entries) > 0:
                                last_new = new_entries[-1]
                                save_last_entry_tracker({
                                    'fecha': last_new['fecha'],
                                    'concepto': last_new['categoria'],
                                    'descripcion': last_new['descripcion'],
                                    'monto': last_new['monto']
                                })
                            
                            st.success(f"✅ Se agregaron {len(new_entries)} transacciones correctamente")
                            st.balloons()
                            
                            # Rerun to refresh
                            time.sleep(1)
                            st.rerun()
                    
                    with confirm_col2:
                        if st.button("❌ Cancelar", use_container_width=True):
                            st.info("Cancelado. No se agregaron cambios.")
                    
                    with confirm_col3:
                        st.info(f"📌 Revisión pendiente: {len(new_entries)} nuevas entradas")
                
                else:
                    if len(new_entries) == 0 and len(duplicate_entries) > 0:
                        st.warning(f"⚠️ Todas las entradas ({len(duplicate_entries)}) ya están en el sistema. Nada nuevo que agregar.")
                    else:
                        st.info("No se encontraron nuevas entradas para procesar.")
            
            else:
                st.info("Haz clic en 'Procesar Recibos' para extraer datos de los recibos cargados")
    
    # ================== PAGE: DASHBOARD ==================
    elif page == "📈 Dashboard":
        st.markdown('<div class="header-animated">', unsafe_allow_html=True)
        st.markdown("## 📈 Dashboard Diario", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        if len(df) == 0:
            st.warning("No hay datos. Por favor, agrega transacciones primero.")
        else:
            # TIME SLIDER WITH ANIMATION
            st.markdown('<div class="chart-animated">', unsafe_allow_html=True)
            col1, col2, col3 = st.columns([1, 3, 1])
            
            with col1:
                st.markdown("**📅 Período:**")
            
            with col2:
                days_range = st.slider(
                    "Selecciona rango de días",
                    min_value=7,
                    max_value=365,
                    value=30,
                    step=7,
                    label_visibility="collapsed"
                )
            
            with col3:
                if st.button("🔄 Actualizar", use_container_width=True):
                    st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Filter data by range
            filtered_df = df[df['fecha'] >= pd.Timestamp.now() - timedelta(days=days_range)].copy()
            
            # ANIMATED METRIC CARDS
            st.markdown('<div class="metric-card-animated">', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4, gap="small")
            
            with col1:
                ingresos = filtered_df[filtered_df['tipo'] == 'ingreso']['monto'].sum()
                st.metric(
                    "💰 Ingresos",
                    f"${ingresos:,.0f}",
                    f"({len(filtered_df[filtered_df['tipo'] == 'ingreso'])} transacciones)",
                    border=True
                )
            
            with col2:
                egresos = filtered_df[filtered_df['tipo'] == 'egreso']['monto'].sum()
                st.metric(
                    "📊 Egresos",
                    f"${egresos:,.0f}",
                    f"({len(filtered_df[filtered_df['tipo'] == 'egreso'])} transacciones)",
                    border=True
                )
            
            with col3:
                neto = ingresos - egresos
                st.metric(
                    "💵 Neto",
                    f"${neto:,.0f}",
                    delta=f"{'↑' if neto > 0 else '↓'} {abs(neto/ingresos*100) if ingresos > 0 else 0:.1f}%" if ingresos > 0 else "—",
                    delta_color="inverse" if neto < 0 else "normal",
                    border=True
                )
            
            with col4:
                current_balance = df['saldo'].iloc[-1] if len(df) > 0 else 0
                st.metric(
                    "🏦 Balance",
                    f"${current_balance:,.0f}",
                    f"{'✓ Positivo' if current_balance > 0 else '⚠ Negativo'}",
                    border=True
                )
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("---")
            
            # ANIMATED CHARTS
            st.markdown('<div class="chart-animated">', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📈 Tendencia de Balance")
                plot_daily_balance(filtered_df if len(filtered_df) > 0 else df)
            
            with col2:
                st.markdown("### 💹 Ingresos vs Egresos")
                plot_income_vs_expenses(filtered_df if len(filtered_df) > 0 else df, days=days_range)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("---")
            
            # ANIMATED BREAKDOWN
            st.markdown('<div class="category-animated">', unsafe_allow_html=True)
            
            expenses = get_expense_breakdown(filtered_df if len(filtered_df) > 0 else df, days=days_range)
            if len(expenses) > 0:
                col1, col2 = st.columns([1.2, 0.8])
                
                with col1:
                    st.markdown("### 📊 Desglose de Gastos")
                    plot_expense_breakdown(expenses)
                
                with col2:
                    st.markdown("### 📋 Por Categoría")
                    for cat, amount in expenses.items():
                        st.write(f"**{cat}**")
                        st.write(f"${amount:,.0f}")
                        st.divider()
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # ================== PAGE: TRANSACCIONES ==================
    elif page == "📋 Transacciones":
        st.markdown('<div class="header-animated">', unsafe_allow_html=True)
        st.markdown("## 📋 Registro de Transacciones", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
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
        st.markdown('<div class="header-animated">', unsafe_allow_html=True)
        st.markdown("## 👨‍💼 Desempeño de Conductores", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        if len(df) == 0:
            st.warning("No hay datos de conductores")
        else:
            days = st.slider("Analizar últimos X días", 1, 365, 30)
            
            # Get data for period - EXCLUDE OFFICE STAFF (Marbella)
            last_n_days = df[df['fecha'] >= pd.Timestamp.now() - timedelta(days=days)]
            driver_expenses = last_n_days[(last_n_days['tipo'] == 'egreso') & (last_n_days['conductor'] != 'Marbella Martinez')]
            
            # Get driver performance (excluding Marbella)
            driver_perf = driver_expenses.groupby('conductor').agg({
                'monto': ['sum', 'count', 'mean']
            }).round(2)
            
            driver_perf.columns = ['Total_Gasto', 'Transacciones', 'Promedio']
            driver_perf = driver_perf.sort_values('Total_Gasto', ascending=False)
            
            if len(driver_perf) == 0:
                st.info("Sin datos de gastos para este período")
            else:
                
                # Bar chart - Total Gasto
                fig = go.Figure(data=[
                    go.Bar(x=driver_perf.index, y=driver_perf['Total_Gasto'], marker_color='#e74c3c')
                ])
                
                fig.update_layout(
                    title=f"Gasto Total por Conductor - Últimos {days} días",
                    xaxis_title="Conductor",
                    yaxis_title="Gasto Total ($)",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Summary metrics
                st.markdown("### Resumen por Conductor")
                st.dataframe(driver_perf, use_container_width=True)
                
                st.markdown("---")
                
                # Breakdown by category for each driver
                st.markdown("### Desglose por Categoría (por Conductor)")
                
                drivers = sorted(driver_expenses['conductor'].unique())
                
                col1, col2 = st.columns(2)
                
                for idx, driver in enumerate(drivers):
                    if idx % 2 == 0:
                        col = col1
                    else:
                        col = col2
                    
                    with col:
                        driver_data = driver_expenses[driver_expenses['conductor'] == driver]
                        cat_breakdown = driver_data.groupby('categoria')['monto'].sum().sort_values(ascending=False)
                        
                        if len(cat_breakdown) > 0:
                            with st.expander(f"👤 {driver} - ${cat_breakdown.sum():,.0f}"):
                                # Category pie chart
                                fig_pie = go.Figure(data=[
                                    go.Pie(labels=cat_breakdown.index, values=cat_breakdown.values, hole=0.3)
                                ])
                                fig_pie.update_layout(height=300)
                                st.plotly_chart(fig_pie, use_container_width=True)
                                
                                # Category breakdown table
                                st.markdown(f"**Total: ${cat_breakdown.sum():,.0f}**")
                                for cat, amount in cat_breakdown.items():
                                    st.write(f"• {cat}: ${amount:,.0f}")
                
                st.markdown("---")
                
                # Detailed transactions per driver
                st.markdown("### Transacciones Detalladas por Conductor")
                
                for driver in sorted(drivers):
                    with st.expander(f"📋 {driver} - Todas las transacciones"):
                        driver_trans = driver_expenses[driver_expenses['conductor'] == driver].copy()
                        driver_trans_display = driver_trans[['fecha', 'categoria', 'descripcion', 'monto']].sort_values('fecha', ascending=False)
                        driver_trans_display['fecha'] = driver_trans_display['fecha'].astype(str)
                        
                        st.dataframe(driver_trans_display, use_container_width=True)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Gasto", f"${driver_trans['monto'].sum():,.0f}")
                        with col2:
                            st.metric("# Transacciones", len(driver_trans))
                        with col3:
                            st.metric("Promedio", f"${driver_trans['monto'].mean():,.0f}")
    
    # ================== PAGE: PERSONAL ADMINISTRATIVO ==================
    elif page == "👩‍💼 Personal Administrativo":
        st.markdown('<div class="header-animated">', unsafe_allow_html=True)
        st.markdown("## 👩‍💼 Personal Administrativo", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("*Marbella Martinez - Contadora y Administradora*")
        
        if len(df) == 0:
            st.warning("No hay datos")
        else:
            days = st.slider("Analizar últimos X días", 1, 365, 30, key="admin_days")
            
            # Get Marbella's data only
            last_n_days = df[df['fecha'] >= pd.Timestamp.now() - timedelta(days=days)]
            marbella_data = last_n_days[last_n_days['conductor'] == 'Marbella Martinez']
            
            if len(marbella_data) == 0:
                st.info("Sin datos para Marbella Martinez en este período")
            else:
                # Summary metrics
                marbella_expenses = marbella_data[marbella_data['tipo'] == 'egreso']
                marbella_income = marbella_data[marbella_data['tipo'] == 'ingreso']
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Salario/Egresos", f"${marbella_expenses['monto'].sum():,.0f}")
                
                with col2:
                    st.metric("# Transacciones", len(marbella_data))
                
                with col3:
                    st.metric("Promedio Transacción", f"${marbella_data[marbella_data['tipo'] == 'egreso']['monto'].mean():,.0f}")
                
                with col4:
                    st.metric("Período", f"Últimos {days} días")
                
                st.markdown("---")
                
                # Breakdown by category
                st.markdown("### Desglose por Categoría")
                
                cat_breakdown = marbella_expenses.groupby('categoria')['monto'].sum().sort_values(ascending=False)
                
                if len(cat_breakdown) > 0:
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        # Pie chart
                        fig = go.Figure(data=[
                            go.Pie(labels=cat_breakdown.index, values=cat_breakdown.values, hole=0.3)
                        ])
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.markdown("**Gastos por Categoría:**")
                        for cat, amount in cat_breakdown.items():
                            st.write(f"• {cat}: ${amount:,.0f}")
                
                st.markdown("---")
                
                # Detailed transactions
                st.markdown("### Transacciones Detalladas")
                
                marbella_display = marbella_data[['fecha', 'categoria', 'descripcion', 'monto', 'tipo']].sort_values('fecha', ascending=False)
                marbella_display['fecha'] = marbella_display['fecha'].astype(str)
                
                st.dataframe(marbella_display, use_container_width=True)
                
                # Export option
                st.markdown("---")
                csv_export = marbella_display.to_csv(index=False)
                st.download_button(
                    label="📥 Descargar Transacciones (CSV)",
                    data=csv_export,
                    file_name=f"marbella_martinez_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
    
    # ================== PAGE: PRONÓSTICO ==================
    elif page == "🔮 Pronóstico":
        st.markdown('<div class="header-animated">', unsafe_allow_html=True)
        st.markdown("## 🔮 Pronóstico de Flujo de Efectivo", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
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
        st.markdown('<div class="header-animated">', unsafe_allow_html=True)
        st.markdown("## ⚙️ Configuración", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("### 🔍 Seguimiento de Entradas")
        
        last_entry = load_last_entry_tracker()
        
        if last_entry:
            st.info(f"""
            **Última entrada registrada:**
            - Fecha: {last_entry['fecha']}
            - Concepto: {last_entry['concepto']}
            - Monto: ${last_entry['monto']:,.2f}
            - Descripción: {last_entry['descripcion']}
            """)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Resetear Tracker", use_container_width=True, type="secondary"):
                    os.remove("last_entry_tracker.json")
                    st.success("✓ Tracker reseteado. En el próximo procesamiento, se agregarán todas las entradas nuevas.")
                    st.rerun()
            
            with col2:
                st.metric("Entradas Después del Último", len(df[df['fecha'] > pd.to_datetime(last_entry['fecha'])]))
        else:
            st.warning("⚠️ No hay última entrada registrada. Se agregarán todas las nuevas entradas en el próximo procesamiento.")
        
        st.markdown("---")
        
        st.markdown("### 📊 Datos del Sistema")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total de Transacciones", len(df))
        
        with col2:
            st.metric("Fecha de Inicio", df['fecha'].min().strftime('%Y-%m-%d') if len(df) > 0 else 'N/A')
        
        with col3:
            st.metric("Saldo Actual", f"${df['saldo'].iloc[-1]:,.0f}" if len(df) > 0 else 'N/A')
        
        st.markdown("---")
        
        st.markdown("### 🗑️ Gestión de Datos")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Recargar Datos", use_container_width=True):
                st.session_state.refresh_data = True
                st.rerun()
        
        with col2:
            if st.button("⚠️ Limpiar Datos", use_container_width=True, type="secondary"):
                if st.checkbox("Confirmar eliminación de todos los datos"):
                    if os.path.exists("ford7_expenses.csv"):
                        os.remove("ford7_expenses.csv")
                    if os.path.exists("last_entry_tracker.json"):
                        os.remove("last_entry_tracker.json")
                    st.session_state.data_df = pd.DataFrame()
                    st.success("✓ Datos eliminados")
                    st.rerun()

if __name__ == "__main__":
    # Check for API key
    if "ANTHROPIC_API_KEY" not in st.secrets:
        st.error("⚠️ ANTHROPIC_API_KEY no configurada. Por favor, añade tu API key en los secrets.")
        st.stop()
    
    main()
