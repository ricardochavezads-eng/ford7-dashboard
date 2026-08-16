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

def find_new_entries_in_db(extracted_entries, df, last_entry_tracker):
    """Find new entries by comparing against full database"""
    new_entries = []
    duplicate_entries = []
    
    if len(df) == 0:
        # If database is empty, all are new
        return extracted_entries, []
    
    # Get all existing dates and amounts in database
    existing_combinations = set()
    for idx, row in df.iterrows():
        key = (row['fecha'], round(float(row['monto']), 2), row['descripcion'][:30])
        existing_combinations.add(key)
    
    # Check each extracted entry
    for entry in extracted_entries:
        entry_date = pd.to_datetime(entry['fecha']).strftime('%Y-%m-%d')
        key = (entry_date, round(float(entry['monto']), 2), entry['descripcion'][:30])
        
        if key not in existing_combinations:
            new_entries.append(entry)
        else:
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

def image_to_base64(image_file):
    """Convert image to base64 for API"""
    return base64.standard_b64encode(image_file.read()).decode("utf-8")
    """Convert image to base64 for API"""
    return base64.standard_b64encode(image_file.read()).decode("utf-8")

# ==================== OCR WITH CLAUDE ====================
def extract_table_data(image_file):
    """Use Claude vision to extract table/ledger data from screenshots"""
    
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
    
    # Claude prompt for table extraction
    prompt = """Analiza esta tabla/ledger de FORD 7 y extrae TODAS las filas como JSON.

Para cada fila, extrae:
- fecha: YYYY-MM-DD (si no es clara, usa la fecha más cercana lógica)
- concepto: El nombre de la categoría/concepto (Gas, Nómina, Pago, etc)
- descripcion: La descripción completa de la transacción
- ingresos: número si hay ingreso, null si no
- egresos: número si hay egreso, null si no

Retorna SOLO un array JSON válido, sin markdown ni preamble:
[
  {"fecha": "2026-07-20", "concepto": "Pago Oficina Julio", "descripcion": "Renta, servicios", "ingresos": null, "egresos": 2572.25},
  {"fecha": "2026-07-21", "concepto": "Gas", "descripcion": "Bryan Salazar T. 08:02", "ingresos": null, "egresos": 500.00}
]

IMPORTANTE:
- Extrae TODAS las filas visibles
- Si hay dos columnas de ingresos/egresos, suma y pon el total
- Fechas deben ser YYYY-MM-DD
- Números sin formato (sin $ ni comas)
- Responde SOLO con el JSON array"""

    message = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=2000,
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
    
    extracted_rows = json.loads(response_text)
    
    # Convert to dashboard format
    entries = []
    for row in extracted_rows:
        # Determine tipo
        if row.get('ingresos') and row['ingresos'] > 0:
            tipo = 'ingreso'
            monto = row['ingresos']
        elif row.get('egresos') and row['egresos'] > 0:
            tipo = 'egreso'
            monto = row['egresos']
        else:
            continue
        
        entries.append({
            'fecha': row['fecha'],
            'categoria': row['concepto'],
            'descripcion': row['descripcion'],
            'monto': monto,
            'tipo': tipo,
            'conductor': 'Desconocido',
            'saldo': 0,
            'foto_path': image_file.name
        })
    
    return entries

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
        st.markdown("## 📸 Procesar Ledger", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("Sube una captura de pantalla de tu ledger/tabla y edita los datos antes de guardar")
        
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
        
        uploaded_file = st.file_uploader(
            "Sube una captura de pantalla del ledger",
            type=['jpg', 'jpeg', 'png', 'webp']
        )
        
        if uploaded_file:
            st.markdown("---")
            
            if st.button("🔍 Procesar Tabla", use_container_width=True, type="primary"):
                progress_bar = st.progress(0)
                status_container = st.container()
                
                try:
                    with status_container:
                        st.info("Leyendo tabla... Espera un momento")
                    
                    # Extract data using Claude
                    extracted_entries = extract_table_data(uploaded_file)
                    
                    # SAVE TO SESSION STATE - KEY FIX!
                    st.session_state.extracted_entries = extracted_entries
                    st.session_state.duplicate_entries = []
                    st.session_state.last_entry_tracker = load_last_entry_tracker()
                    
                    progress_bar.progress(50)
                    
                    with status_container:
                        st.info("Analizando filas nuevas...")
                    
                    # Find new vs duplicate entries - CHECK AGAINST FULL DATABASE
                    new_entries, duplicate_entries = find_new_entries_in_db(extracted_entries, df, st.session_state.last_entry_tracker)
                    
                    # SAVE TO SESSION STATE
                    st.session_state.new_entries = new_entries
                    st.session_state.duplicate_entries = duplicate_entries
                    st.session_state.uploaded_filename = uploaded_file.name
                    st.session_state.row_selections = {i: True for i in range(len(new_entries))}
                    
                    progress_bar.progress(100)
                    
                except Exception as e:
                    st.error(f"Error procesando tabla: {str(e)}")
                    st.info("Asegúrate de que la imagen sea clara y contenga una tabla legible.")
        
        # SHOW RESULTS FROM SESSION STATE (persists across reruns!)
        if "new_entries" in st.session_state and len(st.session_state.new_entries) > 0:
            new_entries = st.session_state.new_entries
            duplicate_entries = st.session_state.duplicate_entries
            
            # Show results
            st.markdown("---")
            st.markdown("### 📊 Análisis de Tabla")
            
       
