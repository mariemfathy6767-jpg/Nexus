import streamlit as st
import pandas as pd
import numpy as np
import random
import time
import heapq
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

st.set_page_config(page_title='NEXUS – Autonomous City Intelligence', page_icon='🏙️', layout='wide')

st.markdown('''
<style>
.main { background-color: #0F0C29; }
.block-container { padding-top: 1.2rem; }
.stButton>button { border-radius: 8px; font-weight: 700; width: 100%; }
</style>
''', unsafe_allow_html=True)

DATA_FILE = 'RTA Dataset.csv'

@st.cache_data

def load_data():
    df = pd.read_csv(DATA_FILE)
    df.columns = df.columns.str.strip()
    return df

@st.cache_resource

def train_model(data):
    features = [
        'Weather_conditions',
        'Light_conditions',
        'Road_surface_conditions',
        'Road_surface_type',
        'Types_of_Junction',
        'Area_accident_occured',
        'Type_of_collision',
        'Number_of_vehicles_involved',
        'Number_of_casualties',
        'Cause_of_accident'
    ]
    target = 'Accident_severity'
    work = data[features + [target]].copy().dropna()
    X = work[features].copy()
    y = work[target].copy()
    categorical = [c for c in features if X[c].dtype == 'object']
    numeric = [c for c in features if c not in categorical]
    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical),
        ('num', 'passthrough', numeric)
    ])
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced_subsample', n_jobs=-1))
    ])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, pred)
    report = classification_report(y_test, pred, output_dict=True, zero_division=0)
    return model, accuracy, report, features


def make_simulation_data():
    hospitals = pd.DataFrame([
        ['H1', 'Central Hospital', 30.050, 31.240, 12, 4, 25, 'Available'],
        ['H2', 'North District Hospital', 30.065, 31.225, 8, 2, 18, 'Available'],
        ['H3', 'East Emergency Hospital', 30.035, 31.265, 3, 0, 8, 'Limited'],
        ['H4', 'West General Hospital', 30.030, 31.215, 20, 6, 30, 'Available']
    ], columns=['id','name','lat','lon','beds','icu','emergency_capacity','status'])
    ambulances = pd.DataFrame([
        ['AMB-01', 'A', 30.040, 31.225, 'Available'],
        ['AMB-02', 'D', 30.055, 31.235, 'Available'],
        ['AMB-03', 'G', 30.025, 31.225, 'Busy'],
        ['AMB-04', 'B', 30.045, 31.245, 'Available'],
        ['AMB-05', 'H', 30.030, 31.245, 'Available']
    ], columns=['id','node','lat','lon','status'])
    fire = pd.DataFrame([
        ['FIRE-01', 'C', 30.060, 31.255, 'Available'],
        ['FIRE-02', 'F', 30.040, 31.260, 'Available'],
        ['FIRE-03', 'D', 30.055, 31.230, 'Busy']
    ], columns=['id','node','lat','lon','status'])
    roads = [
        ('A','B',3,4),('A','D',4,5),('B','C',3,6),('B','E',4,4),('D','E',2,3),
        ('D','G',4,4),('E','F',3,5),('E','H',3,3),('C','F',3,4),('F','I',3,4),
        ('G','H',2,3),('H','I',3,4),('C','E',5,7),('G','D',4,5),('B','D',5,7)
    ]
    return hospitals, ambulances, fire, roads


def build_graph(roads, traffic_factor, blocked_edges):
    graph = {}
    for a, b, distance, base_time in roads:
        edge = tuple(sorted((a, b)))
        if edge in blocked_edges:
            continue
        cost = base_time * traffic_factor.get(edge, 1.0)
        graph.setdefault(a, []).append((b, cost, distance))
        graph.setdefault(b, []).append((a, cost, distance))
    return graph


def dijkstra(roads, start, end, traffic_factor, blocked_edges):
    graph = build_graph(roads, traffic_factor, blocked_edges)
    queue = [(0, start, [start], 0)]
    best = {start: 0}
    while queue:
        cost, node, path, distance = heapq.heappop(queue)
        if node == end:
            return path, cost, distance
        if cost > best.get(node, float('inf')):
            continue
        for nxt, edge_cost, edge_distance in graph.get(node, []):
            new_cost = cost + edge_cost
            if new_cost < best.get(nxt, float('inf')):
                best[nxt] = new_cost
                heapq.heappush(queue, (new_cost, nxt, path + [nxt], distance + edge_distance))
    return [], float('inf'), float('inf')


def edge_set_from_path(path):
    return [tuple(sorted((path[i], path[i+1]))) for i in range(len(path)-1)]


def severity_to_risk(severity, confidence, casualties, fire, blocked, traffic):
    base = {'Slight Injury': 35, 'Serious Injury': 70, 'Fatal injury': 92}.get(severity, 55)
    score = base + min(casualties * 2, 16)
    if fire:
        score += 10
    if blocked:
        score += 5
    if traffic == 'High':
        score += 4
    score = int(max(0, min(99, score)))
    if score >= 85:
        level = 'CRITICAL 🔴'
    elif score >= 65:
        level = 'HIGH 🟠'
    elif score >= 45:
        level = 'MODERATE 🟡'
    else:
        level = 'LOW 🟢'
    return score, level


def hospital_score(hospital, accident_lat, accident_lon, severity, casualties):
    distance = ((hospital.lat - accident_lat) ** 2 + (hospital.lon - accident_lon) ** 2) ** 0.5 * 111
    severity_weight = 1.5 if severity == 'Fatal injury' else 1.0
    capacity = hospital.beds + hospital.icu * 3
    score = distance * 3 + (1 / max(capacity, 1)) * 100 * severity_weight
    if hospital.status == 'Limited':
        score += 8
    if casualties > hospital.emergency_capacity:
        score += 50
    if severity == 'Fatal injury' and hospital.icu == 0:
        score += 40
    return distance, score


def nearest_available(df, node_distance):
    available = df[df['status'] == 'Available'].copy()
    if available.empty:
        return None
    available['sim_distance'] = available['node'].map(node_distance)
    return available.sort_values('sim_distance').iloc[0]


def node_distance_map(roads, traffic_factor, blocked_edges, target):
    nodes = sorted(set([x[0] for x in roads] + [x[1] for x in roads]))
    result = {}
    for node in nodes:
        _, cost, distance = dijkstra(roads, node, target, traffic_factor, blocked_edges)
        result[node] = distance if np.isfinite(distance) else 999
    return result


def map_points(accident_node, selected_ambulances, selected_fire, hospital, node_coords):
    rows = []
    lat, lon = node_coords[accident_node]
    rows.append([lat, lon, 'Accident'])
    for _, row in selected_ambulances.iterrows():
        rows.append([row.lat, row.lon, row.id])
    for _, row in selected_fire.iterrows():
        rows.append([row.lat, row.lon, row.id])
    rows.append([hospital.lat, hospital.lon, hospital.name])
    return pd.DataFrame(rows, columns=['lat','lon','type'])

try:
    df = load_data()
    model, accuracy, report, model_features = train_model(df)
except Exception as e:
    st.error(f'Unable to load or train the model: {e}')
    st.stop()

hospitals, ambulances, fire_trucks, roads = make_simulation_data()
node_coords = {
    'A': (30.040,31.225), 'B': (30.045,31.245), 'C': (30.060,31.255),
    'D': (30.055,31.230), 'E': (30.050,31.240), 'F': (30.040,31.260),
    'G': (30.025,31.225), 'H': (30.030,31.245), 'I': (30.035,31.265)
}

st.title('🏙️ NEXUS – Autonomous City Intelligence')
st.caption('AI-based emergency management simulation using real accident data and simulated city infrastructure.')

with st.sidebar:
    st.header('🕹️ NEXUS Control Center')
    zone = st.selectbox('Accident Zone', ['Zone A - Downtown','Zone B - Industrial','Zone C - Highway','Zone D - Residential'])
    weather_options = sorted(df['Weather_conditions'].dropna().astype(str).unique().tolist())
    light_options = sorted(df['Light_conditions'].dropna().astype(str).unique().tolist())
    road_options = sorted(df['Road_surface_conditions'].dropna().astype(str).unique().tolist())
    surface_type_options = sorted(df['Road_surface_type'].dropna().astype(str).unique().tolist())
    junction_options = sorted(df['Types_of_Junction'].dropna().astype(str).unique().tolist())
    collision_options = sorted(df['Type_of_collision'].dropna().astype(str).unique().tolist())
    cause_options = sorted(df['Cause_of_accident'].dropna().astype(str).unique().tolist())
    weather = st.selectbox('Weather', weather_options)
    light = st.selectbox('Light Conditions', light_options)
    road_condition = st.selectbox('Road Surface Conditions', road_options)
    road_type = st.selectbox('Road Surface Type', surface_type_options)
    junction = st.selectbox('Junction Type', junction_options)
    collision = st.selectbox('Collision Type', collision_options)
    cause = st.selectbox('Cause of Accident', cause_options)
    vehicles = st.slider('Vehicles Involved', 1, 7, 3)
    casualties = st.slider('Estimated Casualties', 1, 20, 5)
    fire_present = st.checkbox('🔥 Fire / Explosion', value=True)
    road_blocked = st.checkbox('🚧 Main Road Blocked', value=True)
    traffic_level = st.selectbox('Traffic Level', ['Low','Medium','High'], index=2)
    accident_node = st.selectbox('Accident Node', list(node_coords.keys()), index=4)
    start = st.button('🚨 START NEXUS')

st.info(f'Model trained on {len(df):,} real accident records. Test accuracy: {accuracy:.2%}')

if start:
    input_row = pd.DataFrame([{
        'Weather_conditions': weather,
        'Light_conditions': light,
        'Road_surface_conditions': road_condition,
        'Road_surface_type': road_type,
        'Types_of_Junction': junction,
        'Area_accident_occured': zone.split(' - ')[-1],
        'Type_of_collision': collision,
        'Number_of_vehicles_involved': vehicles,
        'Number_of_casualties': casualties,
        'Cause_of_accident': cause
    }])
    predicted = model.predict(input_row)[0]
    probabilities = model.predict_proba(input_row)[0]
    confidence = float(np.max(probabilities))
    risk_score, risk_level = severity_to_risk(predicted, confidence, casualties, fire_present, road_blocked, traffic_level)

    if risk_score >= 85:
        ambulance_count = 3
    elif risk_score >= 65:
        ambulance_count = 2
    else:
        ambulance_count = 1
    fire_count = 1 if fire_present else 0

    traffic_multiplier = {'Low': 1.0, 'Medium': 1.35, 'High': 1.8}[traffic_level]
    traffic_factor = {tuple(sorted((a,b))): 1.0 for a,b,_,_ in roads}
    for edge in traffic_factor:
        if random.random() < 0.35:
            traffic_factor[edge] = traffic_multiplier

    initial_blocked = set()
    if road_blocked:
        initial_blocked.add(tuple(sorted(('B','E'))))

    accident_lat, accident_lon = node_coords[accident_node]
    node_distances = node_distance_map(roads, traffic_factor, initial_blocked, accident_node)
    available_ambulances = ambulances[ambulances.status == 'Available'].copy()
    available_ambulances['distance'] = available_ambulances['node'].map(node_distances)
    selected_ambulances = available_ambulances.sort_values('distance').head(min(ambulance_count, len(available_ambulances)))

    available_fire = fire_trucks[fire_trucks.status == 'Available'].copy()
    available_fire['distance'] = available_fire['node'].map(node_distances)
    selected_fire = available_fire.sort_values('distance').head(min(fire_count, len(available_fire)))

    hospital_rows = []
    for _, hospital in hospitals.iterrows():
        distance, score = hospital_score(hospital, accident_lat, accident_lon, predicted, casualties)
        hospital_rows.append((score, distance, hospital))
    hospital_rows.sort(key=lambda x: x[0])
    selected_hospital = hospital_rows[0][2]
    hospital_distance = hospital_rows[0][1]

    route_start = selected_ambulances.iloc[0]['node'] if not selected_ambulances.empty else 'A'
    initial_path, initial_cost, initial_distance = dijkstra(roads, route_start, accident_node, traffic_factor, initial_blocked)
    if not initial_path:
        initial_path, initial_cost, initial_distance = dijkstra(roads, route_start, accident_node, {k:1.0 for k in traffic_factor}, set())

    hospital_node = min(node_coords, key=lambda n: ((node_coords[n][0]-selected_hospital.lat)**2 + (node_coords[n][1]-selected_hospital.lon)**2))
    hospital_path, hospital_cost, hospital_route_distance = dijkstra(roads, accident_node, hospital_node, traffic_factor, initial_blocked)
    if not hospital_path:
        hospital_path, hospital_cost, hospital_route_distance = dijkstra(roads, accident_node, hospital_node, {k:1.0 for k in traffic_factor}, set())

    eta = max(3, int(round(initial_cost)))

    st.error(f'🚨 ACTIVE INCIDENT — {zone} — {risk_level} — Risk Score: {risk_score}%')

    m1, m2, m3, m4 = st.columns(4)
    m1.metric('🤖 Predicted Severity', predicted)
    m2.metric('🚑 Ambulances', len(selected_ambulances))
    m3.metric('🚒 Fire Trucks', len(selected_fire))
    m4.metric('⏱️ ETA', f'{eta} min')

    st.markdown('---')
    st.subheader('🔄 NEXUS Real-Time Decision Pipeline')
    progress = st.progress(0)
    status = st.empty()
    steps = [
        f'🤖 ML Engine: predicted {predicted} with {confidence:.1%} confidence',
        f'⚠️ Risk Engine: {risk_level} ({risk_score}%)',
        f'🚑 Dispatch: {len(selected_ambulances)} ambulance unit(s) assigned',
        f'🚒 Fire Response: {len(selected_fire)} fire unit(s) assigned',
        f'🏥 Hospital Optimization: {selected_hospital.name} selected',
        f'🛣️ Dijkstra: optimal route calculated, distance {initial_distance:.1f} km',
        '🚦 Smart Infrastructure: emergency corridor simulation activated',
        '🔄 Dynamic Monitoring: incident state processed and response plan finalized'
    ]
    for i, step in enumerate(steps):
        status.write(step)
        progress.progress(int((i+1) * 100 / len(steps)))
        time.sleep(0.18)

    if road_blocked and initial_path:
        old_path = initial_path
        blocked_for_replan = set(initial_blocked)
        edge_to_block = tuple(sorted((old_path[0], old_path[1]))) if len(old_path) > 1 else None
        if edge_to_block:
            blocked_for_replan.add(edge_to_block)
        new_path, new_cost, new_distance = dijkstra(roads, route_start, accident_node, traffic_factor, blocked_for_replan)
        if new_path:
            replanning = True
        else:
            new_path, new_cost, new_distance = old_path, initial_cost, initial_distance
            replanning = False
    else:
        old_path = initial_path
        new_path = initial_path
        new_cost = initial_cost
        new_distance = initial_distance
        replanning = False

    st.success('✨ NEXUS emergency response simulation completed.')

    c1, c2 = st.columns(2)
    with c1:
        st.subheader('🗺️ City Simulation Map')
        points = map_points(accident_node, selected_ambulances, selected_fire, selected_hospital, node_coords)
        st.map(points[['lat','lon']], zoom=12)
        st.dataframe(points, use_container_width=True, hide_index=True)

    with c2:
        st.subheader('📋 AI Decision Log')
        log = pd.DataFrame({
            'Time': [
                'T+00s','T+01s','T+02s','T+03s','T+04s','T+05s','T+06s','T+07s'
            ],
            'Component': [
                'Accident Detector','Risk Model','Resource Allocator','Fire Response','Hospital Optimizer','Route Optimizer','Traffic Control','Dynamic Monitoring'
            ],
            'Decision': [
                f'Incident detected at {accident_node}',
                f'{predicted} / {confidence:.1%}',
                f'{len(selected_ambulances)} ambulance(s) assigned',
                f'{len(selected_fire)} fire truck(s) assigned',
                selected_hospital.name,
                ' → '.join(initial_path) if initial_path else 'No route found',
                'Emergency Corridor Activated',
                'Dynamic replanning checked'
            ]
        })
        st.dataframe(log, use_container_width=True, hide_index=True)

    r1, r2 = st.columns(2)
    with r1:
        st.subheader('🛣️ Optimal Route')
        st.write(' → '.join(initial_path) if initial_path else 'No route found')
        st.write(f'Distance: {initial_distance:.1f} km | Travel Cost: {initial_cost:.1f}')
        st.write('Hospital Route: ' + (' → '.join(hospital_path) if hospital_path else 'No route found'))
    with r2:
        st.subheader('🔄 Dynamic Replanning')
        if replanning:
            st.write('Previous Route ❌')
            st.code(' → '.join(old_path))
            st.write('New Route ✅')
            st.code(' → '.join(new_path))
            st.write(f'New Distance: {new_distance:.1f} km | New Travel Cost: {new_cost:.1f}')
        else:
            st.write('No route change was required in this simulation.')

    h1, h2, h3 = st.columns(3)
    with h1:
        st.metric('🏥 Selected Hospital', selected_hospital.name)
    with h2:
        st.metric('🛏️ Available Beds', int(selected_hospital.beds))
    with h3:
        st.metric('❤️ ICU Beds', int(selected_hospital.icu))

    st.subheader('🤖 Model Evaluation')
    e1, e2, e3 = st.columns(3)
    e1.metric('Accuracy', f'{accuracy:.2%}')
    e2.metric('Serious Injury Recall', f"{report.get('Serious Injury', {}).get('recall', 0):.2%}")
    e3.metric('Fatal Injury Recall', f"{report.get('Fatal injury', {}).get('recall', 0):.2%}")
else:
    st.subheader('🏙️ NEXUS Ready')
    st.write('Adjust the incident inputs from the control center, then press START NEXUS to run the complete emergency-response simulation.')
    a, b, c = st.columns(3)
    a.metric('Real Accident Records', f'{len(df):,}')
    b.metric('ML Test Accuracy', f'{accuracy:.2%}')
    c.metric('ML Features', len(model_features))
