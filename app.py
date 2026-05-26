import streamlit as st
import rebound
import reboundx
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const

# --- Constants ---
pi = const.pi
c = const.c

# --- Physics Functions ---
def seperation_distance_r(m_1, m_2, e_i, f_gw=10.0, G=6.67430e-11):
    f_orb = f_gw / 2.0  
    numerator = G * (m_1 + m_2)
    denominator = (2 * pi * f_orb * ((1 - e_i**2)**1.5) / ((1 + e_i)**1.1954))**2
    r = (numerator / denominator)**(1/3)
    return r

def isco_frequency(m_1, m_2):
    M_total_kg = (m_1 + m_2) * 1.989e30 
    G_si = 6.674e-11
    c_si = 3e8
    f_isco = (c_si**3) / (6**(3/2) * np.pi * G_si * M_total_kg)
    return f_isco

def radiation_reaction_force(sim_pointer):
    sim = sim_pointer.contents
    if sim.N < 2: return

    ps = sim.particles
    p1, p2 = ps[0], ps[1]
    
    # Relative position and velocity
    dx, dy, dz = p1.x - p2.x, p1.y - p2.y, p1.z - p2.z
    dvx, dvy, dvz = p1.vx - p2.vx, p1.vy - p2.vy, p1.vz - p2.vz

    r_sq = dx**2 + dy**2 + dz**2
    r = np.sqrt(r_sq)
    v_sq = dvx**2 + dvy**2 + dvz**2

    rv_dot = dx*dvx + dy*dvy + dz*dvz
    r_dot = rv_dot / r

    m1, m2 = p1.m, p2.m
    M = m1 + m2
    mu = m1 * m2 / M
    eta = mu / M
    G = sim.G
    
    A_const = (8.0/5.0) * (G * M * eta) / (r**3) * (G * M / (c**5))
    term_A = (17.0/3.0) * (G * M / r) + 3.0 * v_sq
    term_B = v_sq + 3.0 * (G * M / r)

    ax_rel = -A_const * (term_B * dvx - term_A * r_dot * (dx/r))
    ay_rel = -A_const * (term_B * dvy - term_A * r_dot * (dy/r))
    az_rel = -A_const * (term_B * dvz - term_A * r_dot * (dz/r))

    p1.ax += ax_rel * (m2 / M)
    p1.ay += ay_rel * (m2 / M)
    p1.az += az_rel * (m2 / M)

    p2.ax -= ax_rel * (m1 / M)
    p2.ay -= ay_rel * (m1 / M)
    p2.az -= az_rel * (m1 / M)

def run_simulation(m_1, m_2, e_i):
    sim = rebound.Simulation()
    sim.units = ("m", "s", "Msun")

    r = seperation_distance_r(m_1, m_2, e_i, f_gw=10.0, G=sim.G)
    f_isco = isco_frequency(m_1, m_2)

    sim.add(m=m_1)
    sim.add(m=m_2, a=r, e=e_i)
    sim.move_to_com()

    sim.integrator = "ias15"
    rebx = reboundx.Extras(sim)
    gr = rebx.load_force("gr")
    rebx.add_force(gr)
    gr.params["c"] = c
    sim.additional_forces = radiation_reaction_force
    sim.force_is_velocity_dependent = 1

    ps = sim.particles
    ps[0].r = 2 * sim.G * ps[0].m / (c**2)
    ps[1].r = 2 * sim.G * ps[1].m / (c**2)
    sim.collision = "direct"
    sim.collision_resolve = "merge"

    dt_record = 0.1
    time_data, separation_data, eccentricity_data = [], [], []
    
    while sim.N == 2:
        sim.integrate(sim.t + dt_record)
        if sim.N < 2:
            break
        
        orbit = sim.orbits()[0]
        current_r = np.sqrt((ps[1].x-ps[0].x)**2 + (ps[1].y-ps[0].y)**2 + (ps[1].z-ps[0].z)**2)
        
        time_data.append(sim.t)
        separation_data.append(current_r)
        eccentricity_data.append(orbit.e)
        
        f_orb = orbit.n / (2 * np.pi)
        peak_freq = f_orb * 2 * (1 + orbit.e)**1.1954 / (1 - orbit.e**2)**1.5
        
        if peak_freq > f_isco:
            break
            
    return np.array(time_data), np.array(separation_data), np.array(eccentricity_data)

# --- Streamlit UI ---
st.set_page_config(layout="wide", page_title="Binary Black Hole Merger Simulation")

st.title("Binary Black Hole Merger Simulation")
st.markdown("Simulate the separation and eccentricity decay of a binary black hole system due to gravitational radiation.")

# Sidebar for inputs
st.sidebar.header("Simulation Parameters")
m_1 = st.sidebar.slider("Mass 1 (Solar mass)", min_value=1.0, max_value=100.0, value=30.0, step=1.0)
m_2 = st.sidebar.slider("Mass 2 (Solar mass)", min_value=1.0, max_value=100.0, value=30.0, step=1.0)
e_i = st.sidebar.slider("Initial Eccentricity", min_value=0.01, max_value=0.99, value=0.20, step=0.01)

if st.sidebar.button("Run Simulation", type="primary"):
    with st.spinner("Integrating orbital dynamics... This may take a moment."):
        time, separation, eccentricity = run_simulation(m_1, m_2, e_i)
        
        # Plotting the results
        col1, col2 = st.columns(2)
        
        with col1:
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            ax1.plot(time, separation, lw=2, color='blue')
            ax1.set_xlabel("Time (s)")
            ax1.set_ylabel("Separation (m)")
            ax1.set_title("Separation vs Time")
            ax1.grid(True, alpha=0.3)
            st.pyplot(fig1)
            
        with col2:
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            ax2.plot(time, eccentricity, lw=2, color='crimson')
            ax2.set_xlabel("Time (s)")
            ax2.set_ylabel("Eccentricity")
            ax2.set_title("Eccentricity vs Time")
            ax2.grid(True, alpha=0.3)
            st.pyplot(fig2)
        
        st.success("Simulation Complete!")
else:
    st.info("Adjust the parameters in the sidebar and click 'Run Simulation' to see the decay charts.")
