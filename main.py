import marimo

__generated_with = "0.23.13"
app = marimo.App()


@app.cell
def _():
    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np
    import scipy.sparse as sp
    import stim
    import marimo as mo
    from ldpc.bp_decoder import BpDecoder
    from pyvis.network import Network
    import plotly.graph_objects as go
    import time



    return BpDecoder, Network, go, mo, np, nx, plt, sp, time


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Universal PyVis Helper Function
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Universal PyVis Helper Function
    """)
    return


@app.cell
def _(Network, mo, sp):
    def plot_tanner_pyvis(H: sp.csr_matrix, title: str = "Tanner Graph"):
        """
        Renders an interactive PyVis Tanner graph inside Marimo for ANY parity-check matrix H.
        """
        num_checks, num_data = H.shape

        # Initialize PyVis Network with dark theme
        net = Network(height="500px", width="100%", bgcolor="#1e1e1e", font_color="white")

        # Configure physics for drag-and-drop untangling
        net.barnes_hut(gravity=-2500, central_gravity=0.3, spring_length=80, spring_strength=0.05)

        # Check Nodes (Orange Squares)
        for i in range(num_checks):
            net.add_node(f"C{i}", label=f"C{i}", title=f"Check {i}", color="#ff7f0e", shape="square", size=15)

        # Data Qubits (Blue Circles)
        for j in range(num_data):
            net.add_node(f"D{j}", label=f"D{j}", title=f"Data Qubit {j}", color="#1f77b4", shape="dot", size=12)

        # Edges from Non-zero Entries in H
        H_coo = H.tocoo()
        for r, c in zip(H_coo.row, H_coo.col):
            net.add_edge(f"C{r}", f"D{c}", color="#666666", width=1.5)

        # Output raw HTML to Marimo
        return mo.vstack([
            mo.md(f"#### {title}"),
            mo.Html(net.generate_html())
        ])

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    3D Torus Grid Transformer
    """)
    return


@app.cell
def _(np):
    def grid_to_torus(r_idx, c_idx, L, R=3.0, r=1.2):
        """
        Maps continuous grid coordinates (r_idx, c_idx) in [0, L) x [0, L)
        to 3D Cartesian coordinates on a torus.
        """
        theta = 2 * np.pi * r_idx / L  # Major angle (around the main ring)
        phi = 2 * np.pi * c_idx / L    # Minor angle (around the tube)

        x = (R + r * np.cos(phi)) * np.cos(theta)
        y = (R + r * np.cos(phi)) * np.sin(theta)
        z = r * np.sin(phi)

        return x, y, z

    return (grid_to_torus,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Repitition Code
    """)
    return


@app.cell
def _(sp):
    def make_repetition_code(d: int):
        """Generates parity check matrix H for a distance-d repetition code."""
        num_checks = d - 1
        num_data = d

        rows, cols = [], []
        for i in range(num_checks):
            rows.extend([i, i])
            cols.extend([i, i + 1])

        data = [1] * len(rows)
        return sp.csr_matrix((data, (rows, cols)), shape=(num_checks, num_data))

    return (make_repetition_code,)


@app.cell
def _(mo):
    d_slider = mo.ui.slider(start=3, stop=15, step=1, value=5, label="Code Distance (d)")
    d_slider
    return (d_slider,)


@app.cell
def _(d_slider, make_repetition_code, nx, plt):
    d_val = d_slider.value
    H = make_repetition_code(d_val)
    num_checks, num_data = H.shape

    # 1. Build Bipartite Graph from Matrix H
    G = nx.Graph()
    data_nodes = [f"D{i}" for i in range(num_data)]
    check_nodes = [f"C{i}" for i in range(num_checks)]

    G.add_nodes_from(data_nodes, bipartite=0)
    G.add_nodes_from(check_nodes, bipartite=1)

    H_coo = H.tocoo()
    for r, c in zip(H_coo.row, H_coo.col):
        G.add_edge(f"C{r}", f"D{c}")

    # 2. Assign Spatial Positions for Bipartite Layout
    pos = {}
    for i, node in enumerate(data_nodes):
        pos[node] = (i, 1)
    for i, node in enumerate(check_nodes):
        pos[node] = (i + 0.5, 0)

    # 3. Render Tanner Graph
    fig, ax = plt.subplots(figsize=(8, 3))

    nx.draw_networkx_nodes(G, pos, nodelist=data_nodes, node_color="#1f77b4", node_size=600, label="Data Qubits", ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=check_nodes, node_color="#ff7f0e", node_shape="s", node_size=600, label="Check Ancillas", ax=ax)
    nx.draw_networkx_edges(G, pos, width=2, edge_color="gray", ax=ax)
    nx.draw_networkx_labels(G, pos, font_color="white", font_size=9, font_weight="bold", ax=ax)

    ax.set_title(f"Tanner Graph — Repetition Code (d = {d_val})")
    ax.legend(loc="upper right", frameon=True)
    ax.axis("off")

    plt.tight_layout()
    fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Shor Code
    """)
    return


@app.cell
def _(np, sp):
    def make_shor_code():
        """
        Generates the parity check matrix H for Shor's [[9, 1, 3]] code.
        - Physical Qubits (n): 9
        - Checks (m): 8 total (6 bit-flip checks X, 2 phase-flip checks Z)
        """
        # 6 Bit-flip check rows (Z-type checks inside each 3-qubit block)
        Hz = np.array([
            [1, 1, 0,  0, 0, 0,  0, 0, 0],
            [0, 1, 1,  0, 0, 0,  0, 0, 0],
            [0, 0, 0,  1, 1, 0,  0, 0, 0],
            [0, 0, 0,  0, 1, 1,  0, 0, 0],
            [0, 0, 0,  0, 0, 0,  1, 1, 0],
            [0, 0, 0,  0, 0, 0,  0, 1, 1]
        ], dtype=int)

        # 2 Phase-flip check rows (X-type checks across 3-qubit blocks)
        Hx = np.array([
            [1, 1, 1,  1, 1, 1,  0, 0, 0],
            [0, 0, 0,  1, 1, 1,  1, 1, 1]
        ], dtype=int)

        # Combine into full Parity Check Matrix (8 checks x 9 qubits)
        H_full = np.vstack([Hz, Hx])
        return sp.csr_matrix(H_full)

    return (make_shor_code,)


@app.cell
def _(make_shor_code, nx, plt):
    def plot_shor_tanner_graph():
        H_shor = make_shor_code()
        num_checks_shor, num_data_shor = H_shor.shape

        # 1. Construct NetworkX Bipartite Graph
        G_shor = nx.Graph()
        data_nodes = [f"D{i}" for i in range(num_data_shor)]

        # 6 Local Bit-Flip Checks (C0-C5) and 2 Global Phase-Flip Checks (C6-C7)
        check_nodes_local = [f"C{i}" for i in range(6)]
        check_nodes_global = [f"C{i}" for i in range(6, 8)]
        all_checks = check_nodes_local + check_nodes_global

        G_shor.add_nodes_from(data_nodes, bipartite=0)
        G_shor.add_nodes_from(all_checks, bipartite=1)

        H_coo = H_shor.tocoo()
        for r_idx, c_idx in zip(H_coo.row, H_coo.col):
            G_shor.add_edge(f"C{r_idx}", f"D{c_idx}")

        # 2. Hierarchical Block Layout (Reflecting the 3 concatenated triplets)
        pos_shor = {}

        # Data Qubits: Placed in 3 distinct horizontal clusters
        for i in range(9):
            block_idx = i // 3
            offset_within_block = i % 3
            x_pos = block_idx * 3 + offset_within_block * 0.7
            y_pos = 1.0
            pos_shor[f"D{i}"] = (x_pos, y_pos)

        # Local Bit-Flip Checks (Z-checks): Positioned directly below their respective triplets
        for i in range(6):
            block_idx = i // 2
            offset_within_block = i % 2
            x_pos = block_idx * 3 + offset_within_block * 0.7 + 0.35
            y_pos = 0.3
            pos_shor[f"C{i}"] = (x_pos, y_pos)

        # Global Phase-Flip Checks (X-checks): Positioned at the bottom spanning across blocks
        pos_shor["C6"] = (1.0, -0.4)
        pos_shor["C7"] = (4.0, -0.4)

        # 3. Render Graph
        fig_shor, ax_shor = plt.subplots(figsize=(10, 5))

        nx.draw_networkx_nodes(
            G_shor, pos_shor, nodelist=data_nodes, node_color="#1f77b4",
            node_size=500, label="Data Qubits (9 total)", ax=ax_shor
        )
        nx.draw_networkx_nodes(
            G_shor, pos_shor, nodelist=check_nodes_local, node_color="#ff7f0e",
            node_shape="s", node_size=450, label="Local Z-Checks (6)", ax=ax_shor
        )
        nx.draw_networkx_nodes(
            G_shor, pos_shor, nodelist=check_nodes_global, node_color="#d62728",
            node_shape="s", node_size=500, label="Global X-Checks (2)", ax=ax_shor
        )

        nx.draw_networkx_edges(G_shor, pos_shor, width=1.5, edge_color="gray", alpha=0.6, ax=ax_shor)
        nx.draw_networkx_labels(G_shor, pos_shor, font_color="white", font_size=8, font_weight="bold", ax=ax_shor)

        ax_shor.set_title("Tanner Graph — Shor [[9, 1, 3]] Code\n(Concatenated 3-Block Architecture)")
        ax_shor.legend(loc="upper right", frameon=True)
        ax_shor.axis("off")

        plt.tight_layout()
        return fig_shor

    # Plot the graph
    plot_shor_tanner_graph()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Generalized Bicycle Code
    """)
    return


@app.cell
def _(sp):
    def make_gb_code(l: int, a_powers: list, b_powers: list):
        """Generates the parity check matrix H for a Generalized Bicycle code."""
        # Cyclic shift matrix P of size l x l
        P = sp.diags([1], [-1], shape=(l, l), format="csr")
        P[0, l - 1] = 1

        # Construct polynomials A and B
        A = sum(P**p for p in a_powers).tocsr()
        A.data %= 2

        B = sum(P**p for p in b_powers).tocsr()
        B.data %= 2

        # Block matrix H_X = [A | B]
        HX = sp.hstack([A, B], format="csr")
        return HX

    return (make_gb_code,)


@app.cell
def _(mo):
    l_torus_slider = mo.ui.slider(start=3, stop=8, step=1, value=4, label="Torus Dimension (L)")
    l_torus_slider
    return (l_torus_slider,)


@app.cell
def _(go, grid_to_torus, l_torus_slider, np):
    def plot_3d_toric_code(L_val):
        # 1. Generate Continuous Torus Surface Mesh (Semi-transparent background)
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, 2 * np.pi, 50)
        u_grid, v_grid = np.meshgrid(u, v)

        R_maj, r_min = 3.0, 1.2
        x_surf = (R_maj + r_min * np.cos(v_grid)) * np.cos(u_grid)
        y_surf = (R_maj + r_min * np.cos(v_grid)) * np.sin(u_grid)
        z_surf = r_min * np.sin(v_grid)

        torus_surface = go.Surface(
            x=x_surf, y=y_surf, z=z_surf,
            colorscale=[[0, "#2c3e50"], [1, "#2c3e50"]],
            opacity=0.15,
            showscale=False,
            hoverinfo="none"
        )

        # 2. Extract Discrete Toric Code Components
        # Star Checks Z (Vertices) @ integer grid points (r, c)
        star_x, star_y, star_z = [], [], []
        for r in range(L_val):
            for c in range(L_val):
                x, y, z = grid_to_torus(r, c, L_val, R_maj, r_min)
                star_x.append(x); star_y.append(y); star_z.append(z)

        # Plaquette Checks X (Faces) @ half-integer grid centers (r+0.5, c+0.5)
        plaq_x, plaq_y, plaq_z = [], [], []
        for r in range(L_val):
            for c in range(L_val):
                x, y, z = grid_to_torus(r + 0.5, c + 0.5, L_val, R_maj, r_min)
                plaq_x.append(x); plaq_y.append(y); plaq_z.append(z)

        # Data Qubits (Edges): Horizontal (r+0.5, c) and Vertical (r, c+0.5)
        data_x, data_y, data_z = [], [], []
        for r in range(L_val):
            for c in range(L_val):
                # Horizontal qubit
                x_h, y_h, z_h = grid_to_torus(r + 0.5, c, L_val, R_maj, r_min)
                data_x.append(x_h); data_y.append(y_h); data_z.append(z_h)
                # Vertical qubit
                x_v, y_v, z_v = grid_to_torus(r, c + 0.5, L_val, R_maj, r_min)
                data_x.append(x_v); data_y.append(y_v); data_z.append(z_v)

        # 3. Generate Curved Edge Grid Lines wrapping around the torus
        edge_x, edge_y, edge_z = [], [], []
        n_samples = 15

        for r in range(L_val):
            for c in range(L_val):
                # Horizontal segment (from (r,c) to (r+1,c))
                r_samples = np.linspace(r, r + 1, n_samples)
                for r_s in r_samples:
                    x, y, z = grid_to_torus(r_s, c, L_val, R_maj, r_min)
                    edge_x.append(x); edge_y.append(y); edge_z.append(z)
                edge_x.append(None); edge_y.append(None); edge_z.append(None)

                # Vertical segment (from (r,c) to (r,c+1))
                c_samples = np.linspace(c, c + 1, n_samples)
                for c_s in c_samples:
                    x, y, z = grid_to_torus(r, c_s, L_val, R_maj, r_min)
                    edge_x.append(x); edge_y.append(y); edge_z.append(z)
                edge_x.append(None); edge_y.append(None); edge_z.append(None)

        grid_trace = go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            mode="lines",
            line=dict(color="#888888", width=3),
            name="Lattice Edges",
            hoverinfo="none"
        )

        # 4. Create Scatter Traces for Nodes
        stars_trace = go.Scatter3d(
            x=star_x, y=star_y, z=star_z,
            mode="markers",
            marker=dict(size=7, color="#ff7f0e", symbol="circle"),
            name="Star Checks Z (Vertices)",
            hovertext=[f"Star Check A_{i}" for i in range(len(star_x))]
        )

        plaques_trace = go.Scatter3d(
            x=plaq_x, y=plaq_y, z=plaq_z,
            mode="markers",
            marker=dict(size=7, color="#d62728", symbol="square"),
            name="Plaquette Checks X (Faces)",
            hovertext=[f"Plaquette Check B_{i}" for i in range(len(plaq_x))]
        )

        data_trace = go.Scatter3d(
            x=data_x, y=data_y, z=data_z,
            mode="markers",
            marker=dict(size=5, color="#1f77b4", symbol="circle"),
            name="Data Qubits (Edges)",
            hovertext=[f"Data Qubit D_{i}" for i in range(len(data_x))]
        )

        # 5. Assemble Figure
        fig = go.Figure(data=[torus_surface, grid_trace, stars_trace, plaques_trace, data_trace])

        fig.update_layout(
            title=f"3D Surface Embedding — Toric Code [[2×{L_val}², 2, {L_val}]]",
            template="plotly_dark",
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                aspectmode="data"
            ),
            margin=dict(l=0, r=0, b=0, t=40),
            legend=dict(x=0.05, y=0.95)
        )

        return fig

    # Render dynamically in Marimo
    plot_3d_toric_code(l_torus_slider.value)
    return


@app.cell
def _(mo):
    l_slider = mo.ui.slider(start=5, stop=21, step=2, value=7, label="Block Size (l)")
    l_slider
    return (l_slider,)


@app.cell
def _(Network, l_slider, make_gb_code, mo):
    def plot_gb_pyvis(l_val: int):
        a_p = [0, 1, 6 % l_val]
        b_p = [0, 3 % l_val, 7 % l_val]

        H_gb = make_gb_code(l_val, a_p, b_p)
        num_checks_gb, num_data_gb = H_gb.shape

        # 1. Configure PyVis network
        net = Network(
            height="500px",
            width="100%",
            bgcolor="#1e1e1e",
            font_color="white",
            cdn_resources="remote"
        )

        # 2. Configure Barnes-Hut physics engine
        net.barnes_hut(
            gravity=-3000,
            central_gravity=0.3,
            spring_length=90,
            spring_strength=0.05
        )

        # 3. Add Check Nodes
        for i in range(num_checks_gb):
            net.add_node(f"C{i}", label=f"C{i}", title=f"Check {i}", color="#ff7f0e", shape="square", size=15)

        # 4. Add Data Qubit Nodes
        for j in range(num_data_gb):
            is_block_b = j >= l_val
            node_color = "#2ca02c" if is_block_b else "#1f77b4"
            net.add_node(f"D{j}", label=f"D{j}", title=f"Data Qubit {j}", color=node_color, shape="dot", size=12)

        # 5. Add Edges
        H_coo_gb = H_gb.tocoo()
        for r_idx, c_idx in zip(H_coo_gb.row, H_coo_gb.col):
            net.add_edge(f"C{r_idx}", f"D{c_idx}", color="#888888", width=1.2)

        # 6. Generate HTML payload string
        html_content = net.generate_html()

        # 7. Render via lowercase mo.iframe (executes PyVis JS engine safely)
        return mo.vstack([
            mo.md(f"#### Interactive GB Tanner Graph — $l = {l_val}$ ($N = {num_data_gb}$ Data Qubits, ${num_checks_gb}$ Checks)"),
            mo.iframe(html_content, width="100%", height="520px")
        ])

    # Execute with slider
    plot_gb_pyvis(l_slider.value)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Toric Code
    """)
    return


@app.cell
def _(sp):
    def make_toric_code(L: int):
        """
        Generates parity check matrices H_X and H_Z for a distance-L Toric Code on an L x L torus.

        - Physical Qubits (n): 2 * L^2 (1 horizontal qubit + 1 vertical qubit per vertex)
        - Check Nodes: L^2 vertex checks (H_Z) + L^2 face checks (H_X)
        """
        N_vertices = L * L
        n_qubits = 2 * N_vertices

        # Vertex indexing: v = row * L + col
        # Horizontal qubit at v: index 2*v
        # Vertical qubit at v:   index 2*v + 1

        # 1. Construct H_Z (Star / Vertex Checks)
        # Each vertex v connects to 4 incident edges: 
        # H(v, col), H(v - 1 mod L, col), V(v, row), V(v - L mod L^2)
        Hz_rows, Hz_cols = [], []
        for r in range(L):
            for c in range(L):
                v = r * L + c
                # Incident horizontal edges
                e_h1 = 2 * v
                e_h2 = 2 * (r * L + (c - 1) % L)
                # Incident vertical edges
                e_v1 = 2 * v + 1
                e_v2 = 2 * (((r - 1) % L) * L + c) + 1

                for e in [e_h1, e_h2, e_v1, e_v2]:
                    Hz_rows.append(v)
                    Hz_cols.append(e)

        Hz = sp.csr_matrix(([1] * len(Hz_rows), (Hz_rows, Hz_cols)), shape=(N_vertices, n_qubits))
        Hz.data %= 2

        # 2. Construct H_X (Plaquette / Face Checks)
        # Each face f (indexed by its top-left vertex) bounds 4 edges
        Hx_rows, Hx_cols = [], []
        for r in range(L):
            for c in range(L):
                f = r * L + c
                e_h1 = 2 * f
                e_h2 = 2 * (((r + 1) % L) * L + c)
                e_v1 = 2 * f + 1
                e_v2 = 2 * (r * L + (c + 1) % L) + 1

                for e in [e_h1, e_h2, e_v1, e_v2]:
                    Hx_rows.append(f)
                    Hx_cols.append(e)

        Hx = sp.csr_matrix(([1] * len(Hx_rows), (Hx_rows, Hx_cols)), shape=(N_vertices, n_qubits))
        Hx.data %= 2

        return Hx, Hz

    return (make_toric_code,)


@app.cell
def _(mo):
    l_toric_slider = mo.ui.slider(start=2, stop=6, step=1, value=3, label="Lattice Dimension (L)")
    l_toric_slider
    return (l_toric_slider,)


@app.cell
def _(l_toric_slider, make_toric_code, nx, plt, sp):
    def plot_toric_tanner_graph(L_val):
        Hx, Hz = make_toric_code(L_val)
        num_faces, n_qubits = Hx.shape
        num_vertices = Hz.shape[0]

        # Combine H_X and H_Z into a single joint Tanner check matrix
        H_joint = sp.vstack([Hz, Hx])
        total_checks = H_joint.shape[0]

        # 1. Build Bipartite NetworkX Graph
        G_toric = nx.Graph()
        data_nodes = [f"D{i}" for i in range(n_qubits)]
        star_checks = [f"A{i}" for i in range(num_vertices)]    # Z-type Star checks
        plaquette_checks = [f"B{i}" for i in range(num_faces)] # X-type Plaquette checks
        all_checks = star_checks + plaquette_checks

        G_toric.add_nodes_from(data_nodes, bipartite=0)
        G_toric.add_nodes_from(all_checks, bipartite=1)

        H_coo = H_joint.tocoo()
        for r_idx, c_idx in zip(H_coo.row, H_coo.col):
            check_name = star_checks[r_idx] if r_idx < num_vertices else plaquette_checks[r_idx - num_vertices]
            G_toric.add_edge(check_name, f"D{c_idx}")

        # 2. Assign Spatial Positions Matching the Physical 2D Lattice
        pos_toric = {}

        # Star Checks (Vertices) positioned at integer grid intersections (r, c)
        for r in range(L_val):
            for c in range(L_val):
                v_idx = r * L_val + c
                pos_toric[f"A{v_idx}"] = (c, -r)

        # Plaquette Checks (Faces) positioned at half-integer grid centers (r + 0.5, c + 0.5)
        for r in range(L_val):
            for c in range(L_val):
                f_idx = r * L_val + c
                pos_toric[f"B{f_idx}"] = (c + 0.5, -r - 0.5)

        # Data Qubits placed directly on the edges between vertices
        for r in range(L_val):
            for c in range(L_val):
                v_idx = r * L_val + c
                # Horizontal Qubit (offset right)
                pos_toric[f"D{2 * v_idx}"] = (c + 0.5, -r)
                # Vertical Qubit (offset down)
                pos_toric[f"D{2 * v_idx + 1}"] = (c, -r - 0.5)

        # 3. Render Plot
        fig_toric, ax_toric = plt.subplots(figsize=(8, 8))

        nx.draw_networkx_nodes(
            G_toric, pos_toric, nodelist=data_nodes, node_color="#1f77b4",
            node_size=180, label="Data Qubits (Edges)", ax=ax_toric
        )
        nx.draw_networkx_nodes(
            G_toric, pos_toric, nodelist=star_checks, node_color="#ff7f0e",
            node_shape="o", node_size=250, label="Star Checks Z (Vertices)", ax=ax_toric
        )
        nx.draw_networkx_nodes(
            G_toric, pos_toric, nodelist=plaquette_checks, node_color="#d62728",
            node_shape="s", node_size=250, label="Plaquette Checks X (Faces)", ax=ax_toric
        )

        nx.draw_networkx_edges(G_toric, pos_toric, width=0.8, edge_color="gray", alpha=0.5, ax=ax_toric)
        nx.draw_networkx_labels(G_toric, pos_toric, font_color="white", font_size=6, font_weight="bold", ax=ax_toric)

        ax_toric.set_title(f"Tanner Graph — Toric Code [[{n_qubits}, 2, {L_val}]]\n({L_val}x{L_val} Toroidal Grid)")
        ax_toric.legend(loc="upper right", frameon=True)
        ax_toric.axis("off")

        plt.tight_layout()
        return fig_toric

    # Render plot using slider input
    plot_toric_tanner_graph(l_toric_slider.value)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Bivariate Bicycle Code
    """)
    return


@app.cell
def _(sp):
    def make_bb_code(L: int, M: int, a_terms: list, b_terms: list):
        """
        Generates H_X for a Bivariate Bicycle code defined on an L x M grid.
        a_terms, b_terms: list of tuples (x_power, y_power)
        """
        # Core 1D cyclic permutation matrices
        P_L = sp.diags([1], [-1], shape=(L, L), format="csr")
        P_L[0, L - 1] = 1

        P_M = sp.diags([1], [-1], shape=(M, M), format="csr")
        P_M[0, M - 1] = 1

        # 2D shift operators
        x = sp.kron(P_L, sp.eye(M), format="csr")
        y = sp.kron(sp.eye(L), P_M, format="csr")

        # Build A and B polynomials
        A = sum((x**px) @ (y**py) for px, py in a_terms).tocsr()
        A.data %= 2

        B = sum((x**px) @ (y**py) for px, py in b_terms).tocsr()
        B.data %= 2

        # Block matrix H_X = [A | B]
        return sp.hstack([A, B], format="csr")

    return (make_bb_code,)


@app.cell
def _(mo):
    # Preset dictionary containing the 3 gold-standard BB codes
    BB_PRESETS = {
        "[[72, 12, 6]] (L=6, M=6)": {
            "L": 6, "M": 6,
            "a_terms": [(3, 0), (0, 1), (0, 2)],  # A = x^3 + y + y^2
            "b_terms": [(0, 3), (1, 0), (2, 0)],  # B = y^3 + x + x^2
        },
        "[[144, 12, 12]] (L=12, M=6)": {
            "L": 12, "M": 6,
            "a_terms": [(3, 0), (0, 1), (0, 2)],  # A = x^3 + y + y^2
            "b_terms": [(0, 3), (1, 0), (2, 0)],  # B = y^3 + x + x^2
        },
        "[[288, 12, 18]] (L=12, M=12)": {
            "L": 12, "M": 12,
            "a_terms": [(3, 0), (0, 1), (0, 2)],  # A = x^3 + y + y^2
            "b_terms": [(0, 3), (1, 0), (2, 0)],  # B = y^3 + x + x^2
        },
    }

    code_dropdown = mo.ui.dropdown(
        options=list(BB_PRESETS.keys()),
        value="[[144, 12, 12]] (L=12, M=6)",
        label="Select Bivariate Bicycle Code:"
    )
    code_dropdown
    return BB_PRESETS, code_dropdown


@app.cell
def _(BB_PRESETS, code_dropdown, make_bb_code, np, nx, plt):
    def plot_selected_bb_code(selected_key):
        preset = BB_PRESETS[selected_key]
        L_val, M_val = preset["L"], preset["M"]
        a_terms, b_terms = preset["a_terms"], preset["b_terms"]

        H_bb = make_bb_code(L_val, M_val, a_terms, b_terms)
        num_checks_bb, num_data_bb = H_bb.shape
        N_sub = L_val * M_val  # Size of one block

        # 1. Build Bipartite NetworkX Graph
        G_bb = nx.Graph()
        data_nodes_bb = [f"D{i}" for i in range(num_data_bb)]
        check_nodes_bb = [f"C{i}" for i in range(num_checks_bb)]

        G_bb.add_nodes_from(data_nodes_bb, bipartite=0)
        G_bb.add_nodes_from(check_nodes_bb, bipartite=1)

        H_coo_bb = H_bb.tocoo()
        for r_idx, c_idx in zip(H_coo_bb.row, H_coo_bb.col):
            G_bb.add_edge(f"C{r_idx}", f"D{c_idx}")

        # 2. Concentric Ring Layout (Equidistant Angular Spacing)
        pos_bb = {}

        # Ring 1: Checks (Inner Ring, radius = 1.0)
        r_checks = 1.0
        for i in range(N_sub):
            angle = 2 * np.pi * i / N_sub
            pos_bb[f"C{i}"] = (r_checks * np.cos(angle), r_checks * np.sin(angle))

        # Ring 2: Block A Data Qubits (Middle Ring, radius = 2.0)
        r_data_A = 2.0
        for i in range(N_sub):
            angle = 2 * np.pi * i / N_sub
            pos_bb[f"D{i}"] = (r_data_A * np.cos(angle), r_data_A * np.sin(angle))

        # Ring 3: Block B Data Qubits (Outer Ring, radius = 3.0)
        r_data_B = 3.0
        for i in range(N_sub):
            angle = 2 * np.pi * (i + 0.5) / N_sub
            pos_bb[f"D{i + N_sub}"] = (r_data_B * np.cos(angle), r_data_B * np.sin(angle))

        # 3. Render Graph
        fig_bb, ax_bb = plt.subplots(figsize=(8, 8))

        data_A_list = [f"D{i}" for i in range(N_sub)]
        data_B_list = [f"D{i + N_sub}" for i in range(N_sub)]

        nx.draw_networkx_nodes(
            G_bb, pos_bb, nodelist=check_nodes_bb, node_color="#ff7f0e", 
            node_shape="s", node_size=180, label="Check Nodes (Inner Ring)", ax=ax_bb
        )
        nx.draw_networkx_nodes(
            G_bb, pos_bb, nodelist=data_A_list, node_color="#1f77b4", 
            node_size=140, label="Block A Qubits (Middle Ring)", ax=ax_bb
        )
        nx.draw_networkx_nodes(
            G_bb, pos_bb, nodelist=data_B_list, node_color="#2ca02c", 
            node_size=140, label="Block B Qubits (Outer Ring)", ax=ax_bb
        )

        nx.draw_networkx_edges(G_bb, pos_bb, width=0.4, edge_color="gray", alpha=0.3, ax=ax_bb)

        ax_bb.set_title(f"Tanner Graph — {selected_key}\n(Total Qubits N = {num_data_bb}, Checks = {num_checks_bb})")
        ax_bb.legend(loc="upper right", frameon=True)
        ax_bb.axis("off")

        plt.tight_layout()
        return fig_bb

    # Render using the dropdown value
    plot_selected_bb_code(code_dropdown.value)
    return


@app.cell
def _(
    make_bb_code,
    make_gb_code,
    make_repetition_code,
    make_shor_code,
    make_toric_code,
    mo,
):
    # 1. Repetition Code
    d_rep = 5
    H_rep = make_repetition_code(d_rep)

    # 2. Shor's Code (Slicing your combined matrix)
    H_shor = make_shor_code()
    H_Z_shor = H_shor[:6, :]  # Rows 0 to 5 (Bit-flip / Z-checks)
    H_X_shor = H_shor[6:, :]  # Rows 6 and 7 (Phase-flip / X-checks)

    # 3. Generalized Bicycle Code (1D)
    l_gb = 5
    H_gb = make_gb_code(l_gb, [0, 1, 2], [0, 2, 3])

    # 4. Bivariate Bicycle Code (2D)
    l1_bb, l2_bb = 3, 3
    a_exps = [(0, 0), (1, 0), (0, 1)]
    b_exps = [(0, 0), (0, 1), (1, 1)]
    H_bb = make_bb_code(l1_bb, l2_bb, a_exps, b_exps)

    # 5. Toric Code
    L_toric = 3
    H_X_toric, H_Z_toric = make_toric_code(L_toric)

    # Render All Parity-Check Matrices
    mo.vstack([
        mo.md("# 📊 Parity-Check Matrices Summary"),

        # Repetition Code
        mo.md(f"### 1. Repetition Code ($d = {d_rep}$)"),
        mo.md(f"**Shape:** {H_rep.shape} | **Matrix $H$:**"),
        mo.md(f"```text\n{H_rep.toarray()}\n```"),
        mo.md("---"),

        # Shor's Code
        mo.md("### 2. Shor's [[9, 1, 3]] Code"),
        mo.md(f"**Full Combined Matrix $H$ — Shape {H_shor.shape}:**"),
        mo.md(f"```text\n{H_shor.toarray()}\n```"),
        mo.md(f"**Bit-Flip Checks ($H_Z$, Rows 0-5) — Shape {H_Z_shor.shape}:**"),
        mo.md(f"```text\n{H_Z_shor.toarray()}\n```"),
        mo.md(f"**Phase-Flip Checks ($H_X$, Rows 6-7) — Shape {H_X_shor.shape}:**"),
        mo.md(f"```text\n{H_X_shor.toarray()}\n```"),
        mo.md("---"),

        # Generalized Bicycle Code
        mo.md(f"### 3. Generalized Bicycle Code ($l = {l_gb}$) [1D]"),
        mo.md(f"**Shape:** {H_gb.shape} | **Matrix $H = [A \mid B]$:**"),
        mo.md(f"```text\n{H_gb.toarray()}\n```"),
        mo.md("---"),

        # Bivariate Bicycle Code
        mo.md(f"### 4. Bivariate Bicycle Code ($l_1 = {l1_bb}, l_2 = {l2_bb}$) [2D]"),
        mo.md(f"**Shape:** {H_bb.shape} | **Matrix $H = [A \mid B]$:**"),
        mo.md(f"```text\n{H_bb.toarray()}\n```"),
        mo.md("---"),

        # Toric Code
        mo.md(f"### 5. Toric Code ($L = {L_toric}$)"),
        mo.md(f"**Data Qubits:** $N = {2 * L_toric**2}$"),
        mo.md(f"**Plaquette Checks ($H_X$) — Shape {H_X_toric.shape}:**"),
        mo.md(f"```text\n{H_X_toric.toarray()}\n```"),
        mo.md(f"**Star Checks ($H_Z$) — Shape {H_Z_toric.shape}:**"),
        mo.md(f"```text\n{H_Z_toric.toarray()}\n```"),
    ])
    return (H_gb,)


@app.cell(hide_code=True)
def _():
    return


@app.cell
def _(BpDecoder, H_gb, mo, np, sp):
    def run_bp_and_extract_soft_info(H: sp.csr_matrix, error_rate: float = 0.15, max_iter: int = 4000):
        """
        Runs BP decoding using BpDecoder and extracts soft LLR marginals from failed or converged runs.
        """
        num_checks, num_bits = H.shape

        # 1. Generate a random physical error vector and syndrome
        np.random.seed(42)  # Fixed seed to simulate a tricky error pattern
        true_error = (np.random.rand(num_bits) < error_rate).astype(np.uint8)
        syndrome = (H @ true_error) % 2

        # 2. Initialize BpDecoder
        decoder = BpDecoder(
            H,
            error_rate=error_rate,
            max_iter=max_iter,
            bp_method="product_sum",  # Options: 'product_sum', 'min_sum'
            input_vector_type="syndrome"
        )

        # 3. Execute Decoding
        decoding_result = decoder.decode(syndrome)

        # 4. Correct Attribute Access
        converged = bool(decoder.converge)  # Fix: decoder.converge (without 'd')
        soft_llrs = decoder.log_prob_ratios   # Posterior LLR array
        soft_probs = 1.0 / (1.0 + np.exp(soft_llrs))  # Soft error probabilities P(e_i = 1)
        hard_decisions = decoder.decoding.astype(int)

        return {
            "true_error": true_error,
            "syndrome": syndrome,
            "converged": converged,
            "soft_llrs": soft_llrs,
            "soft_probs": soft_probs,
            "hard_decisions": hard_decisions,
            "num_bits": num_bits
        }

    # Execute on Generalized Bicycle Code
    bp_res = run_bp_and_extract_soft_info(H_gb, error_rate=0.15, max_iter=4000)

    # Display Execution Summary
    status = "✅ BP Converged" if bp_res["converged"] else "❌ BP Failed to Converge (Trapped in Cycle)"

    mo.vstack([
        mo.md(f"### BP Execution Status: **{status}**"),
        mo.md(f"**True Physical Error:**\n```text\n{bp_res['true_error']}\n```"),
        mo.md(f"**BP Hard Output:**\n```text\n{bp_res['hard_decisions']}\n```"),
        mo.md(f"**Extracted Soft LLRs (Marginals):**\n```text\n{np.round(bp_res['soft_llrs'], 2)}\n```"),
        mo.md(f"**Soft Error Probabilities $P(e_i = 1)$:**\n```text\n{np.round(bp_res['soft_probs'], 3)}\n```")
    ])
    return (bp_res,)


@app.cell
def _(bp_res, go, np):
    def plot_bp_soft_llrs(res):
        bits = np.arange(res["num_bits"])
        llrs = res["soft_llrs"]
        true_err = res["true_error"]

        # Red bar = True physical error exists; Blue bar = No physical error
        bar_colors = ["#d62728" if err == 1 else "#1f77b4" for err in true_err]

        fig = go.Figure()

        # Soft LLR Bar Chart
        fig.add_trace(go.Bar(
            x=bits,
            y=llrs,
            marker_color=bar_colors,
            text=[f"Qubit D{b}<br>LLR: {l:.2f}<br>True Error: {e}" for b, l, e in zip(bits, llrs, true_err)],
            hoverinfo="text",
            name="Posterior LLR"
        ))

        # Highlight the ambiguity region where BP oscillates or gets stuck
        fig.add_hrect(
            y0=-1.5, y1=1.5,
            fillcolor="#ffbb00", opacity=0.2,
            line_width=0,
            annotation_text="Ambiguity Region (Trapping Set / Oscillating Marginals)",
            annotation_position="top left"
        )

        fig.update_layout(
            title=f"Soft Output: Marginal Log-Likelihood Ratios (LLRs) | Converged: {res['converged']}",
            xaxis_title="Data Qubit Index",
            yaxis_title="Posterior LLR (+Val = Confident 0, -Val = Confident 1)",
            template="plotly_dark",
            height=450
        )

        return fig

    # Render Plotly Chart
    plot_bp_soft_llrs(bp_res)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


@app.cell
def _(np, sp):


    def sysmatch_bb_decoder(H: sp.csr_matrix, syndrome: np.ndarray):
        """
        GF(2) Exact Syndrome Matching Decoder for BB Codes.
        Solves H @ e = syndrome (mod 2) using GF(2) elimination.
        """
        H_dense = H.toarray().astype(int) % 2
        num_checks, num_bits = H_dense.shape
        s = syndrome.copy().astype(int) % 2

        if not np.any(s):
            return np.zeros(num_bits, dtype=int), True

        # Build augmented matrix [H | s] over GF(2)
        aug = np.hstack([H_dense, s.reshape(-1, 1)])
        rows, cols = aug.shape

        # Forward Gaussian elimination over GF(2)
        pivot_row = 0
        pivot_cols = []

        for col in range(num_bits):
            swap_row = -1
            for r in range(pivot_row, rows):
                if aug[r, col] == 1:
                    swap_row = r
                    break

            if swap_row == -1:
                continue

            # Swap rows
            aug[[pivot_row, swap_row]] = aug[[swap_row, pivot_row]]

            # Eliminate column entries
            for r in range(rows):
                if r != pivot_row and aug[r, col] == 1:
                    aug[r] ^= aug[pivot_row]

            pivot_cols.append(col)
            pivot_row += 1
            if pivot_row >= rows:
                break

        # Check for inconsistency: [0 0 ... 0 | 1]
        for r in range(pivot_row, rows):
            if aug[r, -1] == 1:
                return np.zeros(num_bits, dtype=int), False

        # Back-substitution for candidate error vector
        e = np.zeros(num_bits, dtype=int)
        for i, col in enumerate(pivot_cols):
            e[col] = aug[i, -1]

        # Verify residual syndrome: S_residual = H @ e (mod 2)
        residual = (H_dense @ e) % 2
        success = np.array_equal(residual, s)

        return e, success



    return (sysmatch_bb_decoder,)


@app.cell
def _(BB_PRESETS, code_dropdown, make_bb_code, mo, np, sysmatch_bb_decoder):
    # --- Reactive Pipeline linked to code_dropdown ---
    selected_preset_key = code_dropdown.value
    preset_data = BB_PRESETS[selected_preset_key]

    # 1. Build H matrix using your existing make_bb_code function
    H_bb_selected = make_bb_code(
        L=preset_data["L"],
        M=preset_data["M"],
        a_terms=preset_data["a_terms"],
        b_terms=preset_data["b_terms"]
    )

    # 2. Inject multi-qubit test error on selected preset
    num_qubits_selected = H_bb_selected.shape[1]
    np.random.seed(42)

    # Pick 2 distinct physical qubit indices
    error_indices = [3, 15]
    test_error_bb = np.zeros(num_qubits_selected, dtype=int)
    test_error_bb[error_indices] = 1

    # Generate syndrome vector
    test_syndrome_bb = (H_bb_selected @ test_error_bb) % 2

    # 3. Execute SysMatch
    est_error_bb, sys_success_bb = sysmatch_bb_decoder(H_bb_selected, test_syndrome_bb)

    # Render results reactively
    mo.vstack([
        mo.md(f"### 🎯 SysMatch Decoder — **{selected_preset_key}**"),
        mo.md(f"**Code Dimensions:** $N = {num_qubits_selected}$ Data Qubits | $M = {H_bb_selected.shape[0]}$ Checks"),
        mo.md(f"**Decoder Status:** {'✅ Converged (Syndrome Cleared)' if sys_success_bb else '❌ Unmatched Syndrome'}"),
        mo.md(f"**Injected Error Indices:** `{error_indices}`"),
        mo.md(f"**Active Syndrome Weight:** `{np.sum(test_syndrome_bb)}` active checks"),
        mo.md(f"**Reconstructed Error Non-Zero Indices:** `{np.where(est_error_bb == 1)[0].tolist()}`"),
        mo.md(f"**Syndrome Residual Match:** `{sys_success_bb}`")
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Greedy Peeling Decoder
    """)
    return


@app.cell
def _(np, sp):
    def pakhunov_greedy_peeling_decoder(H: sp.csr_matrix, syndrome: np.ndarray, max_passes: int = 20):
        """
        Primary Pass: Fast Fault-Signature Greedy Peeling.
        O(1) to O(N) lookup pass that resolves sparse physical errors instantly.
        """
        H_dense = H.toarray().astype(int) % 2 if sp.issparse(H) else H.astype(int) % 2
        curr_syndrome = syndrome.copy().astype(int) % 2
        num_checks, num_bits = H_dense.shape
        estimated_error = np.zeros(num_bits, dtype=int)

        if not np.any(curr_syndrome):
            return estimated_error, True

        signatures = [H_dense[:, j] for j in range(num_bits)]
        sig_weights = np.array([np.sum(sig) for sig in signatures])

        for _ in range(max_passes):
            if not np.any(curr_syndrome):
                break

            peeled_in_pass = False
            candidates = [
                j for j in range(num_bits)
                if sig_weights[j] > 0 and np.all((curr_syndrome & signatures[j]) == signatures[j])
            ]

            if not candidates:
                break

            # Prioritize single-qubit fault signatures with larger overlap
            candidates.sort(key=lambda j: sig_weights[j], reverse=True)

            for candidate_j in candidates:
                sig = signatures[candidate_j]
                if np.all((curr_syndrome & sig) == sig):
                    estimated_error[candidate_j] ^= 1
                    curr_syndrome = (curr_syndrome ^ sig)
                    peeled_in_pass = True

                    if not np.any(curr_syndrome):
                        break

            if not peeled_in_pass:
                break

        success = not np.any(curr_syndrome)
        return estimated_error, success




    return (pakhunov_greedy_peeling_decoder,)


@app.cell
def _(
    BB_PRESETS,
    code_dropdown,
    make_bb_code,
    mo,
    np,
    pakhunov_greedy_peeling_decoder,
):
    # --- Reactive Pipeline linked to code_dropdown ---
    selected_preset_pak = code_dropdown.value
    preset_data_pak = BB_PRESETS[selected_preset_pak]

    # 1. Dynamically construct selected BB code matrix
    H_bb_pak = make_bb_code(
        L=preset_data_pak["L"],
        M=preset_data_pak["M"],
        a_terms=preset_data_pak["a_terms"],
        b_terms=preset_data_pak["b_terms"]
    )

    num_qubits_pak = H_bb_pak.shape[1]
    np.random.seed(42)

    # 2. Inject a 2-qubit physical error (typical sparse fault pair)
    pak_error_indices = [7, 43]
    test_error_pak = np.zeros(num_qubits_pak, dtype=int)
    test_error_pak[pak_error_indices] = 1

    # Generate syndrome
    test_syndrome_pak = (H_bb_pak @ test_error_pak) % 2

    # 3. Execute Pakhunov's Signature Greedy Peeling
    est_error_pak, pak_success = pakhunov_greedy_peeling_decoder(H_bb_pak, test_syndrome_pak)

    # Render results reactively
    mo.vstack([
        mo.md(f"### ⚡ Pakhunov's Deferred Greedy Peeling — **{selected_preset_pak}**"),
        mo.md(f"**Code Dimensions:** $N = {num_qubits_pak}$ Data Qubits | $M = {H_bb_pak.shape[0]}$ Checks"),
        mo.md(f"**Decoder Status:** {'✅ Converged (Syndrome Cleared)' if pak_success else '❌ Collision / Unresolved Residual'}"),
        mo.md(f"**Injected Error Indices:** `{pak_error_indices}`"),
        mo.md(f"**Active Syndrome Weight:** `{np.sum(test_syndrome_pak)}` active checks"),
        mo.md(f"**Reconstructed Error Non-Zero Indices:** `{np.where(est_error_pak == 1)[0].tolist()}`"),
        mo.md(f"**Exact Match to Injected Error:** `{np.array_equal(est_error_pak, test_error_pak)}`")
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Decoder Switch
    """)
    return


@app.cell
def _(greedy_peeling_decoder, np, sp, sysmatch_bb_decoder):
    def toshio_decoder_switch(H: sp.csr_matrix, syndrome: np.ndarray):
        """
        Toshio Decoder Switch Architecture:
        1. Runs fast Greedy Peeling first.
        2. Switches dynamically to SysMatch if Peeling gets trapped.
        """
        # --- Primary Pass: Greedy Peeling ---
        est_error, success = greedy_peeling_decoder(H, syndrome)
        if success:
            return {
                "estimated_error": est_error,
                "success": True,
                "decoder_used": "Greedy Peeling (Primary)",
                "switched": False
            }

        # --- Secondary Pass: Toshio Switch Triggered ---
        est_error_sys, success_sys = sysmatch_bb_decoder(H, syndrome)
        return {
            "estimated_error": est_error_sys,
            "success": success_sys,
            "decoder_used": "SysMatch (Secondary Switch)",
            "switched": True
        }

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Complex Error
    """)
    return


@app.cell
def _(np, sp):

    # ==========================================
    # 1. CLEAN NOISE INJECTION MODULE
    # ==========================================

    def inject_complex_error(
        H: sp.csr_matrix,
        noise_model: str = "depolarizing",
        error_rate: float = 0.02,
        measurement_error_rate: float = 0.0,
        seed: int = 42
    ):
        """
        Generates clean physical errors and valid syndrome vectors.
        Guarantees clean 1D int array casting to prevent bitwise_xor type errors.
        """
        np.random.seed(seed)
        num_checks, num_bits = H.shape
        physical_error = np.zeros(num_bits, dtype=int)
        flag_triggered = False

        # --- Channel 1: Depolarizing Noise (Default X-errors) ---
        if noise_model == "depolarizing":
            prob_x = (2.0 / 3.0) * error_rate
            physical_error = (np.random.rand(num_bits) < prob_x).astype(int)

        # --- Channel 2: Independent Bit-Flip (BSC) ---
        elif noise_model == "i_i_d_bitflip":
            physical_error = (np.random.rand(num_bits) < error_rate).astype(int)

        # --- Channel 3: Spatially Clustered Burst Noise ---
        elif noise_model == "burst_cluster":
            if np.random.rand() < error_rate * 5:
                cluster_center = np.random.randint(0, num_bits)
                cluster_size = np.random.randint(2, max(3, num_bits // 10))
                for offset in range(cluster_size):
                    idx = (cluster_center + offset) % num_bits
                    physical_error[idx] ^= 1
                flag_triggered = True

        # --- Channel 4: Phenomenological Noise ---
        elif noise_model == "phenomenological":
            physical_error = (np.random.rand(num_bits) < error_rate).astype(int)

        # Type-Safe Clean Syndrome Calculation
        raw_syndrome = H @ physical_error
        clean_syndrome = np.asarray(raw_syndrome).flatten().astype(int) % 2

        # --- Add Syndrome Measurement Noise ---
        noisy_syndrome = clean_syndrome.copy()
        if measurement_error_rate > 0.0 or noise_model == "phenomenological":
            m_noise_rate = measurement_error_rate if measurement_error_rate > 0 else error_rate
            measurement_flips = (np.random.rand(num_checks) < m_noise_rate).astype(int)
        
            measurement_flips = np.asarray(measurement_flips).flatten().astype(int) % 2
            noisy_syndrome = clean_syndrome ^ measurement_flips
        
            if np.any(measurement_flips):
                flag_triggered = True

        return {
            "physical_error": physical_error,
            "clean_syndrome": clean_syndrome,
            "noisy_syndrome": noisy_syndrome,
            "flag_triggered": flag_triggered,
            "error_weight": int(np.sum(physical_error)),
            "syndrome_weight": int(np.sum(noisy_syndrome))
        }


    return (inject_complex_error,)


@app.cell
def _(mo):

    # ==========================================
    # 2. MARIMO UI CONTROLS FOR NOISE
    # ==========================================

    noise_model_dropdown = mo.ui.dropdown(
        options={
            "Depolarizing Channel (X-errors)": "depolarizing",
            "i.i.d. Bit-Flip Channel": "i_i_d_bitflip",
            "Spatial Burst Cluster Noise": "burst_cluster",
            "Phenomenological (Data + Meas)": "phenomenological"
        },
        value="Depolarizing Channel (X-errors)",
        label="Noise Model:"
    )

    error_rate_slider = mo.ui.slider(
        start=0.001,
        stop=0.15,
        step=0.005,
        value=0.02,
        label="Physical Error Rate (p):"
    )

    meas_error_slider = mo.ui.slider(
        start=0.0,
        stop=0.10,
        step=0.01,
        value=0.0,
        label="Measurement Flip Rate (q):"
    )

    seed_number_input = mo.ui.number(
        start=1,
        stop=9999,
        value=42,
        label="Random Seed:"
    )

    mo.hstack([
        noise_model_dropdown,
        error_rate_slider,
        meas_error_slider,
        seed_number_input
    ], justify="start", gap=1)
    return (
        error_rate_slider,
        meas_error_slider,
        noise_model_dropdown,
        seed_number_input,
    )


@app.cell
def _(
    BB_PRESETS,
    H_gb,
    code_dropdown,
    error_rate_slider,
    inject_complex_error,
    make_bb_code,
    meas_error_slider,
    mo,
    noise_model_dropdown,
    np,
    seed_number_input,
    toshio_fts_decoder_switch,
):
    # Extract selected code matrix dynamically
    selected_code_key_sw = code_dropdown.value

    if "BB_PRESETS" in globals() and selected_code_key_sw in BB_PRESETS:
        p_sw = BB_PRESETS[selected_code_key_sw]
        H_active_sw = make_bb_code(
            L=p_sw["L"],
            M=p_sw["M"],
            a_terms=p_sw["a_terms"],
            b_terms=p_sw["b_terms"]
        )
    elif "H_gb" in globals():
        H_active_sw = H_gb
    else:
        H_active_sw = make_bb_code(L=12, M=6, a_terms=[(3,0),(0,1),(0,2)], b_terms=[(0,3),(1,0),(2,0)])

    num_checks_sw, num_qubits_sw = H_active_sw.shape

    # 1. Inject Complex Error Pattern (Defaults to Depolarizing Noise)
    noise_result_sw = inject_complex_error(
        H=H_active_sw,
        noise_model=noise_model_dropdown.value if "noise_model_dropdown" in globals() else "depolarizing",
        error_rate=error_rate_slider.value if "error_rate_slider" in globals() else 0.02,
        measurement_error_rate=meas_error_slider.value if "meas_error_slider" in globals() else 0.0,
        seed=int(seed_number_input.value) if "seed_number_input" in globals() else 42
    )

    # Ensure 1D binary numpy arrays to prevent ufunc bitwise_xor type errors
    test_error_sw = np.asarray(noise_result_sw["physical_error"]).flatten().astype(int) % 2
    syndrome_sw = np.asarray(noise_result_sw["noisy_syndrome"]).flatten().astype(int) % 2
    flag_signal_sw = bool(noise_result_sw.get("flag_triggered", False))

    # 2. Execute Flag-Triggered Switch (FTS)
    res_sw = toshio_fts_decoder_switch(
        H=H_active_sw, 
        syndrome=syndrome_sw, 
        flag_triggered=flag_signal_sw
    )

    # 3. Verify Residual & Analyze Solution Weight (Safely Cast Types)
    reconstructed_error_sw = np.asarray(res_sw["estimated_error"]).flatten().astype(int) % 2

    syn_arr = syndrome_sw.copy()
    est_syn_arr = np.asarray(H_active_sw @ reconstructed_error_sw).flatten().astype(int) % 2

    residual_syndrome_sw = syn_arr ^ est_syn_arr
    syndrome_cleared_sw = not np.any(residual_syndrome_sw)

    # 4. Logical Error Verification (e_true ^ e_est)
    diff_error_sw = (test_error_sw ^ reconstructed_error_sw) % 2
    diff_syndrome = np.asarray(H_active_sw @ diff_error_sw).flatten().astype(int) % 2

    # Checks if difference operator is a stabilizer/kernel (0 syndrome) or logical error
    is_in_kernel = not np.any(diff_syndrome)
    reconstructed_weight = int(np.sum(reconstructed_error_sw))
    injected_weight = int(noise_result_sw['error_weight'])

    # Solution Quality Indicator
    if not syndrome_cleared_sw:
        quality_status = "❌ Decoder Failed to Clear Syndrome"
    elif np.array_equal(test_error_sw, reconstructed_error_sw):
        quality_status = "✅ Exact Physical Vector Recovery"
    elif is_in_kernel and reconstructed_weight <= injected_weight + 2:
        quality_status = "✅ Equivalent Low-Weight Stabilizer Correction"
    else:
        quality_status = "⚠️ High-Weight Degenerate Operator (Potential Logical Fault)"

    # 5. Render Reactive Diagnostics
    mo.vstack([
        mo.md(f"### 🔀 Flag-Triggered Switch (FTS) — **{selected_code_key_sw}**"),
        mo.md(f"**Injected Noise Model:** `{noise_model_dropdown.value if 'noise_model_dropdown' in globals() else 'depolarizing'}`"),
        mo.md(f"**Hardware Flag Triggered?:** `{'🚩 Yes (Measurement/Trigger Event)' if flag_signal_sw else '⚪ No (Standard Syndrome)'}`"),
        mo.md(f"**Injected Error Weight:** `{injected_weight}` qubits"),
        mo.md(f"**Reconstructed Error Weight:** `{reconstructed_weight}` qubits"),
        mo.md("---"),
        mo.md(f"**Execution Path:** `{res_sw['decoder_used']}`"),
        mo.md(f"**Switch Triggered?:** `{'Yes 🔀' if res_sw['switched'] else 'No ⚡ (Resolved in Primary Pass)'}`"),
        mo.md(f"**Syndrome Cleared:** `{'✅ True' if syndrome_cleared_sw else '❌ False'}`"),
        mo.md(f"**Correction Fidelity:** `{quality_status}`"),
        mo.md(f"**Injected Error Indices:** `{np.where(test_error_sw == 1)[0].tolist()}`"),
        mo.md(f"**Reconstructed Error Indices:** `{np.where(reconstructed_error_sw == 1)[0].tolist()}`")
    ])
    return H_active_sw, selected_code_key_sw


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


@app.cell
def _(np, sp):

    def greedy_peeling_decoder(H: sp.csr_matrix, syndrome: np.ndarray, max_passes: int = 10):
        """
        Standard Greedy Peeling Decoder.
        Returns: (estimated_error, residual_syndrome, success)
        """
        H_dense = H.toarray().astype(int) % 2 if sp.issparse(H) else H.astype(int) % 2
        curr_syndrome = np.asarray(syndrome).flatten().astype(int) % 2
        num_checks, num_bits = H_dense.shape
    
        estimated_error = np.zeros(num_bits, dtype=int)
        signatures = [H_dense[:, j] for j in range(num_bits)]
        sig_weights = np.array([np.sum(sig) for sig in signatures])

        for _ in range(max_passes):
            if not np.any(curr_syndrome):
                break

            peeled = False
            for j in range(num_bits):
                sig = signatures[j]
                if sig_weights[j] > 0 and np.all((curr_syndrome & sig) == sig):
                    estimated_error[j] ^= 1
                    curr_syndrome ^= sig
                    peeled = True
                    if not np.any(curr_syndrome):
                        break

            if not peeled:
                break

        success = not np.any(curr_syndrome)
        return estimated_error, curr_syndrome, success

    return (greedy_peeling_decoder,)


@app.cell
def _(np, sp):
    def sysmatch_exact_decoder(H: sp.csr_matrix, syndrome: np.ndarray):
        """Standard Row Echelon GF(2) linear solver for parity check matrices."""
        H_dense = H.toarray().astype(int) % 2 if sp.issparse(H) else H.astype(int) % 2
        num_checks, num_bits = H_dense.shape
        s = np.asarray(syndrome).flatten().astype(int) % 2

        if not np.any(s):
            return np.zeros(num_bits, dtype=int), True

        # Standard column ordering prioritized by check connectivity
        check_degrees = np.sum(H_dense[s == 1, :], axis=0) if np.any(s) else np.zeros(num_bits)
        col_order = np.argsort(-check_degrees)
    
        aug = np.hstack([H_dense[:, col_order], s.reshape(-1, 1)]).astype(int) % 2
        rows, cols = aug.shape

        pivot_row = 0
        pivot_cols = []

        for col in range(num_bits):
            swap_row = -1
            for r in range(pivot_row, rows):
                if aug[r, col] == 1:
                    swap_row = r
                    break

            if swap_row == -1:
                continue

            aug[[pivot_row, swap_row]] = aug[[swap_row, pivot_row]]

            for r in range(rows):
                if r != pivot_row and aug[r, col] == 1:
                    aug[r] ^= aug[pivot_row]

            pivot_cols.append(col)
            pivot_row += 1
            if pivot_row >= rows:
                break

        # Inconsistency check
        for r in range(pivot_row, rows):
            if aug[r, -1] == 1:
                return np.zeros(num_bits, dtype=int), False

        e_ordered = np.zeros(num_bits, dtype=int)
        for i, col in enumerate(pivot_cols):
            e_ordered[col] = aug[i, -1]

        e_final = np.zeros(num_bits, dtype=int)
        e_final[col_order] = e_ordered

        residual = (H_dense @ e_final) % 2
        return e_final, np.array_equal(residual, s)

    return (sysmatch_exact_decoder,)


@app.cell
def _(greedy_peeling_decoder, np, sp, sysmatch_exact_decoder):
    def toshio_fts_decoder_switch(H: sp.csr_matrix, syndrome: np.ndarray, flag_triggered: bool = False):
        """
        Standard Flag-Triggered Switch (FTS).
        """
        syn_arr = np.asarray(syndrome).flatten().astype(int) % 2

        # 1. Hardware Flag Direct Pass
        if flag_triggered:
            e_sys, success = sysmatch_exact_decoder(H, syn_arr)
            return {
                "estimated_error": e_sys,
                "success": success,
                "decoder_used": "SysMatch (Hardware Flag Direct Pass)",
                "switched": True
            }

        # 2. Primary Pass: Greedy Peeling
        e_peel, residual_syn, success_peel = greedy_peeling_decoder(H, syn_arr)
    
        if success_peel:
            return {
                "estimated_error": e_peel,
                "success": True,
                "decoder_used": "Greedy Peeling (Primary Pass)",
                "switched": False
            }

        # 3. Trapped Fallback: SysMatch on Residual Syndrome
        e_residual, success_sys = sysmatch_exact_decoder(H, residual_syn)
        final_error = (e_peel ^ e_residual) % 2

        return {
            "estimated_error": final_error,
            "success": success_sys,
            "decoder_used": "SysMatch (Peeling Trapped Fallback)",
            "switched": True
        }

    return (toshio_fts_decoder_switch,)


@app.cell
def _(
    greedy_peeling_decoder,
    np,
    sp,
    sysmatch_exact_decoder,
    time,
    toshio_fts_decoder_switch,
):

    # ==========================================
    # 1. MONTE CARLO SIMULATION BENCHMARK
    # ==========================================

    def run_fts_benchmark(
        H: sp.csr_matrix,
        p_range: np.ndarray = np.linspace(0.001, 0.05, 7),
        trials_per_p: int = 10000,
        seed: int = 42
    ):
        """
        Monte Carlo benchmarking suite for QEC decoders.
        Evaluates LER, syndrome clearance, and microsecond timing profiles.
        """
        np.random.seed(seed)
        H_dense = H.toarray().astype(int) % 2 if sp.issparse(H) else H.astype(int) % 2
        num_checks, num_bits = H_dense.shape

        results = {
            "p_vals": p_range,
            "peeling": {"ler": [], "time_us": [], "cleared": []},
            "sysmatch": {"ler": [], "time_us": [], "cleared": []},
            "fts": {"ler": [], "time_us": [], "cleared": [], "switch_rate": []}
        }

        for p in p_range:
            # Metrics collectors per physical error rate
            peel_errs, peel_times, peel_cleared = 0, [], 0
            sys_errs, sys_times, sys_cleared = 0, [], 0
            fts_errs, fts_times, fts_cleared, fts_switches = 0, [], 0, 0

            for _ in range(trials_per_p):
                # 1. Inject Depolarizing Channel Physical Errors
                prob_x = (2.0 / 3.0) * p
                e_true = (np.random.rand(num_bits) < prob_x).astype(int)
                syndrome = np.asarray(H_dense @ e_true).flatten().astype(int) % 2

                # Simulate Hardware Flag (fires if 2+ local checks trigger simultaneously)
                flag_triggered = bool(np.sum(syndrome) >= 2 and np.random.rand() < 0.15)

                # --- ROUTE A: Pure Greedy Peeling ---
                t0 = time.perf_counter_ns()
                e_peel, syn_peel_rem, success_peel = greedy_peeling_decoder(H_dense, syndrome)
                t1 = time.perf_counter_ns()
                peel_times.append((t1 - t0) / 1000.0)

                # Check correctness (e_diff in kernel & commute check)
                diff_peel = (e_true ^ e_peel) % 2
                syn_diff_peel = np.asarray(H_dense @ diff_peel).flatten().astype(int) % 2
                is_peel_cleared = not np.any(syn_peel_rem)
            
                if is_peel_cleared:
                    peel_cleared += 1
                    # If cleared but e_diff != 0 and e_diff has weight >= d_eff, flag logical error
                    if np.any(diff_peel) and np.sum(diff_peel) >= 3:
                        peel_errs += 1
                else:
                    peel_errs += 1

                # --- ROUTE B: Pure SysMatch ---
                t0 = time.perf_counter_ns()
                e_sys, success_sys = sysmatch_exact_decoder(H_dense, syndrome)
                t1 = time.perf_counter_ns()
                sys_times.append((t1 - t0) / 1000.0)

                diff_sys = (e_true ^ e_sys) % 2
                syn_diff_sys = np.asarray(H_dense @ diff_sys).flatten().astype(int) % 2
                is_sys_cleared = not np.any(syn_diff_sys)

                if is_sys_cleared:
                    sys_cleared += 1
                    if np.any(diff_sys) and np.sum(diff_sys) >= 3:
                        sys_errs += 1
                else:
                    sys_errs += 1

                # --- ROUTE C: Flag-Triggered Switch (FTS) ---
                t0 = time.perf_counter_ns()
                res_fts = toshio_fts_decoder_switch(H_dense, syndrome, flag_triggered=flag_triggered)
                t1 = time.perf_counter_ns()
                fts_times.append((t1 - t0) / 1000.0)

                e_fts = res_fts["estimated_error"]
                if res_fts["switched"]:
                    fts_switches += 1

                diff_fts = (e_true ^ e_fts) % 2
                syn_diff_fts = np.asarray(H_dense @ diff_fts).flatten().astype(int) % 2
                is_fts_cleared = not np.any(syn_diff_fts)

                if is_fts_cleared:
                    fts_cleared += 1
                    if np.any(diff_fts) and np.sum(diff_fts) >= 3:
                        fts_errs += 1
                else:
                    fts_errs += 1

            # Store mean stats for rate p
            results["peeling"]["ler"].append(peel_errs / trials_per_p)
            results["peeling"]["time_us"].append(np.mean(peel_times))
            results["peeling"]["cleared"].append(peel_cleared / trials_per_p)

            results["sysmatch"]["ler"].append(sys_errs / trials_per_p)
            results["sysmatch"]["time_us"].append(np.mean(sys_times))
            results["sysmatch"]["cleared"].append(sys_cleared / trials_per_p)

            results["fts"]["ler"].append(fts_errs / trials_per_p)
            results["fts"]["time_us"].append(np.mean(fts_times))
            results["fts"]["cleared"].append(fts_cleared / trials_per_p)
            results["fts"]["switch_rate"].append(fts_switches / trials_per_p)

        return results



    return (run_fts_benchmark,)


@app.cell
def _(
    H_active_sw,
    mo,
    np,
    run_fts_benchmark,
    seed_number_input,
    selected_code_key_sw,
):
    # Execute Monte Carlo Run over active code
    bench_data = run_fts_benchmark(
        H=H_active_sw,
        p_range=np.linspace(0.001, 0.05, 6),
        trials_per_p=1000,  # Increase to 10000 for thesis publication plots
        seed=int(seed_number_input.value) if "seed_number_input" in globals() else 42
    )

    # Benchmark Summary Metrics
    p_mid_idx = len(bench_data["p_vals"]) // 2
    p_mid = bench_data["p_vals"][p_mid_idx]

    avg_time_peel = bench_data["peeling"]["time_us"][p_mid_idx]
    avg_time_sys = bench_data["sysmatch"]["time_us"][p_mid_idx]
    avg_time_fts = bench_data["fts"]["time_us"][p_mid_idx]
    time_reduction = ((avg_time_sys - avg_time_fts) / avg_time_sys) * 100.0 if avg_time_sys > 0 else 0.0

    # Render Reactive Benchmark Diagnostics
    mo.vstack([
        mo.md(f"### 📊 Monte Carlo Benchmark Suite — **{selected_code_key_sw}**"),
        mo.md(f"**Physical Error Rate Range:** `p = {bench_data['p_vals'][0]:.4f}` to `p = {bench_data['p_vals'][-1]:.4f}`"),
        mo.md("---"),
        mo.md(f"**Execution Latency at p = {p_mid:.4f}:**"),
        mo.md(f"- **Pure Greedy Peeling:** `{avg_time_peel:.2f} µs`"),
        mo.md(f"- **Pure SysMatch:** `{avg_time_sys:.2f} µs`"),
        mo.md(f"- **Flag-Triggered Switch (FTS):** `{avg_time_fts:.2f} µs` (**{time_reduction:.1f}% fast-path latency reduction** vs. Pure SysMatch)"),
        mo.md("---"),
        mo.md("#### **Logical Error Rate (LER) & Switch Distribution**"),
        mo.md(f"""
    | $p_{{\\text{{phys}}}}$ | Pure Peeling LER | Pure SysMatch LER | FTS Switch LER | FTS Trigger Rate |
    | :--- | :--- | :--- | :--- | :--- |
    """ + "\n".join([
            f"| `{bench_data['p_vals'][i]:.4f}` | `{bench_data['peeling']['ler'][i]:.4f}` | `{bench_data['sysmatch']['ler'][i]:.4f}` | `{bench_data['fts']['ler'][i]:.4f}` | `{bench_data['fts']['switch_rate'][i]*100:.1f}%` |"
            for i in range(len(bench_data['p_vals']))
        ]))
    ])
    return


if __name__ == "__main__":
    app.run()
