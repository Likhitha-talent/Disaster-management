# Mathematical Formulation: AHP & TOPSIS Decision Engines

**Course:** BIS586 | 5th Semester | Dept. of Information Science & Engineering  
**Institution:** The National Institute of Engineering, Mysuru  
**Guide:** Nandini P, Assistant Professor  

---

## 1. Multi-Hazard Composite Risk Mapping via AHP

### 1.1 Composite Risk Formula
For any spatial unit or cell $x$, the overall hazard score $R(x)$ is computed as the transparent linear weighted sum of normalized environmental and observational hazard factors:

$$R(x) = \sum_{i=1}^{M} w_i \cdot r_i(x)$$

Where:
- $r_i(x) \in [0, 1]$ is the normalized rating for factor $i$.
- $w_i \in [0, 1]$ is the weight assigned to factor $i$, satisfying $\sum_{i=1}^M w_i = 1.0$.

### 1.2 Factors Considered
1. **$r_1$: Damage Density / Structural Impact ($w_1 = 0.35$):** Extracted from Siamese U-Net building predictions.
2. **$r_2$: Flood & Hydrological Exposure ($w_2 = 0.25$):** Flow accumulation and flood inundation buffer.
3. **$r_3$: Terrain Slope & Landslide Susceptibility ($w_3 = 0.18$):** Derived from Digital Elevation Model (DEM).
4. **$r_4$: Land Cover Hazard / Soil Permeability ($w_4 = 0.12$):** ESA WorldCover classification.
5. **$r_5$: Population Exposure ($w_5 = 0.10$):** Gridded population density from WorldPop.

### 1.3 Analytic Hierarchy Process (AHP) Consistency Check
Given the $M \times M$ pairwise comparison matrix $A$:
1. Compute the normalized priority vector $\vec{w}$ by averaging normalized column sums:
   $$w_i = \frac{1}{M} \sum_{j=1}^{M} \frac{a_{ij}}{\sum_{k=1}^M a_{kj}}$$
2. Determine the maximum eigenvalue $\lambda_{\max}$:
   $$\lambda_{\max} = \frac{1}{M} \sum_{i=1}^{M} \frac{(A \vec{w})_i}{w_i}$$
3. Compute Consistency Index ($CI$):
   $$CI = \frac{\lambda_{\max} - M}{M - 1}$$
4. Calculate Consistency Ratio ($CR$):
   $$CR = \frac{CI}{RI}$$
   For $M = 5$, the Random Index $RI = 1.12$.  
   *Requirement:* $CR < 0.10$ (10%) guarantees consistent human judgments. In our baseline matrix, $CR = 0.042$ (Valid).

### 1.4 Hazard Screening Categories
- **Red Zone ($R(x) \ge 0.65$):** High hazard risk area requiring relocation prioritization.
- **Amber Zone ($0.35 \le R(x) < 0.65$):** Cautionary buffer zone requiring structural mitigation.
- **Green Zone ($R(x) < 0.35$):** Low hazard baseline safe for temporary or permanent habitation.

---

## 2. Relocation Candidate Site Screening & TOPSIS Ranking

### 2.1 Hard Exclusion Constraints
Before ranking, any candidate polygon violating any mandatory threshold is excluded:
1. **Red Zone Intersect:** If site geometry overlaps documented Red Hazard Zone $\implies$ EXCLUDED.
2. **Terrain Slope:** If $\text{Slope} > 16.0^\circ \implies$ EXCLUDED (unstable terrain).
3. **Minimum Area:** If $A_{\text{total}} < 1,500 \text{ m}^2 \implies$ EXCLUDED (insufficient for settlement layout).
4. **Hazard Proximity:** If $\text{Buffer to Red Zone} < 250 \text{ m} \implies$ EXCLUDED.

### 2.2 Preliminary Carrying Capacity Estimation
For each eligible site $k$:

$$C_k = \frac{A_k \times u_k}{a_k}$$

Where:
- $A_k$: Total site polygon area ($\text{m}^2$).
- $u_k$: Usable development fraction (Default: $0.65$, reserving $35\%$ for internal roads, sanitation, and setbacks).
- $a_k$: Standard humanitarian planning area per person (Sphere Project Standard: $30 \text{ m}^2 / \text{person}$).

### 2.3 TOPSIS Multi-Criteria Decision Algorithm
Let $m$ be the number of eligible sites and $n$ be the number of criteria:
1. **Decision Matrix:** Construct $X = [x_{ij}]_{m \times n}$.
   - $C_1$: Distance to Main Road (km) — *Cost Criterion* ($\min$)
   - $C_2$: Essential Service Proximity Score — *Beneficial* ($\max$)
   - $C_3$: Net Usable Area (ha) — *Beneficial* ($\max$)
   - $C_4$: Terrain Slope (deg) — *Cost Criterion* ($\min$)
   - $C_5$: Buffer from Hazard Zone (km) — *Beneficial* ($\max$)

2. **Vector Normalization:**
   $$r_{ij} = \frac{x_{ij}}{\sqrt{\sum_{k=1}^m x_{kj}^2}}$$

3. **Weighted Normalized Matrix:**
   $$v_{ij} = w_j \cdot r_{ij}$$

4. **Identify Ideal Best ($A^+$) and Ideal Worst ($A^-$):**
   $$A_j^+ = \begin{cases} \max_i v_{ij} & \text{if criterion } j \text{ is beneficial} \\ \min_i v_{ij} & \text{if criterion } j \text{ is cost} \end{cases}$$
   $$A_j^- = \begin{cases} \min_i v_{ij} & \text{if criterion } j \text{ is beneficial} \\ \max_i v_{ij} & \text{if criterion } j \text{ is cost} \end{cases}$$

5. **Euclidean Separation Distances:**
   $$S_i^+ = \sqrt{\sum_{j=1}^n (v_{ij} - A_j^+)^2}, \quad S_i^- = \sqrt{\sum_{j=1}^n (v_{ij} - A_j^-)^2}$$

6. **Relative Closeness Score:**
   $$C_i^* = \frac{S_i^-}{S_i^+ + S_i^-} \in [0, 1]$$

7. **Priority Classification:**
   - $C_i^* \ge 0.70$: Immediate Priority (Phase 1 Rehabilitation)
   - $0.45 \le C_i^* < 0.70$: Short-Term Priority (Phase 2)
   - $C_i^* < 0.45$: Medium-Term Contingency (Phase 3)
