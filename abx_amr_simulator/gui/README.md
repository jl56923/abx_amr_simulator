# GUI Applications

This directory contains Streamlit-based GUI applications for the ABX AMR RL project.

## Available Apps

### 1. Experiment Runner (`experiment_runner.py`)
Launch and configure RL training experiments with an interactive interface.

**Features:**
- Configure environment parameters (antibiotics, AMR dynamics, crossresistance)
- Set reward calculator parameters
- Configure training hyperparameters
- Live training log streaming
- Auto-display results after completion

**Launch:**
```bash
streamlit run gui/experiment_runner.py
```
Access at: http://localhost:8501

---

### 2. Experiment Viewer (`experiment_viewer.py`)
Browse and review diagnostic plots from completed experiments.

**Features:**
- Browse all experiment runs (sorted newest first)
- Filter experiments by name
- View full configuration (organized by sections)
- Download config as YAML
- Display diagnostic plots grouped by category
- Responsive 2-column image grid

**Launch:**
```bash
streamlit run gui/experiment_viewer.py --server.port 8502
```
Access at: http://localhost:8502

---

## Running Both Apps Simultaneously

For the best workflow, run both apps in separate terminals:

```bash
# Terminal 1: Start experiment runner
streamlit run gui/experiment_runner.py

# Terminal 2: Start experiment viewer (different port)
streamlit run gui/experiment_viewer.py --server.port 8502
```

Then open two browser tabs:
- **Tab 1:** http://localhost:8501 (configure and launch experiments)
- **Tab 2:** http://localhost:8502 (review past results)

This allows you to monitor training progress in one tab while browsing previous experiments in another.

---

## Tips

- **Experiment Runner:** Stay on the page while training runs to see live progress updates
- **Experiment Viewer:** Automatically shows newest experiments first; use the name filter to find specific runs
- **Config Download:** Use the download button to save experiment configurations for reproducibility
