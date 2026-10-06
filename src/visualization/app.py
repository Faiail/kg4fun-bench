import random
import streamlit as st
import torch
from pathlib import Path

from src.visualization.runner import (
    infer_run_class_name,
    get_config_path,
    load_and_init_run,
    get_schema_embeddings,
    find_sample_index,
    predict_sample,
)

TASKS = ["node", "edge", "complete"]
DATASETS = ["dataset1", "dataset2"]
SPLIT_MODES = ["hybrid", "inductive"]
ARCHITECTURES = ["shared", "dedicated"]
MODELS = ["bert", "bge", "roberta", "modernbert"]
DATA_SPLITS = ["val", "test"]  # Only val and test are usable


def inject_custom_styles():
    st.markdown(
        """
        <style>
        .result-box-match {
            background-color: rgba(16, 185, 129, 0.1);
            border: 1px solid #10b981;
            border-radius: 10px;
            padding: 16px 20px;
            margin-bottom: 15px;
        }
        .result-box-mismatch {
            background-color: rgba(244, 63, 94, 0.1);
            border: 1px solid #f43f5e;
            border-radius: 10px;
            padding: 16px 20px;
            margin-bottom: 15px;
        }
        .card-header-gt {
            color: #60a5fa;
            font-size: 0.9rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 6px;
        }
        .card-header-pred {
            color: #c084fc;
            font-size: 0.9rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 6px;
        }
        .card-title {
            font-size: 1.15rem;
            font-weight: 600;
            color: #f8fafc;
            margin-bottom: 8px;
        }
        .card-desc {
            color: #cbd5e1;
            font-size: 0.95rem;
            line-height: 1.4;
        }
        .status-badge-match {
            display: inline-block;
            background-color: #065f46;
            color: #34d399;
            font-weight: 700;
            padding: 4px 14px;
            border-radius: 9999px;
            border: 1px solid #059669;
            font-size: 0.9rem;
        }
        .status-badge-mismatch {
            display: inline-block;
            background-color: #881337;
            color: #fb7185;
            font-weight: 700;
            padding: 4px 14px;
            border-radius: 9999px;
            border: 1px solid #e11d48;
            font-size: 0.9rem;
        }
        .input-chip {
            background-color: #1e293b;
            border: 1px solid #334155;
            border-radius: 8px;
            padding: 12px 16px;
            margin-bottom: 12px;
            font-size: 0.95rem;
            color: #e2e8f0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(
        page_title="KG4FUN Predictions Visualizer",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_custom_styles()

    st.title("🎯 KG4FUN Neural Predictions Visualizer")
    st.caption("Live forward-pass prediction and string inspection powered directly by the inferred experiment run class.")

    # ==========================================
    # 1. Sidebar: Experiment & Run Inferences
    # ==========================================
    with st.sidebar:
        st.header("⚙️ Experiment Selection")

        task = st.selectbox(
            "Task / Problem Type",
            options=TASKS,
            index=0,
            format_func=lambda x: f"📌 {x.upper()} Classification",
        )

        col_d1, col_d2 = st.columns(2)
        with col_d1:
            dataset_name = st.selectbox("Dataset", options=DATASETS, index=0)
        with col_d2:
            split_mode = st.selectbox("Split Mode", options=SPLIT_MODES, index=0)

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            architecture = st.selectbox("Architecture", options=ARCHITECTURES, index=0)
        with col_m2:
            model_name = st.selectbox("Backbone Model", options=MODELS, index=0)

        st.markdown("---")
        st.header("📂 Split & Hardware")

        # Requirement 2: The only data splits usable are val/test
        split = st.selectbox(
            "Data Split",
            options=DATA_SPLITS,
            index=0,
            help="Usable data splits: val and test.",
        )

        has_cuda = torch.cuda.is_available()
        device = st.radio(
            "Compute Device",
            options=["cuda", "cpu"] if has_cuda else ["cpu"],
            index=0,
            horizontal=True,
        )

        # Requirement 1: Infer the right run class to load + init
        run_class_name = infer_run_class_name(task=task, architecture=architecture)
        config_path = get_config_path(
            task=task,
            dataset=dataset_name,
            split_mode=split_mode,
            architecture=architecture,
            model=model_name,
        )

        st.markdown("---")
        st.subheader("Inferred Run Details")
        st.markdown(f"**🏃 Run Class:** `{run_class_name}`")
        if config_path.exists():
            st.success(f"⚙️ Config: `{config_path.name}`")
        else:
            st.error(f"❌ Config missing:\n`{config_path}`")
            return

        # Explicit Cache Reset Button
        if st.button("🔄 Reload Model & Reset Cache", use_container_width=True):
            st.cache_resource.clear()
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

    # Track active run setting to reset sample state when switching runs
    current_run_key = f"{task}_{dataset_name}_{split_mode}_{architecture}_{model_name}_{split}"
    if st.session_state.get("active_run_key") != current_run_key:
        st.session_state.active_run_key = current_run_key
        st.session_state.current_sample_idx = 0
        st.session_state.sample_id_input = ""

    # ==========================================
    # 2. Run Class Loading & Init (Requirements 1 & 3)
    # ==========================================
    config_mtime = config_path.stat().st_mtime if config_path.exists() else 0.0

    try:
        run = load_and_init_run(
            config_path_str=str(config_path),
            run_class_name=run_class_name,
            device=device,
            config_mtime=config_mtime,
        )
    except Exception as e:
        st.error(f"Error initializing run class `{run_class_name}`: {e}")
        return

    # Status of state dict from run.early_stop.path
    if run.checkpoint_loaded_from:
        st.sidebar.success(f"🧠 State dict loaded from:\n`{Path(run.checkpoint_loaded_from).name}`")
    else:
        st.sidebar.warning(f"⚠️ Checkpoint not found at `run.early_stop.path`:\n`{getattr(run.early_stop, 'path', '')}`")

    # ==========================================
    # 3. Data Split Loader & Dataset (Requirement 2)
    # ==========================================
    loader = run.val_loader if split == "val" else run.test_loader
    if loader is None or loader.dataset is None:
        st.error(f"Loader or dataset for split `{split}` is not available in the initialized run.")
        return

    dataset = loader.dataset
    total_samples = len(dataset)

    # Ensure current sample index is within valid bounds for this dataset
    if "current_sample_idx" not in st.session_state:
        st.session_state.current_sample_idx = 0
    else:
        st.session_state.current_sample_idx = max(0, min(st.session_state.current_sample_idx, total_samples - 1))

    # ==========================================
    # 4. Gather Schema Item Embeddings (Requirement 4)
    # ==========================================
    try:
        # Schema embeddings are cached per run & split on run._schema_embeddings_cache
        schema_embeddings = get_schema_embeddings(run, dataset, task=task, split=split)
    except Exception as e:
        st.error(f"Failed to gather schema item embeddings via runner: {e}")
        return

    # ==========================================
    # 5. Sample Selection & Navigation
    # ==========================================
    st.markdown("### 🔍 Select Sample to Analyze")

    col_id_search, col_idx_nav = st.columns([1.5, 2.5])

    with col_id_search:
        sample_id_input = st.text_input(
            "Direct Sample ID / QID Lookup",
            value=st.session_state.get("sample_id_input", ""),
            placeholder="e.g. Q1060131 or sample index...",
            help="Type an entity QID or numeric sample index to jump directly.",
            key="sample_id_input_widget",
        ).strip()

        if sample_id_input and sample_id_input != st.session_state.get("sample_id_input", ""):
            st.session_state.sample_id_input = sample_id_input
            found_idx = find_sample_index(dataset, sample_id_input, task=task)
            if found_idx is not None:
                st.session_state.current_sample_idx = found_idx
                st.success(f"Found sample at index `#{found_idx}`")
            else:
                st.warning(f"Identifier `{sample_id_input}` not found in `{split}` split.")

    with col_idx_nav:
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
        with col_btn1:
            if st.button("⬅️ Previous", use_container_width=True):
                st.session_state.current_sample_idx = (st.session_state.current_sample_idx - 1) % total_samples
        with col_btn2:
            if st.button("Next ➡️", use_container_width=True):
                st.session_state.current_sample_idx = (st.session_state.current_sample_idx + 1) % total_samples
        with col_btn3:
            if st.button("🎲 Random", use_container_width=True):
                st.session_state.current_sample_idx = random.randint(0, total_samples - 1)

        st.session_state.current_sample_idx = st.slider(
            f"Sample Index in `{split}` ({total_samples:,} total samples)",
            min_value=0,
            max_value=max(0, total_samples - 1),
            value=st.session_state.current_sample_idx,
        )

    current_idx = st.session_state.current_sample_idx

    # ==========================================
    # 6. Predict & Gather Strings (Requirements 4 & 5)
    # ==========================================
    with st.spinner("Running forward pass and scoring..."):
        result = predict_sample(
            run=run,
            dataset=dataset,
            loader=loader,
            schema_embeddings=schema_embeddings,
            sample_idx=current_idx,
            task=task,
            top_k=5,
        )

    st.markdown("---")

    # ==========================================
    # 7. Render Visualization Strings
    # ==========================================
    header_col1, header_col2 = st.columns([3, 1])
    with header_col1:
        st.markdown(f"### Sample `#{current_idx}`: `{result['sample_id']}`")
    with header_col2:
        if result["is_match"]:
            st.markdown("<div style='text-align: right;'><span class='status-badge-match'>🎯 MATCH (CORRECT)</span></div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='text-align: right;'><span class='status-badge-mismatch'>❌ MISMATCH (ERROR)</span></div>", unsafe_allow_html=True)

    # Input Formatted String
    st.markdown("**Input Formatted String (passed into model tokenizer):**")
    st.markdown(f"<div class='input-chip'><code>{result['input_string']}</code></div>", unsafe_allow_html=True)

    # Single-item tasks (Node and Edge)
    if task in ["node", "edge"]:
        col_gt, col_pred = st.columns(2)

        with col_gt:
            st.markdown(
                f"""
                <div class="result-box-match" style="border-color: #3b82f6; background-color: rgba(59, 130, 246, 0.08);">
                    <div class="card-header-gt">🏷️ Ground Truth Class (ID: {result['gt_kb_id']})</div>
                    <div class="card-desc" style="color: #e2e8f0; font-family: monospace; white-space: pre-wrap;">
                        {result['gt_str']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_pred:
            box_class = "result-box-match" if result["is_match"] else "result-box-mismatch"
            st.markdown(
                f"""
                <div class="{box_class}">
                    <div class="card-header-pred">🔮 Predicted Class (ID: {result['pred_kb_id']})</div>
                    <div class="card-desc" style="color: #e2e8f0; font-family: monospace; white-space: pre-wrap;">
                        {result['pred_str']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Top-K candidate scores
        if "candidates" in result:
            st.markdown("#### 📊 Top Model Predictions & Scores")
            for c in result["candidates"]:
                match_tag = " &nbsp;<span style='color: #34d399; font-weight: 700;'>🎯 (GROUND TRUTH)</span>" if c["is_match"] else ""
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**#{c['rank']}** `ID: {c['kb_id']}`{match_tag}<br/><span style='color: #cbd5e1; font-size: 0.88rem;'>{c['schema_str']}</span>", unsafe_allow_html=True)
                with col2:
                    st.markdown(f"Score: `{c['score']:.4f}` | Prob: `{c['prob'] * 100:.1f}%`")
                    st.progress(min(max(c['prob'], 0.0), 1.0))
                st.markdown("<hr style='margin: 4px 0; border-color: rgba(255,255,255,0.06);'/>", unsafe_allow_html=True)

    # Complete task
    elif task == "complete":
        cp = result["complete_preds"]
        for comp_name in ["head", "rel", "tail"]:
            c_info = cp[comp_name]
            st.markdown(f"#### 🔹 {comp_name.upper()} Prediction ({'✅ MATCH' if c_info['match'] else '❌ MISMATCH'})")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Ground Truth (ID {c_info['gt_id']}):**\n`{c_info['gt_str']}`")
            with col2:
                st.markdown(f"**Predicted (ID {c_info['pred_id']}):**\n`{c_info['pred_str']}`")
            st.markdown("---")

    # Raw Object Expander
    with st.expander("🔍 Inspect Raw Dataset Item"):
        st.json(result["raw_item"])


if __name__ == "__main__":
    main()
