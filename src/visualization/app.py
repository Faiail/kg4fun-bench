import random
import streamlit as st
from pathlib import Path

from src.visualization.config import (
    DATASETS,
    TASKS,
    SPLIT_MODES,
    ARCHITECTURES,
    MODELS,
    DATA_SPLITS,
    get_model_dir,
    find_prediction_file,
    find_metrics_file,
)
from src.visualization.data import (
    load_node_dataset,
    load_edge_dataset,
    load_predictions,
    load_metrics,
    format_schema_entry,
)
from src.visualization.components import (
    inject_custom_css,
    render_metrics_row,
    render_node_input_card,
    render_edge_input_card,
    render_prediction_vs_gt,
)


def main():
    st.set_page_config(
        page_title="KG4FUN Predictions Visualizer",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_custom_css()

    st.title("🎯 KG4FUN Predictions Visualizer")
    st.caption("Interactive analysis of model predictions against ground truth for KG node and edge classification.")

    # ==========================================
    # 1. Sidebar - Experiment & Dataset Settings
    # ==========================================
    with st.sidebar:
        st.header("⚙️ Experiment Settings")

        task = st.selectbox(
            "Task / Problem Type",
            options=TASKS,
            index=0,
            format_func=lambda x: f"📌 {x.upper()} Classification",
        )

        col_d1, col_d2 = st.columns(2)
        with col_d1:
            dataset = st.selectbox("Dataset", options=DATASETS, index=0)
        with col_d2:
            split_mode = st.selectbox("Split Mode", options=SPLIT_MODES, index=0)

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            architecture = st.selectbox("Architecture", options=ARCHITECTURES, index=0)
        with col_m2:
            model = st.selectbox("Backbone Model", options=MODELS, index=0)

        st.markdown("---")
        st.header("📂 Split Selection")
        split = st.selectbox(
            "Data Split",
            options=DATA_SPLITS,
            index=0,
            help="Choose between test, validation, or training partition.",
        )

        # Path resolution & status
        model_dir = get_model_dir(task, architecture, model, dataset, split_mode)
        pred_file = find_prediction_file(model_dir) if split == "test" else None
        metrics_file = find_metrics_file(model_dir) if split == "test" else None

        st.markdown("---")
        st.subheader("Model Run Status")
        if model_dir.exists():
            st.success(f"📁 Model folder found:\n`{model_dir.name}`")
            if pred_file:
                st.info(f"🔮 Predictions: `{pred_file.name}`")
            else:
                if split == "test":
                    st.warning("⚠️ No predictions file (`inferece.json`) found in this model folder.")
                else:
                    st.info(f"ℹ️ Currently viewing `{split}` dataset. Predictions are typically logged for `test`.")
        else:
            st.warning(f"⚠️ Model directory not found:\n`{model_dir.relative_to(model_dir.parent.parent.parent)}`")

    # ==========================================
    # 2. Data & Predictions Loading
    # ==========================================
    if task == "node":
        samples, schema_map = load_node_dataset(dataset, split_mode, split)
    elif task in ["edge", "complete"]:
        samples, schema_map = load_edge_dataset(dataset, split_mode, split)
    else:
        st.error(f"Unsupported task: {task}")
        return

    if not samples:
        st.error(f"No samples found for {dataset} / {split_mode} / {split}. Please verify the raw dataset files.")
        return

    preds_map, raw_preds = load_predictions(str(pred_file)) if pred_file else (None, None)
    metrics = load_metrics(str(metrics_file)) if metrics_file else None

    # Calculate match stats across the split if predictions are present
    correct_count = 0
    total_evaluated = 0
    for s in samples:
        s_id = s.get("qid") if task == "node" else s.get("index")
        pred_idx = preds_map.get(s_id) if preds_map else None
        s["pred_idx"] = pred_idx
        if pred_idx is not None:
            total_evaluated += 1
            if pred_idx == s["gt_idx"]:
                correct_count += 1
            s["is_correct"] = (pred_idx == s["gt_idx"])
        else:
            s["is_correct"] = None

    # ==========================================
    # 3. Overview Metrics Bar
    # ==========================================
    render_metrics_row(
        metrics=metrics,
        total_samples=len(samples),
        matched_samples=correct_count if total_evaluated > 0 else None,
    )
    st.write("")

    # ==========================================
    # 4. Interactive Filters & Search
    # ==========================================
    filter_col1, filter_col2, filter_col3 = st.columns([1.5, 2, 1.2])

    with filter_col1:
        prediction_filter = st.radio(
            "Filter Prediction Status",
            options=["All Samples", "❌ Errors Only (Mismatch)", "✅ Correct Only (Match)"],
            horizontal=True,
        )

    with filter_col2:
        search_query = st.text_input(
            "🔎 Search (by QID / Entity Label / Keyword)",
            value="",
            placeholder="e.g. Q1129466 or 'analysis'...",
        ).strip().lower()

    # Apply filtering
    filtered_samples = samples
    if prediction_filter == "❌ Errors Only (Mismatch)":
        filtered_samples = [s for s in filtered_samples if s.get("is_correct") is False]
    elif prediction_filter == "✅ Correct Only (Match)":
        filtered_samples = [s for s in filtered_samples if s.get("is_correct") is True]

    if search_query:
        filtered_samples = [
            s for s in filtered_samples
            if search_query in s.get("sample_id", "").lower()
            or search_query in s.get("label", "").lower()
            or search_query in s.get("description", "").lower()
            or search_query in s.get("full_input_text", "").lower()
        ]

    with filter_col3:
        st.write("")
        st.markdown(f"**Showing {len(filtered_samples):,} / {len(samples):,} samples**")

    if not filtered_samples:
        st.warning("No samples match the selected filter and search criteria.")
        return

    # ==========================================
    # 5. Sample Navigation
    # ==========================================
    st.markdown("---")
    nav_col1, nav_col2, nav_col3, nav_col4 = st.columns([2, 1, 1, 1])

    # Manage current index in session state
    if "sample_index" not in st.session_state:
        st.session_state.sample_index = 0

    # Ensure index within bounds
    if st.session_state.sample_index >= len(filtered_samples):
        st.session_state.sample_index = 0

    with nav_col2:
        if st.button("⬅️ Previous Sample", use_container_width=True):
            st.session_state.sample_index = (st.session_state.sample_index - 1) % len(filtered_samples)

    with nav_col3:
        if st.button("Next Sample ➡️", use_container_width=True):
            st.session_state.sample_index = (st.session_state.sample_index + 1) % len(filtered_samples)

    with nav_col4:
        if st.button("🎲 Random Sample", use_container_width=True):
            st.session_state.sample_index = random.randint(0, len(filtered_samples) - 1)

    with nav_col1:
        # Sample selection dropdown
        sample_options = [
            f"#{i} [{s.get('sample_id')}] - {s.get('label', s.get('rel_label', 'Sample'))[:30]}"
            for i, s in enumerate(filtered_samples)
        ]
        chosen_sample_label = st.selectbox(
            "Select Sample",
            options=sample_options,
            index=st.session_state.sample_index,
            label_visibility="collapsed",
        )
        st.session_state.sample_index = sample_options.index(chosen_sample_label)

    current_sample = filtered_samples[st.session_state.sample_index]

    # ==========================================
    # 6. Detailed Sample Display
    # ==========================================
    st.write("")
    if task == "node":
        render_node_input_card(current_sample)
    else:
        render_edge_input_card(current_sample)

    st.write("")

    # Format GT and Prediction schema strings
    gt_info = format_schema_entry(schema_map, current_sample["gt_idx"], task=task)
    pred_info = format_schema_entry(schema_map, current_sample.get("pred_idx"), task=task)

    render_prediction_vs_gt(gt_info, pred_info)

    # ==========================================
    # 7. Raw JSON Inspection
    # ==========================================
    st.write("")
    with st.expander("🔍 Inspect Full Input Text & Raw JSON"):
        st.markdown("**Full Input Formatted String:**")
        st.code(current_sample.get("full_input_text", ""))

        st.markdown("**Sample Record Object:**")
        st.json(current_sample)

        st.markdown("**Ground Truth Schema Object:**")
        st.json(schema_map.get(current_sample["gt_idx"], {"idx": current_sample["gt_idx"], "info": "Prune or not in map"}))

        if current_sample.get("pred_idx") is not None:
            st.markdown("**Predicted Schema Object:**")
            st.json(schema_map.get(current_sample["pred_idx"], {"idx": current_sample["pred_idx"], "info": "Prune or not in map"}))


if __name__ == "__main__":
    main()
