import streamlit as st
from typing import Dict, Any, Optional


def inject_custom_css():
    """Injects sleek modern styling for cards, badges, and layout."""
    st.markdown(
        """
        <style>
        .metric-card {
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 18px 22px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        }
        .status-pill-match {
            display: inline-block;
            background-color: #065f46;
            color: #34d399;
            font-weight: 700;
            padding: 6px 14px;
            border-radius: 9999px;
            border: 1px solid #059669;
            font-size: 0.85rem;
            letter-spacing: 0.03em;
        }
        .status-pill-mismatch {
            display: inline-block;
            background-color: #881337;
            color: #fb7185;
            font-weight: 700;
            padding: 6px 14px;
            border-radius: 9999px;
            border: 1px solid #e11d48;
            font-size: 0.85rem;
            letter-spacing: 0.03em;
        }
        .status-pill-neutral {
            display: inline-block;
            background-color: #334155;
            color: #94a3b8;
            font-weight: 600;
            padding: 6px 14px;
            border-radius: 9999px;
            border: 1px solid #475569;
            font-size: 0.85rem;
        }
        .info-subtext {
            color: #94a3b8;
            font-size: 0.85rem;
            margin-top: 4px;
        }
        .schema-box-gt {
            background-color: rgba(59, 130, 246, 0.08);
            border: 1px solid rgba(59, 130, 246, 0.3);
            border-radius: 10px;
            padding: 18px;
            height: 100%;
        }
        .schema-box-pred {
            background-color: rgba(168, 85, 247, 0.08);
            border: 1px solid rgba(168, 85, 247, 0.3);
            border-radius: 10px;
            padding: 18px;
            height: 100%;
        }
        .entity-chip {
            background-color: #1e293b;
            border: 1px solid #334155;
            border-radius: 8px;
            padding: 10px 14px;
            margin-bottom: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_metrics_row(metrics: Optional[Dict[str, Any]], total_samples: int, matched_samples: Optional[int] = None):
    """Renders high-level metrics cards."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Split Samples", f"{total_samples:,}")

    with col2:
        if metrics and "f1" in metrics:
            val = metrics["f1"]
            st.metric("Macro F1", f"{val:.4f}" if isinstance(val, (int, float)) else str(val))
        elif matched_samples is not None and total_samples > 0:
            acc = matched_samples / total_samples
            st.metric("Test Accuracy", f"{acc * 100:.2f}%")
        else:
            st.metric("Macro F1", "N/A")

    with col3:
        if metrics and "precision" in metrics:
            val = metrics["precision"]
            st.metric("Macro Precision", f"{val:.4f}" if isinstance(val, (int, float)) else str(val))
        else:
            st.metric("Macro Precision", "N/A")

    with col4:
        if metrics and "recall" in metrics:
            val = metrics["recall"]
            st.metric("Macro Recall", f"{val:.4f}" if isinstance(val, (int, float)) else str(val))
        else:
            st.metric("Macro Recall", "N/A")


def render_node_input_card(sample: Dict[str, Any]):
    """Renders input details for a node classification sample."""
    qid = sample.get("qid", "")
    wikidata_url = f"https://www.wikidata.org/wiki/{qid}" if qid.startswith("Q") else None

    st.markdown("#### 📌 Input Entity Details")
    st.markdown(
        f"""
        <div class="entity-chip">
            <span style="font-size: 1.2rem; font-weight: 700; color: #f8fafc;">{sample.get('label', 'Unknown')}</span>
            &nbsp;&nbsp;
            <code style="font-size: 0.95rem; background-color: #0f172a; padding: 2px 8px; border-radius: 4px;">{qid}</code>
            {f'&nbsp;&nbsp;<a href="{wikidata_url}" target="_blank" style="text-decoration: none; font-size: 0.85rem; color: #60a5fa;">🔗 View Wikidata</a>' if wikidata_url else ''}
            <div style="color: #cbd5e1; margin-top: 8px; font-size: 0.95rem; line-height: 1.4;">
                {sample.get('description', 'No description available')}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_edge_input_card(sample: Dict[str, Any]):
    """Renders input details for an edge classification triple."""
    head_qid = sample.get("head_qid", "")
    tail_qid = sample.get("tail_qid", "")
    pid = sample.get("pid", "")

    st.markdown("#### 🔗 Input Graph Triple")
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.markdown(
            f"""
            <div class="entity-chip">
                <div style="font-size: 0.75rem; text-transform: uppercase; color: #38bdf8; font-weight: 700; letter-spacing: 0.05em;">Head Entity</div>
                <div style="font-weight: 700; font-size: 1.05rem; color: #f8fafc; margin-top: 2px;">{sample.get('head_label', 'Unknown')}</div>
                <code>{head_qid}</code>
                <div style="font-size: 0.85rem; color: #94a3b8; margin-top: 4px;">{sample.get('head_desc', '')}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="entity-chip">
                <div style="font-size: 0.75rem; text-transform: uppercase; color: #a78bfa; font-weight: 700; letter-spacing: 0.05em;">Relation / Predicate</div>
                <div style="font-weight: 700; font-size: 1.05rem; color: #f8fafc; margin-top: 2px;">{sample.get('rel_label', pid)}</div>
                <code>{pid}</code>
                <div style="font-size: 0.85rem; color: #94a3b8; margin-top: 4px;">{sample.get('rel_desc', '')}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class="entity-chip">
                <div style="font-size: 0.75rem; text-transform: uppercase; color: #34d399; font-weight: 700; letter-spacing: 0.05em;">Tail Entity</div>
                <div style="font-weight: 700; font-size: 1.05rem; color: #f8fafc; margin-top: 2px;">{sample.get('tail_label', 'Unknown')}</div>
                <code>{tail_qid}</code>
                <div style="font-size: 0.85rem; color: #94a3b8; margin-top: 4px;">{sample.get('tail_desc', '')}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_prediction_vs_gt(gt_info: Dict[str, Any], pred_info: Dict[str, Any]):
    """Renders side-by-side comparison cards for Ground Truth vs Predicted schema string."""
    gt_idx = gt_info.get("idx")
    pred_idx = pred_info.get("idx")

    has_prediction = pred_idx is not None
    is_match = has_prediction and (gt_idx == pred_idx)

    # Status header
    if not has_prediction:
        status_html = '<span class="status-pill-neutral">⚠️ No Prediction Found in Run</span>'
    elif is_match:
        status_html = '<span class="status-pill-match">✅ CORRECT PREDICTION</span>'
    else:
        status_html = '<span class="status-pill-mismatch">❌ MISMATCH (ERROR)</span>'

    st.markdown(f"### Comparison Status &nbsp;&nbsp; {status_html}", unsafe_allow_html=True)
    st.write("")

    col_gt, col_pred = st.columns(2)

    with col_gt:
        prune_tag = " <span style='color: #f59e0b; font-size: 0.85rem;'>(Prune / Out-of-Domain)</span>" if gt_info.get("is_prune") else ""
        st.markdown(
            f"""
            <div class="schema-box-gt">
                <div style="font-size: 0.8rem; text-transform: uppercase; color: #60a5fa; font-weight: 800; letter-spacing: 0.05em;">
                    📘 Ground Truth (Target)
                </div>
                <div style="font-size: 1.3rem; font-weight: 800; color: #ffffff; margin-top: 6px;">
                    {gt_info.get('label', 'Unknown')} {prune_tag}
                </div>
                <div style="margin-top: 4px;">
                    <code style="background-color: #1e3a8a; color: #93c5fd; padding: 2px 8px; border-radius: 4px;">Class Index: {gt_idx}</code>
                </div>
                <div style="margin-top: 14px; color: #cbd5e1; font-size: 0.95rem; line-height: 1.5;">
                    {gt_info.get('description', '')}
                </div>
                <div style="margin-top: 14px; padding-top: 10px; border-top: 1px solid rgba(255,255,255,0.1); font-size: 0.8rem; color: #94a3b8; word-break: break-all;">
                    <b>Model Target String:</b><br/><code>{gt_info.get('formatted_string', '')}</code>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_pred:
        prune_tag_pred = " <span style='color: #f59e0b; font-size: 0.85rem;'>(Prune / Out-of-Domain)</span>" if pred_info.get("is_prune") else ""
        border_color = "#34d399" if is_match else ("#fb7185" if has_prediction else "#64748b")
        st.markdown(
            f"""
            <div class="schema-box-pred" style="border-color: {border_color};">
                <div style="font-size: 0.8rem; text-transform: uppercase; color: #c084fc; font-weight: 800; letter-spacing: 0.05em;">
                    🔮 Model Prediction
                </div>
                <div style="font-size: 1.3rem; font-weight: 800; color: #ffffff; margin-top: 6px;">
                    {pred_info.get('label', 'No Prediction')} {prune_tag_pred}
                </div>
                <div style="margin-top: 4px;">
                    <code style="background-color: #581c87; color: #e9d5ff; padding: 2px 8px; border-radius: 4px;">Class Index: {pred_idx if pred_idx is not None else 'None'}</code>
                </div>
                <div style="margin-top: 14px; color: #cbd5e1; font-size: 0.95rem; line-height: 1.5;">
                    {pred_info.get('description', '')}
                </div>
                <div style="margin-top: 14px; padding-top: 10px; border-top: 1px solid rgba(255,255,255,0.1); font-size: 0.8rem; color: #94a3b8; word-break: break-all;">
                    <b>Predicted String:</b><br/><code>{pred_info.get('formatted_string', '')}</code>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
