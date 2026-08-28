"""Streamlit interface for the journal OCR pipeline and reading-order V2."""

from __future__ import annotations

import hashlib
import io
import json
import sys
from importlib import import_module
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR
RESULT_VERSION = "baseline_center_v2_2026-08-27"
PIPELINE_CODE_DIR = APP_DIR
FROZEN_CONFIG_PATH = APP_DIR / "deployment_config.json"
if str(PIPELINE_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_CODE_DIR))

RevisedOCRPipeline = import_module("pipeline").RevisedOCRPipeline


st.set_page_config(
    page_title="OCR Aksara Jawa - Eksperimen Jurnal",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;500;600;700&family=Source+Sans+3:wght@300;400;500;600;700&display=swap');

    :root {
        color-scheme: light;
        --ink: #3d2b1f;
        --muted: #806b58;
        --line: #d4c4b0;
        --surface: #f0e6d9;
        --canvas: #faf6f1;
        --primary: #3d2b1f;
        --primary-hover: #5c3d2e;
        --gold: #c4a882;
        --gold-dark: #8b6914;
        --cream: #f5e6d3;
        --paper: #fffdf9;
    }
    .stApp {
        background: var(--canvas);
        color: var(--ink);
        color-scheme: light;
        font-family: 'Source Sans 3', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    .block-container {
        max-width: 1240px;
        padding-top: 1rem;
        padding-bottom: 2.5rem;
    }
    [data-testid="stHeader"] { background: transparent; height: 0.5rem; }
    [data-testid="stToolbar"], #MainMenu, footer { visibility: hidden; }
    h1, h2, h3, p, label, button, input, textarea { letter-spacing: 0; }
    .app-title {
        background: var(--primary);
        border-left: 5px solid var(--gold);
        border-radius: 6px;
        padding: 1.05rem 1.25rem 1rem;
        margin: 0 0 0.8rem;
        text-align: center;
    }
    .app-title h1 {
        color: var(--cream);
        font-family: 'Playfair Display', Georgia, serif;
        font-size: 1.75rem;
        font-weight: 700;
        margin: 0;
    }
    .app-title p { color: var(--gold); margin: 0.28rem 0 0; font-size: 0.9rem; }
    .scope-note {
        border-left: 3px solid var(--gold-dark);
        background: var(--surface);
        padding: 0.62rem 0.85rem;
        color: var(--primary-hover);
        font-size: 0.82rem;
        margin-bottom: 0.8rem;
    }
    .section-heading {
        color: var(--ink);
        font-family: 'Playfair Display', Georgia, serif;
        font-size: 1rem;
        font-weight: 600;
        margin: 0.35rem 0 0.1rem;
    }
    .section-caption { color: var(--muted); font-size: 0.79rem; margin-bottom: 0.55rem; }
    .preview-empty {
        min-height: 280px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        border: 1px dashed #b89d7a;
        border-radius: 5px;
        background: var(--surface);
        color: var(--muted);
    }
    .preview-empty strong { color: var(--ink); font-size: 0.94rem; }
    .file-meta {
        display: flex;
        justify-content: center;
        gap: 0.7rem;
        flex-wrap: wrap;
        margin-top: 0.35rem;
        color: var(--muted);
        font-size: 0.76rem;
    }
    .file-meta strong { color: var(--ink); }
    [data-testid="stFileUploaderDropzone"] {
        background: var(--surface);
        border: 1px dashed #b89d7a;
        border-radius: 5px;
        min-height: 82px;
        align-items: center;
        gap: 1rem;
    }
    [data-testid="stFileUploaderDropzone"] *,
    [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] * {
        color: var(--ink) !important;
        -webkit-text-fill-color: var(--ink) !important;
    }
    [data-testid="stFileUploaderDropzone"] small,
    [data-testid="stFileUploaderDropzone"] [data-testid="stFileUploaderDropzoneInstructions"] span {
        color: var(--muted) !important;
        -webkit-text-fill-color: var(--muted) !important;
    }
    [data-testid="stFileUploaderDropzoneInstructions"] {
        order: 2;
        flex: 1;
    }
    [data-testid="stFileUploaderDropzoneInstructions"] span {
        display: none !important;
    }
    [data-testid="stFileUploaderDropzoneInstructions"] small {
        font-size: 0 !important;
    }
    [data-testid="stFileUploaderDropzoneInstructions"] small::after {
        content: "200MB per file \\2022  JPG, PNG, BMP, TIF";
        color: var(--muted);
        -webkit-text-fill-color: var(--muted);
        font-size: 0.86rem;
        white-space: normal;
    }
    [data-testid="stFileUploaderDropzone"] button {
        order: 1;
        background: var(--paper) !important;
        border-color: var(--line) !important;
        color: var(--ink) !important;
        -webkit-text-fill-color: var(--ink) !important;
        font-size: 0 !important;
    }
    [data-testid="stFileUploaderDropzone"] button::after {
        content: "Upload";
        color: var(--ink);
        -webkit-text-fill-color: var(--ink);
        font-size: 0.9rem;
        font-weight: 600;
    }
    [data-testid="stFileUploaderDropzone"] button::before {
        content: "upload";
        color: var(--ink);
        -webkit-text-fill-color: var(--ink);
        font-family: "Material Symbols Rounded";
        font-size: 1.15rem;
        font-style: normal;
        font-weight: 400;
        line-height: 1;
        margin-right: 0.45rem;
        font-variation-settings: "FILL" 0, "wght" 400, "GRAD" 0, "opsz" 24;
    }
    [data-testid="stFileUploaderDropzone"] button:hover {
        background: var(--cream) !important;
        border-color: var(--gold) !important;
    }
    [data-testid="stImage"] {
        display: flex;
        width: 100%;
        flex-direction: column;
        align-items: center;
    }
    [data-testid="stImage"] figure {
        margin-left: auto !important;
        margin-right: auto !important;
    }
    [data-testid="stImage"] img {
        border: 1px solid var(--line);
        border-radius: 4px;
        background: var(--paper);
    }
    [data-testid="stImage"] figcaption { text-align: center; }
    [data-testid="stMetric"] {
        background: var(--surface);
        border: 1px solid var(--line);
        border-top: 3px solid var(--gold);
        border-radius: 4px;
        padding: 0.62rem 0.78rem;
        min-height: 88px;
    }
    [data-testid="stMetricLabel"] { color: var(--muted); font-size: 0.76rem; }
    [data-testid="stMetricValue"] {
        color: var(--ink);
        font-family: 'Playfair Display', Georgia, serif;
        font-size: 1.55rem;
    }
    .stButton > button, .stDownloadButton > button { border-radius: 4px; min-height: 2.55rem; }
    .stButton > button[kind="primary"] {
        background: var(--primary);
        border-color: var(--primary);
        color: var(--cream);
    }
    .stButton > button[kind="primary"]:hover {
        background: var(--primary-hover);
        border-color: var(--primary-hover);
        color: var(--cream);
    }
    .stDownloadButton > button {
        background: var(--primary) !important;
        border-color: var(--primary) !important;
        color: var(--cream) !important;
        -webkit-text-fill-color: var(--cream) !important;
    }
    .stDownloadButton > button *,
    .stDownloadButton > button svg {
        color: var(--cream) !important;
        fill: var(--cream) !important;
        -webkit-text-fill-color: var(--cream) !important;
    }
    .stDownloadButton > button:hover {
        background: var(--primary-hover) !important;
        border-color: var(--primary-hover) !important;
    }
    [data-testid="stTabs"] [data-baseweb="tab-list"] {
        gap: 0.35rem;
        border-bottom: 1px solid var(--line);
    }
    [data-testid="stTabs"] [data-baseweb="tab"] {
        min-height: 2.7rem;
        padding: 0 0.8rem;
    }
    [data-testid="stTabs"] button[role="tab"],
    [data-testid="stTabs"] button[role="tab"] * {
        color: var(--ink) !important;
        -webkit-text-fill-color: var(--ink) !important;
    }
    [data-testid="stTabs"] button[role="tab"][aria-selected="true"],
    [data-testid="stTabs"] button[role="tab"][aria-selected="true"] * {
        color: var(--primary-hover) !important;
        -webkit-text-fill-color: var(--primary-hover) !important;
    }
    [data-testid="stTextArea"] textarea:disabled {
        -webkit-text-fill-color: var(--cream);
        color: var(--cream);
        opacity: 1;
        background: var(--primary);
        border-color: var(--line);
        font-family: 'Source Sans 3', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        line-height: 1.55;
    }
    [data-testid="stTabs"], [data-testid="stMetric"], [data-testid="stFileUploader"],
    [data-testid="stDataFrame"], [data-testid="stExpander"] {
        font-family: 'Source Sans 3', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    [data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 4px; }
    [data-testid="stExpander"] { background: var(--surface); border-color: var(--line); }
    .small-note { color: var(--muted); font-size: 0.76rem; }
    @media (max-width: 720px) {
        .block-container { padding: 0.75rem 0.85rem 2rem; }
        .app-title { padding: 0.9rem 1rem; }
        .app-title h1 { font-size: 1.4rem; }
        .preview-empty { min-height: 190px; }
        [data-testid="stMetricValue"] { font-size: 1.3rem; }
        [data-testid="stTabs"] [data-baseweb="tab-list"] {
            overflow-x: auto;
            scrollbar-width: thin;
        }
        [data-testid="stTabs"] [data-baseweb="tab"] {
            flex: 0 0 auto;
            padding: 0 0.55rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_frozen_config():
    if not FROZEN_CONFIG_PATH.is_file():
        raise FileNotFoundError(FROZEN_CONFIG_PATH)
    return json.loads(FROZEN_CONFIG_PATH.read_text(encoding="utf-8"))


@st.cache_resource(show_spinner=False)
def load_pipeline():
    config = load_frozen_config()
    yolo_weights = PROJECT_ROOT / config["models"]["yolo_weights"]
    crnn_weights = PROJECT_ROOT / config["models"]["crnn_weights"]
    for path in (yolo_weights, crnn_weights):
        if not path.is_file():
            raise FileNotFoundError(path)
    if sha256(yolo_weights) != config["models"]["yolo_sha256"]:
        raise ValueError("Checksum model YOLOv8n tidak sesuai konfigurasi beku")
    if sha256(crnn_weights) != config["models"]["crnn_sha256"]:
        raise ValueError("Checksum model CRNN tidak sesuai konfigurasi beku")
    pipeline_config = config["pipeline"]
    return RevisedOCRPipeline(
        yolo_weights=yolo_weights,
        crnn_weights=crnn_weights,
        device="cpu",
        preprocessing_mode=pipeline_config["crnn_preprocessing"],
        imgsz=pipeline_config["imgsz"],
        nms_iou=pipeline_config["nms_iou"],
        max_det=pipeline_config["max_det"],
    )


def decode_uploaded_image(uploaded_file):
    encoded = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Berkas tidak dapat dibaca sebagai gambar")
    return image


def display_font(size=16):
    candidates = (
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    )
    for path in candidates:
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def annotate_image(image, rows):
    canvas = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(canvas)
    font = display_font(max(12, min(18, round(canvas.width / 70))))
    palette = (
        (22, 125, 120),
        (179, 107, 21),
        (74, 101, 140),
        (153, 78, 118),
        (72, 125, 68),
        (180, 68, 68),
    )
    for row in rows:
        x1, y1, x2, y2 = row["bbox"]
        color = palette[row["line_id"] % len(palette)]
        label = f"{row['order_index'] + 1}. {row['text']}"
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        label_width = right - left + 8
        label_height = bottom - top + 6
        label_y = max(0, y1 - label_height)
        draw.rectangle((x1, y1, x2, y2), outline=color, width=2)
        draw.rectangle((x1, label_y, x1 + label_width, label_y + label_height), fill=color)
        draw.text((x1 + 4, label_y + 2), label, fill="white", font=font)
    return np.asarray(canvas)


def fit_preview(image, max_width=720, max_height=560):
    height, width = image.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)
    if scale >= 1.0:
        return image
    return cv2.resize(
        image,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


def rgb_png_bytes(image):
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def rows_dataframe(rows):
    return pd.DataFrame(
        [
            {
                "Urutan": row["order_index"] + 1,
                "Baris": row["line_id"] + 1,
                "Kata": row["text"],
                "Confidence deteksi": row["detection_confidence"],
                "Confidence pengenalan": row["recognition_confidence"],
                "Bounding box": ", ".join(map(str, row["bbox"])),
            }
            for row in rows
        ]
    )


def apply_order_edits(rows, edited_frame):
    by_source_index = {row["source_index"]: row for row in edited_frame.to_dict("records")}
    corrected = []
    for source_index, source in enumerate(rows):
        item = dict(source)
        edit = by_source_index[source_index]
        line_value = edit["Baris"]
        order_value = edit["Urutan dalam baris"]
        item["line_id"] = (
            source["line_id"] if pd.isna(line_value) else max(0, int(line_value) - 1)
        )
        item["order_in_line"] = (
            source["order_in_line"]
            if pd.isna(order_value)
            else max(0, int(order_value) - 1)
        )
        item["source_index"] = source_index
        corrected.append(item)
    corrected.sort(
        key=lambda item: (
            item["line_id"],
            item["order_in_line"],
            item["source_index"],
        )
    )
    for order_index, item in enumerate(corrected):
        item["order_index"] = order_index
    return corrected


def text_from_rows(rows):
    lines = {}
    for row in rows:
        lines.setdefault(row["line_id"], []).append(row["text"])
    text_lines = [" ".join(lines[line_id]) for line_id in sorted(lines)]
    return "\n".join(text_lines), text_lines


def dataframe_csv(frame):
    return frame.to_csv(index=False).encode("utf-8")


config = load_frozen_config()
default_threshold = float(config["pipeline"]["confidence_threshold"])

st.markdown(
    """
    <div class="app-title">
        <h1>OCR Aksara Jawa</h1>
        <p>Eksperimen jurnal | Manuskrip Sasradiningrat II</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="scope-note">
        Model dilatih pada kata target yang dianotasi secara terpilih. Hasil tidak merepresentasikan
        transkripsi exhaustif seluruh kata pada halaman dan tetap memerlukan pemeriksaan manusia.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="section-heading">Input halaman</div>', unsafe_allow_html=True)

control_left, preview_right = st.columns([0.9, 1.1], gap="large")
with control_left:
    uploaded_file = st.file_uploader(
        "Berkas citra",
        type=("jpg", "jpeg", "png", "bmp", "tif", "tiff"),
    )
    confidence = st.slider(
        "Confidence deteksi",
        min_value=0.05,
        max_value=0.75,
        value=default_threshold,
        step=0.05,
    )
    st.markdown(
        f'<div class="small-note">Nilai evaluasi yang dibekukan: {default_threshold:.2f}</div>',
        unsafe_allow_html=True,
    )
    process_clicked = st.button(
        "Proses OCR",
        type="primary",
        icon=":material/document_scanner:",
        use_container_width=True,
        disabled=uploaded_file is None,
    )

preview_input_image = None
upload_hash = None
with preview_right:
    st.markdown('<div class="section-heading">Pratinjau</div>', unsafe_allow_html=True)
    if uploaded_file is None:
        st.markdown(
            """
            <div class="preview-empty">
                <strong>Belum ada citra</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        try:
            preview_input_image = decode_uploaded_image(uploaded_file)
            upload_hash = hashlib.sha256(uploaded_file.getvalue()).hexdigest()
            preview_rgb = cv2.cvtColor(preview_input_image, cv2.COLOR_BGR2RGB)
            preview_rgb = fit_preview(preview_rgb, max_width=560, max_height=400)
            st.image(preview_rgb)
            image_height, image_width = preview_input_image.shape[:2]
            size_kb = len(uploaded_file.getvalue()) / 1024
            st.markdown(
                f"<div class='file-meta'><span><strong>{image_width} x {image_height}</strong> piksel</span>"
                f"<span><strong>{size_kb:.1f} KB</strong></span></div>",
                unsafe_allow_html=True,
            )
        except Exception as error:
            st.error(f"Pratinjau gagal dibuat: {error}")

stored_upload_hash = st.session_state.get("journal_ocr_upload_hash")
if stored_upload_hash != upload_hash:
    for state_key in (
        "journal_ocr_result",
        "journal_ocr_image",
        "journal_ocr_filename",
        "journal_ocr_result_version",
        "journal_ocr_upload_hash",
    ):
        st.session_state.pop(state_key, None)

if process_clicked and preview_input_image is not None:
    try:
        with st.spinner("Menjalankan deteksi dan pengenalan..."):
            pipeline = load_pipeline()
            result = pipeline.process_image(
                preview_input_image,
                confidence=confidence,
                margin_ratio=config["pipeline"]["crop_margin"],
                reading_order="baseline_center_v2",
                center_tolerance=config["pipeline"]["line_center_tolerance"],
                min_vertical_overlap=config["pipeline"]["line_min_vertical_overlap"],
            )
        st.session_state["journal_ocr_result"] = result
        st.session_state["journal_ocr_image"] = preview_input_image
        st.session_state["journal_ocr_filename"] = uploaded_file.name
        st.session_state["journal_ocr_result_version"] = RESULT_VERSION
        st.session_state["journal_ocr_upload_hash"] = upload_hash
    except Exception as error:
        st.error(f"OCR gagal dijalankan: {error}")

result = st.session_state.get("journal_ocr_result")
input_image = st.session_state.get("journal_ocr_image")
filename = st.session_state.get("journal_ocr_filename", "hasil_ocr")
if result is not None and st.session_state.get("journal_ocr_result_version") != RESULT_VERSION:
    result = None
    input_image = None
    st.session_state.pop("journal_ocr_result", None)
    st.session_state.pop("journal_ocr_image", None)
    st.info("Versi urutan baca telah diperbarui. Jalankan Proses OCR sekali lagi.")

if result is not None and input_image is not None:
    st.markdown('<div class="section-heading">Hasil OCR</div>', unsafe_allow_html=True)
    metrics = st.columns(4)
    metrics[0].metric("Kata terdeteksi", result["num_detections"])
    metrics[1].metric("Deteksi", f"{result['timing']['detection_ms']:.0f} ms")
    metrics[2].metric("Pengenalan", f"{result['timing']['recognition_ms']:.0f} ms")
    metrics[3].metric("Total", f"{result['timing']['total_ms']:.0f} ms")

    if not result["results"]:
        st.warning("Tidak ada kata yang melewati confidence threshold.")
    else:
        summary_tab, order_tab, detail_tab, crops_tab = st.tabs(
            ("Ringkasan", "Urutan baca", "Detail deteksi", "Crop kata")
        )
        order_frame = pd.DataFrame(
            [
                {
                    "source_index": source_index,
                    "Kata": row["text"],
                    "Baris": row["line_id"] + 1,
                    "Urutan dalam baris": row["order_in_line"] + 1,
                }
                for source_index, row in enumerate(result["results"])
            ]
        )
        editor_key = (
            f"reading_order_{RESULT_VERSION}_{Path(filename).stem}_"
            f"{result['configuration']['confidence_threshold']:.2f}"
        )
        with order_tab:
            st.markdown("#### Koreksi urutan")
            st.caption("Koreksi manual untuk bounding box ambigu.")
            edited_order = st.data_editor(
                order_frame,
                key=editor_key,
                use_container_width=True,
                hide_index=True,
                height=min(460, 38 * (len(order_frame) + 1)),
                disabled=("source_index", "Kata"),
                column_config={
                    "source_index": None,
                    "Kata": st.column_config.TextColumn(width="large"),
                    "Baris": st.column_config.NumberColumn(min_value=1, step=1),
                    "Urutan dalam baris": st.column_config.NumberColumn(
                        min_value=1, step=1, width="medium"
                    ),
                },
            )

        display_rows = apply_order_edits(result["results"], edited_order)
        display_text, display_lines = text_from_rows(display_rows)
        output_frame = rows_dataframe(display_rows)
        annotated = annotate_image(input_image, display_rows)

        with summary_tab:
            text_column, visual_column = st.columns([0.85, 1.15], gap="large")
            with text_column:
                st.markdown("#### Transliterasi Latin")
                st.text_area(
                    "Transliterasi Latin",
                    value=display_text,
                    height=260,
                    disabled=True,
                    label_visibility="collapsed",
                )
                download_text, download_csv = st.columns(2)
                download_text.download_button(
                    "Teks",
                    data=display_text.encode("utf-8"),
                    file_name=f"{Path(filename).stem}_ocr.txt",
                    mime="text/plain",
                    icon=":material/download:",
                    use_container_width=True,
                )
                download_csv.download_button(
                    "CSV",
                    data=dataframe_csv(output_frame),
                    file_name=f"{Path(filename).stem}_ocr.csv",
                    mime="text/csv",
                    icon=":material/table_view:",
                    use_container_width=True,
                )
            with visual_column:
                st.markdown("#### Visualisasi deteksi")
                annotated_preview = fit_preview(annotated, max_width=680, max_height=560)
                st.image(annotated_preview, caption=filename)
                st.download_button(
                    "Unduh visualisasi",
                    data=rgb_png_bytes(annotated),
                    file_name=f"{Path(filename).stem}_visualisasi.png",
                    mime="image/png",
                    icon=":material/download:",
                    use_container_width=True,
                )

        with detail_tab:
            st.dataframe(
                output_frame,
                use_container_width=True,
                hide_index=True,
                height=470,
                column_config={
                    "Confidence deteksi": st.column_config.NumberColumn(format="%.3f"),
                    "Confidence pengenalan": st.column_config.NumberColumn(format="%.3f"),
                },
            )
            with st.expander("Raw CTC output"):
                for row in display_rows:
                    st.code(
                        f"{row['order_index'] + 1}. {row['text']}\n"
                        f"indices: {row['raw_argmax']}\n"
                        f"collapsed: {row['collapsed_argmax']}",
                        language="text",
                    )

        with crops_tab:
            crop_columns = st.columns(4)
            for index, row in enumerate(display_rows):
                with crop_columns[index % 4]:
                    crop_rgb = cv2.cvtColor(row["crop"], cv2.COLOR_BGR2RGB)
                    crop_rgb = fit_preview(crop_rgb, max_width=260, max_height=120)
                    st.image(crop_rgb, use_container_width=True)
                    st.caption(
                        f"{row['order_index'] + 1}. {row['text']} | "
                        f"baris {row['line_id'] + 1}"
                    )

    with st.expander("Konfigurasi inferensi"):
        st.json(
            {
                **result["configuration"],
                "model": "YOLOv8n clean split + CRNN direct resize",
                "reading_order_status": "post-evaluation application fix",
                "configuration_source": str(FROZEN_CONFIG_PATH.relative_to(PROJECT_ROOT)),
            }
        )
