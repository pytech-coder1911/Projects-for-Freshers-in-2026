# ============================================================
# PROJECT 5: Autonomous Data Analysis Agent
# Stack: smolagents, Pandas, Plotly, Gradio
# Install: pip install smolagents pandas plotly gradio
#          pip install huggingface_hub
# ============================================================

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io, sys, traceback, json, os
from pathlib import Path

# ---------- Tools the Agent Can Use ----------
# If using smolagents:
try:
    from smolagents import tool, CodeAgent, HfApiModel
    USE_SMOLAGENTS = True
except ImportError:
    USE_SMOLAGENTS = False
    print("smolagents not found, using OpenAI fallback")


# ---- Tool 1: Inspect Dataset ----
def inspect_dataset(filepath: str) -> str:
    """Returns shape, columns, dtypes, and head of a CSV."""
    df = pd.read_csv(filepath)
    info = {
        "shape"    : list(df.shape),
        "columns"  : list(df.columns),
        "dtypes"   : df.dtypes.astype(str).to_dict(),
        "head"     : df.head(5).to_dict(),
        "nulls"    : df.isnull().sum().to_dict(),
        "describe" : df.describe().to_dict()
    }
    return json.dumps(info, indent=2, default=str)


# ---- Tool 2: Run Pandas Code ----
def run_pandas_code(code: str, filepath: str) -> str:
    """Executes arbitrary pandas code on the CSV. Returns stdout."""
    df = pd.read_csv(filepath)
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    local_vars = {"df": df, "pd": pd, "np": np, "result": None}
    try:
        exec(code, local_vars)
        output = buffer.getvalue() or str(local_vars.get("result", "Done."))
    except Exception:
        output = f"Error:\n{traceback.format_exc()}"
    finally:
        sys.stdout = old_stdout
    return output[:3000]


# ---- Tool 3: Auto EDA ----
def auto_eda(filepath: str) -> dict:
    """
    Runs full Exploratory Data Analysis:
    - Missing values
    - Correlation heatmap
    - Distribution plots
    - Outlier detection (IQR)
    Returns a summary dict + saves plots.
    """
    df      = pd.read_csv(filepath)
    summary = {}
    os.makedirs("eda_output", exist_ok=True)

    # Basic stats
    summary["shape"]        = df.shape
    summary["missing_pct"]  = (df.isnull().mean() * 100).round(2).to_dict()
    summary["numeric_cols"] = list(df.select_dtypes(include=np.number).columns)
    summary["cat_cols"]     = list(df.select_dtypes(include="object").columns)

    # Correlation heatmap
    num_df  = df.select_dtypes(include=np.number)
    if num_df.shape[1] > 1:
        corr = num_df.corr()
        fig  = px.imshow(corr, text_auto=True, title="Correlation Heatmap",
                         color_continuous_scale="RdBu_r")
        fig.write_html("eda_output/correlation.html")
        summary["top_correlations"] = (
            corr.abs().unstack()
            .sort_values(ascending=False)
            .drop_duplicates()
            .head(10)
            .to_dict()
        )

    # Outlier detection (IQR method)
    outlier_counts = {}
    for col in num_df.columns:
        Q1, Q3 = num_df[col].quantile([0.25, 0.75])
        IQR    = Q3 - Q1
        n_out  = ((num_df[col] < Q1 - 1.5*IQR) | (num_df[col] > Q3 + 1.5*IQR)).sum()
        if n_out > 0:
            outlier_counts[col] = int(n_out)
    summary["outliers"] = outlier_counts

    # Distribution plots
    for col in num_df.columns[:5]:  # limit to first 5
        fig = px.histogram(df, x=col, title=f"Distribution of {col}",
                           marginal="box")
        fig.write_html(f"eda_output/dist_{col}.html")

    # Save summary
    with open("eda_output/summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print("EDA complete! Charts saved in eda_output/")
    return summary


# ---- Tool 4: Generate Chart ----
def generate_chart(filepath: str, chart_type: str,
                   x_col: str, y_col: str = None,
                   color_col: str = None) -> str:
    """
    Generates interactive Plotly charts.
    chart_type: bar | line | scatter | pie | box | histogram
    """
    df   = pd.read_csv(filepath)
    args = dict(data_frame=df, x=x_col, color=color_col)
    if y_col:
        args["y"] = y_col

    chart_map = {
        "bar"      : px.bar,
        "line"     : px.line,
        "scatter"  : px.scatter,
        "pie"      : lambda **kw: px.pie(df, names=x_col, values=y_col),
        "box"      : px.box,
        "histogram": px.histogram
    }
    func = chart_map.get(chart_type, px.bar)
    fig  = func(**args)
    out  = f"eda_output/{chart_type}_{x_col}.html"
    fig.write_html(out)
    return f"Chart saved to {out}"


# ---------- Gradio Interface ----------
def run_gradio():
    import gradio as gr

    FILEPATH = [None]

    def upload_file(file):
        FILEPATH[0] = file.name
        df     = pd.read_csv(file.name)
        preview = df.head(10).to_html()
        stats   = f"Shape: {df.shape} | Columns: {list(df.columns)}"
        return stats, preview

    def ask_agent(question, history):
        if FILEPATH[0] is None:
            return history + [("Please upload a CSV first.", None)]

        filepath = FILEPATH[0]

        # Build a simple reasoning loop
        df   = pd.read_csv(filepath)
        cols = list(df.columns)
        dtypes = df.dtypes.to_dict()
        head   = df.head(3).to_string()

        # We construct a prompt that asks for pandas code
        context = f"""
Dataset: {filepath}
Columns: {cols}
Dtypes: {dtypes}
Sample:
{head}

User question: {question}

Write Python pandas code to answer this. Use variable `df` for the dataframe.
Print the result. Keep code short and clear.
        """

        if USE_SMOLAGENTS:
            model  = HfApiModel("meta-llama/Meta-Llama-3-8B-Instruct")

            @tool
            def run_code(code: str) -> str:
                """Run pandas code on the dataset."""
                return run_pandas_code(code, filepath)

            agent  = CodeAgent(tools=[run_code], model=model)
            answer = agent.run(context)
        else:
            # Fallback: simple EDA
            summary = auto_eda(filepath)
            answer  = f"EDA complete!\n{json.dumps(summary, indent=2, default=str)[:1000]}"

        history.append((question, str(answer)))
        return history

    with gr.Blocks(title="Autonomous Data Agent") as demo:
        gr.Markdown("# Autonomous Data Analysis Agent\nUpload a CSV and ask anything!")

        with gr.Row():
            file_input = gr.File(label="Upload CSV", file_types=[".csv"])
            stats_out  = gr.Textbox(label="Dataset Info", interactive=False)

        preview = gr.HTML(label="Data Preview")
        file_input.change(upload_file, inputs=file_input,
                          outputs=[stats_out, preview])

        chatbot  = gr.Chatbot(label="Chat with your data")
        question = gr.Textbox(placeholder="e.g. Which product has highest revenue?")
        submit   = gr.Button("Ask Agent")
        submit.click(ask_agent, inputs=[question, chatbot], outputs=chatbot)

        gr.Examples(
            examples=[
                ["What are the top 5 values in the first column by count?"],
                ["Show summary statistics for all numeric columns"],
                ["Are there any missing values? Which columns have the most?"],
                ["What is the correlation between numeric columns?"]
            ],
            inputs=question
        )

    demo.launch(share=True)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "eda":
        # Quick EDA on a file: python data_agent.py eda mydata.csv
        filepath = sys.argv[2] if len(sys.argv) > 2 else "data.csv"
        result   = auto_eda(filepath)
        print(json.dumps(result, indent=2, default=str))
    else:
        run_gradio()

# Run Gradio UI:     python data_agent.py
# Run Quick EDA:     python data_agent.py eda yourfile.csv
