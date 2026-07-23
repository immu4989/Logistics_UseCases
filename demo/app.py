"""Interactive demo for the Shipping and Logistics Use Cases repo.

Three of the twelve use cases, wired to live controls: score a shipment for
delivery-commit risk (with a per-shipment SHAP breakdown), read its ETA as a
quantile interval with a keepable promise, and watch the intervention budget
allocate itself. Everything runs on synthetic data trained in-process; there is
no private data behind this Space.
"""

from __future__ import annotations

import gradio as gr

import logic

REPO = "https://github.com/immu4989/Logistics_UseCases"

INTRO = f"""
# 🚚 Shipping & Logistics ML — live demo

Three use cases from [the open-source repo]({REPO}), driven by the controls below.
Build a shipment and score it two ways, then see how a fixed intervention budget
should actually be spent. All models are trained in-process on documented synthetic
generators — no real customer data, and every number here reproduces the repo's tests.
"""

SHIP_INTRO = """
### Build a shipment
Set the operational conditions on the left, then score it. The **miss-risk** model and
the **ETA** model both read the same shipment, using only information known at induction
time (no cheating with in-transit scans).
"""

BUDGET_INTRO = """
### Spend an intervention budget
A risk score is not a decision. Given a day of 20,000 shipments and a fixed daily budget,
which ones do you reroute, upgrade, or leave alone? Move the budget and compare the
policies. **Expected-value greedy** weighs each shipment's risk against the cost of a
missed delivery for *that* customer; **top-K** just flags the scariest scores.
"""

ABOUT = f"""
### About this demo

Part of **[Shipping and Logistics Use Cases]({REPO})**, twelve self-contained,
end-to-end machine-learning projects for parcel and freight operations. Each ships with
a documented synthetic generator, audited cleaning, an honest operational baseline,
evaluation in dollars and days, and explainability grounded by tests.

- **Miss risk** comes from `delivery-commit-prediction` (XGBoost + SHAP).
- **ETA** comes from `eta-regression` (XGBoost quantile models, monotone-rearranged).
- **Budget** comes from `intervention-optimization` (expected-value allocation,
  counterfactually evaluated).

The full repo also covers volume forecasting, route optimization, dynamic pricing,
predictive maintenance, address resolution, returns prediction, capacity planning,
network anomaly detection, and exception triage — several validated on real public
datasets (Olist, UCI AI4I, CVRPLIB). The models here are trained small for a fast
demo; the repo's reported numbers come from full runs.
"""


def build() -> gr.Blocks:
    with gr.Blocks(title="Shipping & Logistics ML demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown(INTRO)

        with gr.Tab("Score a shipment"):
            gr.Markdown(SHIP_INTRO)
            with gr.Row():
                with gr.Column(scale=1):
                    distance = gr.Slider(5, 3000, value=1800, step=5, label="Lane distance (miles)")
                    service = gr.Dropdown(
                        ["overnight", "two_day", "ground"], value="ground", label="Service level"
                    )
                    origin_cong = gr.Slider(0, 1, value=0.7, step=0.05, label="Origin hub congestion")
                    dest_cong = gr.Slider(0, 1, value=0.6, step=0.05, label="Destination hub congestion")
                    weather = gr.Slider(0, 3, value=2, step=1, label="Destination weather (0 clear – 3 severe)")
                    cutoff = gr.Slider(
                        -120, 120, value=20, step=5, label="Pickup vs facility cutoff (min; + is late)"
                    )
                    dest_type = gr.Dropdown(
                        ["residential", "commercial"], value="residential", label="Destination type"
                    )
                    with gr.Row():
                        peak = gr.Checkbox(value=True, label="Peak season")
                        rural = gr.Checkbox(value=True, label="Rural destination")
                    score_btn = gr.Button("Score this shipment", variant="primary")

                with gr.Column(scale=2):
                    with gr.Tab("Will it miss the promise?"):
                        risk_label = gr.Markdown()
                        risk_plot = gr.Plot()
                    with gr.Tab("When will it arrive?"):
                        eta_label = gr.Markdown()
                        eta_plot = gr.Plot()

            inputs = [distance, service, origin_cong, dest_cong, weather, cutoff, peak, rural, dest_type]
            score_btn.click(logic.score_commit, inputs=inputs, outputs=[risk_label, risk_plot])
            score_btn.click(logic.predict_eta, inputs=inputs, outputs=[eta_label, eta_plot])

        with gr.Tab("Spend the budget"):
            gr.Markdown(BUDGET_INTRO)
            budget = gr.Slider(1000, 20000, value=6000, step=500, label="Daily intervention budget ($)")
            run_btn = gr.Button("Allocate the budget", variant="primary")
            budget_takeaway = gr.Markdown()
            budget_plot = gr.Plot()
            budget_table = gr.Dataframe(wrap=True)
            run_btn.click(
                logic.run_budget, inputs=budget, outputs=[budget_takeaway, budget_plot, budget_table]
            )

        with gr.Tab("About"):
            gr.Markdown(ABOUT)

    return demo


if __name__ == "__main__":
    logic.warmup()
    build().launch()
