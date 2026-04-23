import os
import re
from collections import defaultdict
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LOG_FILE = "food_waste_log.txt"
OUTPUT_DIR = "analytics_output"
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
MEAL_KCAL_REFERENCE = 700.0

# Serving size constants (W_serving) in grams as per Section 3.9.3
# These are placeholder values; adjust based on your specific restaurant portions.
SERVING_SIZES = {
    "chicken_curry": 400.0,
    "grilled_salmon": 350.0,
    "pizza": 500.0,
    "samosa": 150.0,
    "garlic_bread": 200.0,
    "steak": 450.0,
    "french_fries": 250.0,
    "hamburger": 350.0,
    "spaghetti_bolognese": 450.0,
    "falafel": 200.0,
}

# Approximate calories per full serving for each class.
CALORIES_PER_SERVING = {
    "chicken_curry": 620.0,
    "grilled_salmon": 540.0,
    "pizza": 1330.0,
    "samosa": 390.0,
    "garlic_bread": 620.0,
    "steak": 1035.0,
    "french_fries": 780.0,
    "hamburger": 875.0,
    "spaghetti_bolognese": 720.0,
    "falafel": 520.0,
}


def format_food_name(food_name):
    return food_name.replace("_", " ").title()


def get_kcal_per_gram(food_name):
    serving_size = SERVING_SIZES.get(food_name)
    calories_per_serving = CALORIES_PER_SERVING.get(food_name)
    if not serving_size or not calories_per_serving:
        return None
    return calories_per_serving / serving_size


def parse_logs(file_path):
    events = []
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return events

    pattern = re.compile(
        r"^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+\|\s+"
        r"food=(?P<food>[\w_]+)"
        r"(?:\s+\|\s+confidence=(?P<confidence>[\d.]+)%)?"
        r"\s+\|\s+weight=(?P<weight>[\d.]+)g$"
    )

    with open(file_path, "r", encoding="utf-8") as file_handle:
        for raw_line in file_handle:
            line = raw_line.strip()
            if not line:
                continue

            match = pattern.match(line)
            if not match:
                continue

            confidence_text = match.group("confidence")
            confidence = float(confidence_text) if confidence_text else None

            events.append(
                {
                    "timestamp": datetime.strptime(
                        match.group("timestamp"), TIMESTAMP_FORMAT
                    ),
                    "food": match.group("food"),
                    "confidence": confidence,
                    "weight": float(match.group("weight")),
                }
            )

    return sorted(events, key=lambda event: event["timestamp"])


def build_summary(events):
    category_mass = defaultdict(float)
    category_kcal = defaultdict(float)
    category_counts = defaultdict(int)
    category_confidence = defaultdict(float)
    category_confidence_counts = defaultdict(int)
    hourly_mass = defaultdict(float)
    daily_mass = defaultdict(float)

    for event in events:
        food = event["food"]
        weight = event["weight"]
        timestamp = event["timestamp"]
        kcal_per_gram = get_kcal_per_gram(food)

        category_mass[food] += weight
        category_counts[food] += 1
        hourly_mass[timestamp.hour] += weight
        daily_mass[timestamp.date()] += weight
        if kcal_per_gram is not None:
            category_kcal[food] += weight * kcal_per_gram

        if event["confidence"] is not None:
            category_confidence[food] += event["confidence"]
            category_confidence_counts[food] += 1

    total_waste = sum(category_mass.values())
    total_kcal = sum(category_kcal.values())
    total_events = len(events)
    average_event_weight = total_waste / total_events if total_events else 0.0
    average_event_kcal = total_kcal / total_events if total_events else 0.0
    people_fed_equivalent = total_kcal / MEAL_KCAL_REFERENCE if MEAL_KCAL_REFERENCE else 0.0

    category_rows = []
    for food in sorted(category_mass, key=category_mass.get, reverse=True):
        total_weight = category_mass[food]
        total_kcal_for_food = category_kcal[food]
        count = category_counts[food]
        avg_weight = total_weight / count if count else 0.0
        avg_kcal = total_kcal_for_food / count if count else 0.0
        serving_size = SERVING_SIZES.get(food)
        avg_waste_ratio = (
            (avg_weight / serving_size) * 100 if serving_size and serving_size > 0 else None
        )
        confidence_count = category_confidence_counts[food]
        avg_confidence = (
            category_confidence[food] / confidence_count if confidence_count else None
        )
        share_pct = (total_weight / total_waste) * 100 if total_waste else 0.0

        category_rows.append(
            {
                "food": food,
                "label": format_food_name(food),
                "total_weight": total_weight,
                "total_kcal": total_kcal_for_food,
                "events": count,
                "average_weight": avg_weight,
                "average_kcal": avg_kcal,
                "avg_waste_ratio": avg_waste_ratio,
                "average_confidence": avg_confidence,
                "share_pct": share_pct,
            }
        )

    top_category = category_rows[0] if category_rows else None
    peak_hour = max(hourly_mass.items(), key=lambda item: item[1]) if hourly_mass else None
    operating_days = len(daily_mass)

    return {
        "events": events,
        "category_rows": category_rows,
        "hourly_mass": hourly_mass,
        "daily_mass": daily_mass,
        "total_waste": total_waste,
        "total_kcal": total_kcal,
        "total_events": total_events,
        "average_event_weight": average_event_weight,
        "average_event_kcal": average_event_kcal,
        "people_fed_equivalent": people_fed_equivalent,
        "top_category": top_category,
        "peak_hour": peak_hour,
        "operating_days": operating_days,
    }


def print_report(summary):

    print("FOOD WASTE ANALYTICS REPORT")
    print("-" * 40)
    print(f"Total waste captured: {summary['total_waste']:.2f} g")
    print(f"Total energy wasted: {summary['total_kcal']:.0f} kcal")
    print(f"Logged waste events: {summary['total_events']}")
    print(f"Average waste per event: {summary['average_event_weight']:.2f} g")
    print(f"Average energy per event: {summary['average_event_kcal']:.0f} kcal")
    print(f"Operating days covered: {summary['operating_days']}")
    print(
        "People who could have been fed: "
        f"{summary['people_fed_equivalent']:.1f} meal-equivalents "
        f"(assuming {MEAL_KCAL_REFERENCE:.0f} kcal/person)"
    )

    if summary["top_category"] is not None:
        top_category = summary["top_category"]
        print(
            f"Top waste driver: {top_category['label']} "
            f"({top_category['total_weight']:.2f} g, {top_category['total_kcal']:.0f} kcal, "
            f"{top_category['share_pct']:.1f}% of total)"
        )

    if summary["peak_hour"] is not None:
        peak_hour, peak_hour_weight = summary["peak_hour"]
        print(
            f"Peak waste hour: {peak_hour:02d}:00-{peak_hour:02d}:59 "
            f"({peak_hour_weight:.2f} g)"
        )

    print("\n" + "-" * 116)
    print(
        f"{'Food Category':<22} | {'Total Waste (g)':>15} | {'Total kcal':>10} | "
        f"{'Share %':>8} | {'Events':>6} | {'Avg/Event (g)':>13} | "
        f"{'Avg Waste %':>11} | {'Avg Conf %':>10}"
    )
    print("-" * 116)

    for row in summary["category_rows"]:
        ratio_text = f"{row['avg_waste_ratio']:.2f}%" if row["avg_waste_ratio"] is not None else "n/a"
        confidence_text = (
            f"{row['average_confidence']:.2f}%"
            if row["average_confidence"] is not None
            else "n/a"
        )
        print(
            f"{row['label']:<22} | {row['total_weight']:>15.2f} | {row['total_kcal']:>10.0f} | "
            f"{row['share_pct']:>7.1f}% | {row['events']:>6} | {row['average_weight']:>13.2f} | "
            f"{ratio_text:>11} | {confidence_text:>10}"
        )


def save_dashboard(summary, output_dir):
    rows = summary["category_rows"]
    labels = [row["label"] for row in rows]
    total_weights = [row["total_weight"] for row in rows]
    total_kcal = [row["total_kcal"] for row in rows]
    event_counts = [row["events"] for row in rows]
    hours = list(range(24))
    hourly_weights = [summary["hourly_mass"].get(hour, 0.0) for hour in hours]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(18, 11))
    fig.suptitle("Food Waste Decision Dashboard", fontsize=18, fontweight="bold")

    axes[0, 0].bar(labels, total_weights, color="#c44e52")
    axes[0, 0].set_title("Total Waste by Category")
    axes[0, 0].set_ylabel("Weight (g)")
    axes[0, 0].tick_params(axis="x", rotation=25)

    axes[0, 1].bar(labels, event_counts, color="#4c72b0")
    axes[0, 1].set_title("Waste Events by Category")
    axes[0, 1].set_ylabel("Event Count")
    axes[0, 1].tick_params(axis="x", rotation=25)

    axes[1, 0].bar(labels, total_kcal, color="#55a868")
    axes[1, 0].set_title("Calories Lost by Category")
    axes[1, 0].set_ylabel("Energy (kcal)")
    axes[1, 0].tick_params(axis="x", rotation=25)

    axes[1, 1].bar(hours, hourly_weights, color="#8172b2")
    axes[1, 1].set_title("Waste Distribution by Hour")
    axes[1, 1].set_xlabel("Hour of Day")
    axes[1, 1].set_ylabel("Weight (g)")
    axes[1, 1].set_xticks(hours)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    output_path = os.path.join(output_dir, "waste_dashboard.png")
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_pareto_chart(summary, output_dir):
    rows = summary["category_rows"]
    labels = [row["label"] for row in rows]
    total_weights = [row["total_weight"] for row in rows]
    total_waste = summary["total_waste"]

    cumulative_share = []
    running_total = 0.0
    for weight in total_weights:
        running_total += weight
        cumulative_share.append((running_total / total_waste) * 100 if total_waste else 0.0)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax1 = plt.subplots(figsize=(14, 7))
    ax1.bar(labels, total_weights, color="#dd8452")
    ax1.set_title("Pareto Analysis of Waste Drivers")
    ax1.set_ylabel("Weight (g)")
    ax1.tick_params(axis="x", rotation=25)

    ax2 = ax1.twinx()
    ax2.plot(labels, cumulative_share, color="#222222", marker="o", linewidth=2)
    ax2.axhline(80, color="#c44e52", linestyle="--", linewidth=1.5)
    ax2.set_ylabel("Cumulative Share (%)")
    ax2.set_ylim(0, 110)

    fig.tight_layout()
    output_path = os.path.join(output_dir, "waste_pareto.png")
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def ensure_output_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


if __name__ == "__main__":
    log_data = parse_logs(LOG_FILE)
    if not log_data:
        print("No data available to analyze.")
        raise SystemExit(0)

    output_dir = ensure_output_dir(OUTPUT_DIR)
    summary = build_summary(log_data)

    print_report(summary)

    dashboard_path = save_dashboard(summary, output_dir)
    pareto_path = save_pareto_chart(summary, output_dir)

