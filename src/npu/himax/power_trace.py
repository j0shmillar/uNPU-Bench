import matplotlib.pyplot as plt
import pandas as pd

file_path = "himax_yolo_size.csv"

df = pd.read_csv(file_path)

print(df.head())

time = df["Time (ms)"][:-100]
max_power = df["Main Avg Power (mW)"][:-100]

plt.figure(figsize=(10, 5))
plt.plot(time, max_power, label="Max Power (mW)", color="black")

# Define segments and colors
segments = [
    (16127.97, "pink", "NPU & mem configured"),
    (16130.27, "lightcoral", "Inference"),
    (16130.95, "yellow", "NPU de-init"),
    (time.max(), "lightgreen", "Idle")
]

# Fill background colors for segments
prev_x = time.min()
for x, color, label in segments:
    plt.axvspan(prev_x, x, color=color, alpha=0.5, label=label)
    prev_x = x

plt.xlabel("Time (ms)", fontsize=18)
plt.ylabel("Peak Power (mW)", fontsize=18)
plt.ylim(79, max_power.max()+1)  # Set y-axis to start from 70
plt.xlim(time.min(), time.max())
plt.legend(loc="upper left", fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.savefig('power_trace.pdf', bbox_inches="tight")
