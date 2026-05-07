"""
Demo: Show the limitation of pure ReAct on a multi-step task.

The task: "Find the largest file in the workspace, read its first 10 lines,
calculate the average line length using python, and write a summary to report.txt"

With pure ReAct, the agent has to figure this out step by step with no plan.
It might:
  - Waste steps searching for files without a strategy
  - Forget it still needs to calculate the average
  - Get confused about which sub-task it's on

With Plan-and-Execute, the agent first creates a plan:
  1. List files and find the largest
  2. Read the first 10 lines
  3. Calculate average line length
  4. Write the summary

Then it executes each sub-task methodically.
"""

from react_agent import ReActAgent

# This is the kind of task where planning helps
complex_goal = (
    "Find the largest .py file in the workspace by reading each file's size. "
    "Read the first 10 lines of that file. "
    "Use python to calculate the average line length of those 10 lines. "
    "Write a summary to 'analysis_report.txt' including: the filename, "
    "its total size, the number of lines examined, and the average line length."
)

print("=" * 70)
print("TASK (Complex, multi-step):")
print(complex_goal)
print("=" * 70)

print("\nWith pure ReAct, the agent has to figure this out step by step.")
print("It might waste steps or forget sub-tasks because there's no plan.\n")

print("With Plan-and-Execute, the agent first creates a plan like this:")
print()
print("  Plan:")
print("    1. Find all .py files and determine which is largest")
print("    2. Read the first 10 lines of the largest file")
print("    3. Use python to calculate average line length of those 10 lines")
print("    4. Write a summary to 'analysis_report.txt' with all findings")
print()
print("  Then it executes each sub-task in order, using ReAct within each one.")
print()
print("The plan prevents backtracking and keeps the agent focused.")
