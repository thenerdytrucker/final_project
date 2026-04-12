task_list = ["practice coding", "watch a tutorial", "build a project"]
completed_tasks = []
unfinished_tasks = []

for task in task_list:
    print("="*10)
    print(f"Have I done {task} today!")
    print("="*10)
    valid_answer = False
    while not valid_answer:
        answer = input("(yes/no) or (y/n): ").strip().lower()
        if answer == "yes" or answer == "y":
            completed_tasks.append(task)
            print("="*10)
            print(f"Great! You've completed {task}.")
            print("="*10)
            valid_answer = True
        elif answer == "no" or answer == "n":
            unfinished_tasks.append(task)
            print("="*10)
            print(f"Do {task} next!")
            print("="*10)
            valid_answer = True
        else:
            print("Invalid input. Please enter yes, no, y, or n.")

print("Always keep learning and improving your skills!")
print("="*10)
More_tasks = input(
    "Are there any other tasks you want to add? (yes/no) or (y/n): ")
print("="*10)
if More_tasks.lower() in ['yes', 'y']:
    new_task = input("Please enter the new task: ")
    task_list.append(new_task)
    print(f"New task '{new_task}' added to the list.")
    unfinished_tasks.append(new_task)
elif More_tasks.lower() in ['no', 'n']:
    print("No new tasks added.")

print("="*10)
print("Completed tasks:")
print(completed_tasks)
print("="*10)
print("="*10)
tasks_remaining = unfinished_tasks.copy()
print("Tasks remaining to be done:")
print(tasks_remaining)
print("="*10)
