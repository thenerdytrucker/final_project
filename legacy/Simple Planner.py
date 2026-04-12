task_list = ["practice coding", "watch a tutorial", "build a project"]
completed_tasks = []
unfinished_tasks = []

# note this has no save funstionality, this is a simple planner to help remind you of the task you put in for the day but once closed any data put in the task list is lost
# this is just to show basic python skills
planing = True
while True:
    completed_tasks = []
    unfinished_tasks = []
    for task in task_list:
        print("="*10)
        print("Todays task list:")
        print('-'*10)
        print(task_list)
        print("="*10)

        print(f"Have I done {task} today!")
        print("="*10)
        valid_answer = False
        while not valid_answer:
            answer = input("(yes/no) or (y/n): ").strip().lower()

            if answer == "yes" or answer == "y":
                completed_tasks.append(task)
                print("="*10)
                print(f"Good job on completing {task}.")
                print("="*10)
                valid_answer = True
            elif answer == "no" or answer == "n":
                unfinished_tasks.append(task)
                print("="*10)
                print(f"Do {task} next so I dont forget!")
                print("="*10)
                valid_answer = True
            else:
                print("Invalid input. Please enter yes, no, y, or n.")

    print("Lets make sure to do all the tasks and after I can relax!")
    print("="*10)
    add_more = True

    while add_more is True:
        More_tasks = input(
            "Are there any other tasks I want to add? (yes/no) or (y/n): ")
        print("="*10)
        if More_tasks.lower() in ['yes', 'y']:
            new_task = input("Whats the new task: ")
            task_list.append(new_task)
            print(f"The task '{new_task}' is added to the list.")
            unfinished_tasks.append(new_task)

        elif More_tasks.lower() in ['no', 'n']:
            add_more = False
            print("No new tasks added.")

    print("="*10)

    def remove_task():
        remove_more = True

        while remove_more is True:
            print("\nUnfinished tasks:")
            for i, task in enumerate(unfinished_tasks):
                print(f"{i+1}. {task}")

            task_to_remove = input(
                "\nEnter the number of the task you want to remove (or 'no' to skip over this): ")

            if task_to_remove.lower() == 'no' or task_to_remove.lower() == 'n':
                remove_more = False
                print("Done removing tasks.")
            else:
                try:
                    task_index = int(task_to_remove) - 1
                    if 0 <= task_index < len(unfinished_tasks):
                        removed = unfinished_tasks.pop(task_index)
                        task_list.remove(removed)
                        print(f"the task '{removed}' removed.")
                    else:
                        print("Invalid task number. Please try again.")
                except ValueError:
                    print("Please enter a valid number.")

    remove_choice = input(
        "Do you want to remove any unwanted tasks? (yes/no) or (y/n): ")
    if remove_choice.lower() in ['yes', 'y']:
        remove_task()

    print("="*10)
    print("Completed tasks: " + str(len(completed_tasks)))
    print('-'*10)
    print(completed_tasks)
    print("="*10)
    print("="*10)
    tasks_remaining = unfinished_tasks.copy()
    print("Tasks remaining to be done: " + str(len(tasks_remaining)))
    print('-'*10)
    print(tasks_remaining)
    print("="*10)

    restart = input(
        "Would you like to check your tasks again? (yes/no) or (y/n): ").strip().lower()
    if restart not in ['yes', 'y']:
        print("Thank you for using the planner. Goodbye!")
        break
