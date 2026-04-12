import time
break_len = int(input("Enter how long of a break do you need in seconds: "))
for i in range(break_len, 0, -1):
    print(f"countdown: {i}")

    time.sleep(1)
print("break over")
