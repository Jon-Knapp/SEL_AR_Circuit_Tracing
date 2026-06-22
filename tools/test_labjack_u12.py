import u12
import time

d = u12.U12()

while True:
    r = d.eDigitalIn(0)
    state = r["state"]

    if state == 1:
        print("connected ✅")
    else:
        print("open ❌")

    time.sleep(0.5)