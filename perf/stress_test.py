import requests
import threading
import time

URL = "http://127.0.0.1:8000/chat"

def send_request(i):
    try:
        res = requests.post(URL, json={"message": "I want a loan"})
        print(f"[{i}] {res.status_code}")
    except Exception as e:
        print(f"[{i}] Error:", e)


def run_test(users=20):
    threads = []
    start = time.time()

    for i in range(users):
        t = threading.Thread(target=send_request, args=(i,))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    print("Total Time:", round(time.time() - start, 2))


if __name__ == "__main__":
    run_test(20)