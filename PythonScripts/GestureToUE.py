from gesturedetect.GestureClient import GestureDetecting

targetIP = "127.0.0.1"
targetPort = 27015
worker = GestureDetecting(targetIP, targetPort)

def main():
    print(f"Main Start...")
    worker.execute()
    print(f"Main End...")


if __name__ == "__main__":
    main()
