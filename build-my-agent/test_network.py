# Network isolation test - should fail in --network none container
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(2)
try:
    s.connect(("8.8.8.8", 53))
    print("NETWORK ACCESSIBLE - SECURITY BREACH!")
except Exception as e:
    print(f"Network blocked (expected): {type(e).__name__}")
finally:
    s.close()
