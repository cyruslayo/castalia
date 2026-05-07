# Filesystem write test - should fail in --read-only container
with open("/tmp/test_write.txt", "w") as f:
    f.write("This should fail!")
print("Write succeeded - SECURITY BREACH!")
