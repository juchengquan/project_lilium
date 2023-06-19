import hashlib
import uuid


def gen_sha():
    return hashlib.sha1(uuid.uuid4().hex.encode("utf-8")).hexdigest()

if __name__ == "__main__":
    print(gen_sha())
    print(gen_sha())
