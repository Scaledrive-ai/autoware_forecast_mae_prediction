import uuid
import numpy as np
from unique_identifier_msgs.msg import UUID

def uuid_to_str(uuid_msg):
    # uuid_msg.uuid is a numpy array of 16 bytes
    return str(uuid.UUID(bytes=bytes(uuid_msg.uuid)))

def make_ros_uuid(id_str: str) -> UUID:
    u = uuid.UUID(id_str)  # parse UUID string
    msg = UUID()
    msg.uuid = np.frombuffer(u.bytes, dtype=np.uint8)  # 16-byte uint8 array
    return msg
