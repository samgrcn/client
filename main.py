import numpy as np
from channel_coding import transmitter, receiver

# Input message provided during demo
test_message = "Salut je suis heureux j aime les cerises"
if len(test_message) != 40:
    raise ValueError("Message must be 40 characters")

# Transmit
x = transmitter(test_message, r=15, alpha=0.002)
np.savetxt('input.txt', x, fmt='%f')

# Run client.py
import subprocess
subprocess.run([
    'python3', 'client.py',
    '--input_file', 'input.txt',
    '--output_file', 'output.txt',
    '--srv_hostname', 'iscsrv72.epfl.ch',
    '--srv_port', '80'
])

# Receive
Y = np.loadtxt('output.txt')
decoded_message = receiver(Y, r=15, alpha=0.002)
print(f"Original: {test_message}")
print(f"Decoded:  {decoded_message}")
print(f"Errors: {sum(a != b for a, b in zip(test_message, decoded_message))}")
print(f"Energy: {np.sum(np.square(x)):.2f}")