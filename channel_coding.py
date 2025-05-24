import numpy as np
from scipy.linalg import hadamard
import random

# Define the alphabet (64 characters)
alphabet = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .'
char_to_bits = {c: format(i, '06b') for i, c in enumerate(alphabet)}
bits_to_char = {format(i, '06b'): c for i, c in enumerate(alphabet)}

def message_to_bits(message):
    """Convert 40-character message to 240-bit string."""
    if len(message) != 40:
        raise ValueError("Message must be exactly 40 characters")
    return ''.join(char_to_bits[c] for c in message)

def bits_to_message(bits):
    """Convert 240-bit string back to 40-character message."""
    if len(bits) != 240:
        raise ValueError("Bits length must be 240")
    return ''.join(bits_to_char[bits[i:i+6]] for i in range(0, 240, 6))

def transmitter(message, r=15, alpha=0.002):
    """
    Transmitter: Encodes a 40-character message into a signal x.
    
    Parameters:
    - message: str, 40-character message
    - r: int, parameter for code size (default 15)
    - alpha: float, energy scaling factor (default 0.002)
    
    Returns:
    - x: numpy array, transmitted signal
    """
    # Convert message to 240 bits
    bits = message_to_bits(message)
    
    # Split into 15 chunks of 16 bits
    chunks = [bits[i*16:(i+1)*16] for i in range(15)]
    
    N = 2**r  # Size of Hadamard matrix: 32768 for r=15
    M_r = hadamard(N).astype(np.float64)
    block_length = 2 * N  # 65536 per codeword
    x_list = []
    
    for chunk in chunks:
        i_j = int(chunk, 2)  # Index from 0 to 65535
        # Generate x_i from B_r
        if i_j < N:
            x_i = M_r[i_j, :]
        else:
            x_i = -M_r[i_j - N, :]
        # Form codeword c_i = sqrt(alpha) [x_i x_i]
        c_i = np.sqrt(alpha) * np.concatenate((x_i, x_i))
        x_list.append(c_i)
    
    x = np.concatenate(x_list)
    # Verify constraints
    if x.size > 1000000:
        raise ValueError("Signal length exceeds 1,000,000")
    if np.sum(np.square(x)) > 2000:
        raise ValueError("Energy exceeds 2000")
    return x

def receiver(Y, r=15, alpha=0.002):
    """
    Receiver: Decodes received signal Y back to a 40-character message.
    
    Parameters:
    - Y: numpy array, received signal
    - r: int, parameter for code size (default 15)
    - alpha: float, energy scaling factor (default 0.002)
    
    Returns:
    - message: str, decoded 40-character message
    """
    N = 2**r  # 32768
    block_length = 2 * N  # 65536
    num_blocks = 15
    M_r = hadamard(N).astype(np.float64)
    G = 10
    sqrt_G = np.sqrt(G)
    
    decoded_bits = ''
    
    for j in range(num_blocks):
        # Extract segment
        Y_j = Y[j*block_length:(j+1)*block_length]
        Y_j_prime = Y_j[:N]  # First half
        Y_j_double_prime = Y_j[N:]  # Second half
        
        # Compute inner products with all rows of M_r
        ht_prime = M_r @ Y_j_prime
        ht_double_prime = M_r @ Y_j_double_prime
        
        # Inner products with all rows of B_r
        ip_prime = np.concatenate((ht_prime, -ht_prime))
        ip_double_prime = np.concatenate((ht_double_prime, -ht_double_prime))
        
        # Compute scores for all possible i
        scores = np.maximum(
            sqrt_G * ip_prime + ip_double_prime,  # State 1
            ip_prime + sqrt_G * ip_double_prime   # State 2
        )
        
        hat_i = np.argmax(scores)
        # Convert to 16-bit string
        chunk_bits = format(hat_i, '016b')
        decoded_bits += chunk_bits
    
    # Convert bits back to message
    message = bits_to_message(decoded_bits)
    return message

def local_channel(x):
    """Local simulation of the channel for testing."""
    G = 10
    sigma2 = 10
    s = random.choice([1, 2])
    n = x.size
    Z = np.random.normal(0, np.sqrt(sigma2), n)
    if s == 1:
        x_modified = x.copy()
        x_modified[::2] *= np.sqrt(G)
    else:
        x_modified = x.copy()
        x_modified[1::2] *= np.sqrt(G)
    return x_modified + Z

# Example usage and local testing
if __name__ == "__main__":
    # Test message (exactly 40 characters)
    test_message = "Salut je suis heureux j aime les cerises"
    print(f"Original message: {test_message}")
    
    # Transmit
    x = transmitter(test_message)
    print(f"Signal length: {x.size}, Energy: {np.sum(np.square(x)):.2f}")

    # Save a input.txt file
    np.savetxt('input.txt', x)
    
    # Simulate channel
    Y = local_channel(x)
    
    # Receive
    decoded_message = receiver(Y)
    print(f"Decoded message: {decoded_message}")
    
    # Check errors
    errors = sum(a != b for a, b in zip(test_message, decoded_message))
    print(f"Number of character errors: {errors}")