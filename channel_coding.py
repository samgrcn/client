import numpy as np
from scipy.linalg import hadamard
import random

# alphabet size 64 (2^6 => need 6 bits to encode it)
alphabet = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .'
char_to_bits = {c: format(i, '06b') for i, c in enumerate(alphabet)}
bits_to_char = {format(i, '06b'): c for i, c in enumerate(alphabet)}

def message_to_bits(message):
    if len(message) != 40:
        raise ValueError("Message must be exactly 40 characters")
    return ''.join(char_to_bits[c] for c in message)

def bits_to_message(bits):
    if len(bits) != 240:
        raise ValueError("Bits length must be 240")
    return ''.join(bits_to_char[bits[i:i+6]] for i in range(0, 240, 6))

def transmitter(message, r=15, alpha=0.002):
    bits = message_to_bits(message)
    
    # 15 chunks
    chunks = [bits[i*16:(i+1)*16] for i in range(15)]
    
    N = 2**r
    M_r = hadamard(N).astype(np.float64) # 32768 x 32768
    block_length = 2 * N
    x_list = []
    
    for chunk in chunks:
        i_j = int(chunk, 2)
        if i_j < N:
            x_i = M_r[i_j, :]
        else:
            x_i = -M_r[i_j - N, :]
        c_i = np.sqrt(alpha) * np.concatenate((x_i, x_i))
        x_list.append(c_i)
    
    x = np.concatenate(x_list)
    if x.size > 1000000:
        raise ValueError("Signal length >1,000,000")
    if np.sum(np.square(x)) > 2000:
        raise ValueError("Energy >2000")
    return x

def receiver(Y, r=15, alpha=0.002):
    # Y length 983040 => 15 segmenrs of 65536 elmt each
    N = 2**r
    block_length = 2 * N
    num_blocks = 15
    M_r = hadamard(N).astype(np.float64)
    G = 10
    sqrt_G = np.sqrt(G)
    
    decoded_bits = ''
    
    for j in range(num_blocks):
        # Split into chunks
        Y_j = Y[j*block_length:(j+1)*block_length]

        # Split in two halves
        Y_j_prime = Y_j[:N]
        Y_j_double_prime = Y_j[N:]
        
        # Inner product
        ht_prime = M_r @ Y_j_prime
        ht_double_prime = M_r @ Y_j_double_prime
        
        # Extend to include negative rows
        ip_prime = np.concatenate((ht_prime, -ht_prime))
        ip_double_prime = np.concatenate((ht_double_prime, -ht_double_prime))
        
        # Pick the most likely transmitted codeword (0 to 65535)
        scores = np.maximum(
            sqrt_G * ip_prime + ip_double_prime,
            ip_prime + sqrt_G * ip_double_prime
        )
        hat_i = np.argmax(scores)

        # Convert to 16-bit string => 240 bits
        chunk_bits = format(hat_i, '016b')
        decoded_bits += chunk_bits
    
    # Convert 240 bits to text
    message = bits_to_message(decoded_bits)
    return message

def local_channel(x):
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

if __name__ == "__main__":
    test_message = "Salut je suis heureux j aime les cerises"
    print(f"Original message: {test_message}")
    
    x = transmitter(test_message)
    print(f"Signal length: {x.size}, Energy: {np.sum(np.square(x)):.2f}")

    np.savetxt('input.txt', x)
    
    Y = local_channel(x)
    
    decoded_message = receiver(Y)
    print(f"Decoded message: {decoded_message}")
    
    errors = sum(a != b for a, b in zip(test_message, decoded_message))
    print(f"Number of character errors: {errors}")