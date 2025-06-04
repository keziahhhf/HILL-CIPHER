# HILL CIPHER
HILL CIPHER
EX. NO: 3 AIM:
 

IMPLEMENTATION OF HILL CIPHER
 
## To write a C program to implement the hill cipher substitution techniques.

## DESCRIPTION:

Each letter is represented by a number modulo 26. Often the simple scheme A = 0, B
= 1... Z = 25, is used, but this is not an essential feature of the cipher. To encrypt a message, each block of n letters is  multiplied by an invertible n × n matrix, against modulus 26. To
decrypt the message, each block is multiplied by the inverse of the m trix used for
 
encryption. The matrix used
 
for encryption is the cipher key, and it sho
 
ld be chosen
 
randomly from the set of invertible n × n matrices (modulo 26).


## ALGORITHM:

STEP-1: Read the plain text and key from the user. STEP-2: Split the plain text into groups of length three. STEP-3: Arrange the keyword in a 3*3 matrix.
STEP-4: Multiply the two matrices to obtain the cipher text of length three.
STEP-5: Combine all these groups to get the complete cipher text.

## PROGRAM 

```
import numpy as np

def text_to_numbers(text):
    return [ord(char) - ord('A') for char in text.upper()]

def numbers_to_text(numbers):
    return ''.join(chr(int(num) + ord('A')) for num in numbers)

def mod_inverse(a, m):
    # Extended Euclidean Algorithm for modular inverse
    a = a % m
    for x in range(1, m):
        if (a * x) % m == 1:
            return x
    raise ValueError("No modular inverse exists")

def matrix_mod_inv(matrix, modulus):
    n = matrix.shape[0]
    det = int(round(np.linalg.det(matrix))) % modulus
    det_inv = mod_inverse(det, modulus)

    # Find matrix of cofactors
    cofactors = np.zeros((n, n), dtype=int)
    for row in range(n):
        for col in range(n):
            minor = np.delete(np.delete(matrix, row, axis=0), col, axis=1)
            cofactor = int(round(np.linalg.det(minor)))
            sign = (-1) ** (row + col)
            cofactors[row, col] = (sign * cofactor) % modulus

    # Adjugate = transpose of cofactor matrix
    adjugate = cofactors.T % modulus

    # Modular inverse matrix
    return (det_inv * adjugate) % modulus


def hill_cipher_encrypt(plaintext, key):
    n = len(key)
    plaintext = plaintext.upper().replace(" ", "")
    
    # Padding with 'X' if not divisible by matrix size
    while len(plaintext) % n != 0:
        plaintext += 'X'
    
    text_numbers = text_to_numbers(plaintext)
    key_matrix = np.array(key)

    encrypted_numbers = []
    for i in range(0, len(text_numbers), n):
        block = np.array(text_numbers[i:i+n]).reshape(n, 1)
        encrypted_block = np.dot(key_matrix, block) % 26
        encrypted_numbers.extend(encrypted_block.flatten())

    return numbers_to_text(encrypted_numbers)

def remove_padding(text):
    return text.rstrip('X')  # Note: Be careful if original text ends in 'X'


def hill_cipher_decrypt(ciphertext, key):
    n = len(key)
    key_matrix = np.array(key)
    inverse_matrix = matrix_mod_inv(key_matrix, 26).astype(int)

    text_numbers = text_to_numbers(ciphertext)
    decrypted_numbers = []

    for i in range(0, len(text_numbers), n):
        block = np.array(text_numbers[i:i+n]).reshape(n, 1)
        decrypted_block = np.dot(inverse_matrix, block) % 26
        decrypted_numbers.extend(decrypted_block.flatten())
    
    return numbers_to_text(decrypted_numbers)

# ==== Driver Code ====
if __name__ == "__main__":
    plaintext = input("Enter plaintext: ")
    key = [[6, 24, 1], 
           [13, 16, 10], 
           [20, 17, 15]]  # 3x3 key matrix
    
    ciphertext = hill_cipher_encrypt(plaintext, key)
    print("Encrypted Text:", ciphertext)
    
    decrypted_text = hill_cipher_decrypt(ciphertext, key)
    decrypted_text = remove_padding(decrypted_text)
    print("Decrypted Text:", decrypted_text)


```

## OUTPUT

![image](https://github.com/user-attachments/assets/730c02cd-85dc-454c-a208-931465e4bbae)


## RESULT
Thus, a python program is implement for hill cipher substitution techniques.
