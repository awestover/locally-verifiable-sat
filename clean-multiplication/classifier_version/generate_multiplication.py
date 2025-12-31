def generate_multiplication_text(A: int, B: int, chunk_size: int = 3) -> str:
    """
    Generate step-by-step multiplication explanation in chunked format.
    
    Args:
        A: First number to multiply
        B: Second number to multiply  
        chunk_size: Number of digits per chunk (default 3)
    
    Returns:
        Formatted string showing the multiplication steps
    """
    BASE = 10 ** chunk_size
    lines = []
    
    # Header
    lines.append(f"{A} x {B}")
    lines.append(f"A={A}")
    lines.append(f"B={B}")
    lines.append("")
    lines.append(f"Chunked multiplication (chunk size = {chunk_size} digits, BASE = {BASE})")
    lines.append("")
    
    # PHASE 1: Split into chunks
    def split_into_chunks(n):
        chunks = []
        while n > 0:
            chunks.append(n % BASE)
            n //= BASE
        if not chunks:
            chunks = [0]
        return chunks
    
    a_chunks = split_into_chunks(A)
    b_chunks = split_into_chunks(B)
    
    lines.append("PHASE 1) Split into {}-digit chunks (right to left)".format(chunk_size))
    lines.append("  A chunks: " + ", ".join(str(c) for c in a_chunks))
    lines.append("  B chunks: " + ", ".join(str(c) for c in b_chunks))
    lines.append("")
    
    # PHASE 2: Convolution
    lines.append("PHASE 2) Convolution: multiply every chunk of A with every chunk of B; put each product into bucket (i+j), where i and j are the chunk distances from the right; add within buckets, no carrying yet.")
    lines.append("")
    
    num_buckets = len(a_chunks) + len(b_chunks) - 1
    buckets = [0] * num_buckets
    
    # For each bucket, collect the terms
    for bucket_idx in range(num_buckets):
        terms = []
        products = []
        for i in range(len(a_chunks)):
            for j in range(len(b_chunks)):
                if i + j == bucket_idx:
                    terms.append((a_chunks[i], b_chunks[j]))
                    products.append(a_chunks[i] * b_chunks[j])
        
        lines.append(f"  bucket{bucket_idx} = " + " + ".join(f"{a}*{b}" for a, b in terms))
        
        if len(products) == 1:
            lines.append(f"          = {products[0]}")
        else:
            # Show progressive addition
            running_sum = products[0]
            lines.append(f"          = " + " + ".join(str(p) for p in products))
            for k in range(1, len(products)):
                if k == 1:
                    running_sum = products[0] + products[1]
                    if len(products) > 2:
                        lines.append(f"          = {running_sum} + " + " + ".join(str(p) for p in products[k+1:]))
                    else:
                        lines.append(f"          = {running_sum}")
                else:
                    new_sum = running_sum + products[k]
                    if k < len(products) - 1:
                        lines.append(f"          = {new_sum} + " + " + ".join(str(p) for p in products[k+1:]))
                    else:
                        lines.append(f"          = {new_sum}")
                    running_sum = new_sum
        
        buckets[bucket_idx] = sum(products)
        lines.append("")
    
    lines.append("Raw buckets:")
    lines.append("  [" + ", ".join(str(b) for b in buckets) + "]")
    lines.append("")
    
    # PHASE 3: Carrying
    lines.append(f"PHASE 3) Carrying (make each bucket < {BASE})")
    lines.append(f"Rule: if bucket = carry*{BASE} + digit, keep digit and add carry to next bucket.")
    lines.append("")
    
    final_chunks = []
    carry = 0
    
    for i in range(len(buckets)):
        buckets[i] += carry
        current_val = buckets[i]
        carry = current_val // BASE
        digit = current_val % BASE
        
        lines.append(f"  bucket{i}={current_val} -> carry={carry}, digit={digit}")
        lines.append(f"    bucket{i}={digit}")
        
        if i < len(buckets) - 1:
            lines.append(f"    bucket{i+1}={buckets[i+1]}+{carry}={buckets[i+1]+carry}")
        else:
            lines.append(f"    top={carry}")
        
        final_chunks.append(digit)
        buckets[i] = digit
        lines.append("")
    
    # Add final carry if any
    if carry > 0:
        final_chunks.append(carry)
    
    # Reverse for left-to-right display
    final_chunks_reversed = final_chunks[::-1]
    
    lines.append("Final chunks (left to right):")
    lines.append("  " + " | ".join(str(c) for c in final_chunks_reversed))
    lines.append("")
    
    # PHASE 4: Recombine
    lines.append(f"PHASE 4) Recombine (pad chunks to {chunk_size} digits except the first)")
    
    # Build the final number string
    result_str = str(final_chunks_reversed[0])
    for c in final_chunks_reversed[1:]:
        result_str += str(c).zfill(chunk_size)
    
    lines.append(f"  {result_str}")
    lines.append("")
    
    lines.append("Answer:")
    lines.append(f"  {A} * {B} = ")
    lines.append(f"\\box{{{result_str}}}")
    
    return "\n".join(lines)


if __name__ == "__main__":
    # Example from the file
    text = generate_multiplication_text(123456789, 987654321)
    print(text)

