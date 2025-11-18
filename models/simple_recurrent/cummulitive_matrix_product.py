import numpy as np

import torch


def batched_cumulative_matrix_multiplication(U):
    batch_size, num_matrices, rows, cols = U.shape
    result = torch.zeros((batch_size, num_matrices, rows, cols), device=U.device, dtype=U.dtype)
    result[:, 0] = U[:, 0]
    for i in range(1, num_matrices):
        result[:, i] = torch.bmm(U[:, i], result[:, i - 1])
    return result


def naive_cumulative_matrix_multiplication(U):
    batch_size, num_matrices, rows, cols = U.shape
    # Cary out the computation in double precision to avoid numerical instability
    result = torch.zeros((batch_size, num_matrices, rows, cols), device=U.device, dtype=torch.float64)
    U = U.to(torch.float64)
    for b in range(batch_size):
        result[b, 0] = U[b, 0]
        for i in range(1, num_matrices):
            result[b, i] = torch.mm(U[b, i], result[b, i - 1])
    return result


def test_batched_cumulative_matrix_multiplication():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Test cases with different shapes
    test_cases = [
        (2, 3, 4, 4),  # batch_size, num_matrices, rows, cols
        (5, 10, 3, 3),
        (1, 100, 2, 2),
        (10, 5, 6, 6),
    ]

    for batch_size, num_matrices, rows, cols in test_cases:
        # Generate random input
        U = torch.rand((batch_size, num_matrices, rows, cols))

        # Compute results using both methods
        batched_result = batched_cumulative_matrix_multiplication(U)
        naive_result = naive_cumulative_matrix_multiplication(U)

        # Check if results are equal within a small tolerance
        assert torch.allclose(batched_result, naive_result, atol=1e-6), \
            f"Results do not match for shape: {U.shape}"

        # Check specific properties
        assert batched_result.shape == naive_result.shape == U.shape, \
            f"Output shape mismatch for input shape: {U.shape}"

        # Check if the first matrix is unchanged
        assert torch.allclose(batched_result[:, 0], U[:, 0]), \
            "First matrix should be unchanged"

        print(f"Test passed for shape: {U.shape}")

    # Test for numerical stability with large numbers
    U_large = torch.rand((2, 5, 3, 3)) * 1e5
    batched_result_large = batched_cumulative_matrix_multiplication(U_large)
    naive_result_large = naive_cumulative_matrix_multiplication(U_large)
    assert torch.allclose(batched_result_large, naive_result_large, rtol=1e-4), \
        "Results do not match for large numbers"

    print("All tests passed successfully!")


if __name__ == '__main__':
    # Run the test
    test_batched_cumulative_matrix_multiplication()
