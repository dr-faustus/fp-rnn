import unittest

import torch

from full_matrix import FullMatrix, FullMatrixConfig
from simple_recurrent.cumulative_matrix_product import batched_cumulative_matrix_multiplication, \
    naive_cumulative_matrix_multiplication


class TestFullMatrix(unittest.TestCase):
    def setUp(self):
        embedding_dim = 2
        self.config = FullMatrixConfig(embedding_dim=embedding_dim)
        self.model = FullMatrix(self.config)

        # Set random seed for reproducibility
        torch.manual_seed(42)

        # Create a fixed input tensor
        self.input_tensor = torch.randn((2, 6, 2))

        # Set fixed weights for A and B
        with torch.no_grad():
            self.model.A.weight.data = torch.tensor(
                torch.randn(embedding_dim * embedding_dim * embedding_dim, dtype=torch.float).reshape(
                    embedding_dim * embedding_dim, embedding_dim)
            )
            self.model.A.bias.data = torch.tensor([0.1, 0.2, 0.3, 0.4])
            self.model.B.weight.data = torch.tensor(
                torch.randn(embedding_dim * embedding_dim * embedding_dim, dtype=torch.float).reshape(
                    embedding_dim * embedding_dim, embedding_dim) / 10
            )
            self.model.B.bias.data = torch.tensor([0.1, 0.2, 0.3, 0.4])

    def test_forward_methods_equality(self):
        # for simplicity no batching
        input_tensor = self.input_tensor[0].unsqueeze(0)
        with torch.no_grad():
            loop_output_naive = self.model.forward_loop_naive(input_tensor)
            efficient_output = self.model.forward_efficient(input_tensor)
            loop_output = self.model.forward_loop(input_tensor)

        self.assertTrue(torch.allclose(loop_output_naive, loop_output, atol=1e-6))
        for t in range(5):
            self.assertTrue(torch.allclose(efficient_output[0, t], loop_output[0, t], atol=1e-6),
                            f"Mismatch at time step {t}:\nEfficient: {efficient_output[0, t]}\nLoop: {loop_output[0, t]}")

    def test_intermediate_values(self):
        batch_size, sequence_length, emb_dim = self.input_tensor.shape
        input_tensor = self.input_tensor
        with torch.no_grad():
            # Test A and B tensors
            A = self.model.A(input_tensor).view(batch_size, sequence_length, emb_dim, emb_dim)
            B = self.model.B(input_tensor).view(batch_size, sequence_length, emb_dim, emb_dim)

            # Test Bx calculation
            Bx = torch.matmul(B, input_tensor.unsqueeze(-1)).squeeze(-1)
            for idx in range(sequence_length):
                for batch_idx in range(batch_size):
                    self.assertTrue(torch.allclose(Bx[batch_idx, idx],
                                                   torch.matmul(B[batch_idx, idx], input_tensor[batch_idx, idx]),
                                                   atol=1e-6))

            # Test A_cumprod calculation
            A_sequence = A[:, 1:]
            A_cumprod = batched_cumulative_matrix_multiplication(A_sequence)
            A_cumprod_naive = naive_cumulative_matrix_multiplication(A_sequence)
            self.assertTrue(torch.allclose(A_cumprod, A_cumprod_naive.to(torch.float32), atol=1e-6))

            # Check that the initial A is correct
            for batch_idx in range(batch_size):
                self.assertTrue(torch.allclose(A_cumprod[batch_idx, 0], A_sequence[batch_idx, 0], atol=1e-6))
            for idx in range(1, sequence_length - 1):
                for batch_idx in range(batch_size):
                    self.assertTrue(torch.allclose(
                        A_cumprod[batch_idx, idx],
                        torch.matmul(A_sequence[batch_idx, idx], A_cumprod[batch_idx, idx - 1]),
                        atol=1e-6))

            # Print intermediate values for debugging
            print("A tensor:", A)
            print("B tensor:", B)
            print("Bx tensor:", Bx)
            print("A_cumprod tensor:", A_cumprod)

    def test_edge_cases(self):
        # Test with zero input
        zero_input = torch.zeros(1, 5, 2)
        with torch.no_grad():
            zero_efficient = self.model.forward_efficient(zero_input)
            zero_loop = self.model.forward_loop(zero_input)
        self.assertTrue(torch.allclose(zero_efficient, zero_loop, atol=1e-6))

        # Test with very large input
        large_input = torch.ones(1, 5, 2) * 1e6
        with torch.no_grad():
            large_efficient = self.model.forward_efficient(large_input)
            large_loop = self.model.forward_loop(large_input)
        self.assertTrue(torch.allclose(large_efficient, large_loop, atol=1e-6))


if __name__ == '__main__':
    unittest.main()
