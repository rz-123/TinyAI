package io.leavesfly.tinyai.nnet.v2.layer.embedding;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.v2.util.GradientChecker;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Embedding层的单元测试
 */
public class EmbeddingTest {

    @Test
    public void testEmbeddingCreation() {
        Embedding embedding = new Embedding("emb", 1000, 64);

        assertEquals("emb", embedding.getName());
        assertEquals(1000, embedding.getNumEmbeddings());
        assertEquals(64, embedding.getEmbeddingDim());
    }

    @Test
    public void testEmbeddingForward1D() {
        Embedding embedding = new Embedding("emb", 1000, 64);

        // 创建1D索引输入 (seq_len=10)
        NdArray indices = NdArray.of(new float[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, Shape.of(10));
        Variable input = new Variable(indices);

        // 前向传播
        Variable output = embedding.forward(input);

        // 验证输出形状 (seq_len=10, embedding_dim=64)
        assertEquals(Shape.of(10, 64), output.getShape());
    }

    @Test
    public void testEmbeddingForward2D() {
        Embedding embedding = new Embedding("emb", 1000, 64);

        // 创建2D索引输入 (batch=2, seq_len=5)
        NdArray indices = NdArray.of(new float[]{
            1, 2, 3, 4, 5,
            6, 7, 8, 9, 10
        }, Shape.of(2, 5));
        Variable input = new Variable(indices);

        // 前向传播
        Variable output = embedding.forward(input);

        // 验证输出形状 (batch=2, seq_len=5, embedding_dim=64)
        assertEquals(Shape.of(2, 5, 64), output.getShape());
    }

    @Test
    public void testEmbeddingGradientCheck() {
        Embedding embedding = new Embedding("emb", 1000, 64);
        
        // 创建2D索引输入 (batch=2, seq_len=5)
        NdArray indices = NdArray.of(new float[]{
            1, 2, 3, 4, 5,
            6, 7, 8, 9, 10
        }, Shape.of(2, 5));
        Variable input = new Variable(indices);
        
        // 使用 GradientChecker 检查计算图连通性
        GradientChecker.checkGraphConnectivity(embedding, input);
    }
}

