package io.leavesfly.tinyai.qwen3.training;

import io.leavesfly.tinyai.qwen3.Qwen3Config;
import org.junit.Test;
import java.util.ArrayList;
import java.util.List;
import static org.junit.Assert.*;

/**
 * Qwen3Dataset单元测试
 * 
 * @author leavesfly
 * @version 1.0
 */
public class Qwen3DatasetTest {
    
    @Test
    public void testDatasetCreation() {
        List<int[]> sequences = new ArrayList<>();
        sequences.add(new int[]{1, 2, 3, 4, 5});
        sequences.add(new int[]{6, 7, 8, 9, 10});
        
        Qwen3Dataset dataset = new Qwen3Dataset(sequences, 128, 2, false);
        
        assertNotNull(dataset);
        assertEquals(2, dataset.getSampleCount());
        assertEquals(1, dataset.getBatchCount());
    }
    
    @Test
    public void testCreateDemoDataset() {
        Qwen3Dataset dataset = Qwen3Dataset.createDemoDataset(
            10000,  // vocabSize
            50,     // numSamples
            128,    // maxSeqLength
            4       // batchSize
        );
        
        assertNotNull(dataset);
        assertEquals(50, dataset.getSampleCount());
        assertEquals(13, dataset.getBatchCount());  // ceiling(50/4)
    }
    
    @Test
    public void testBatchIteration() {
        List<int[]> sequences = new ArrayList<>();
        for (int i = 0; i < 10; i++) {
            sequences.add(new int[]{i, i+1, i+2, i+3, i+4});
        }
        
        Qwen3Dataset dataset = new Qwen3Dataset(sequences, 128, 3, false);
        dataset.prepare(false);
        
        int batchCount = 0;
        while (dataset.hasNext()) {
            Qwen3Dataset.Batch batch = dataset.nextBatch();
            assertNotNull(batch);
            assertNotNull(batch.getInputIds());
            assertNotNull(batch.getTargetIds());
            batchCount++;
        }
        
        assertEquals(4, batchCount);  // ceiling(10/3)
    }
    
    @Test
    public void testBatchShape() {
        List<int[]> sequences = new ArrayList<>();
        sequences.add(new int[]{1, 2, 3, 4, 5, 6, 7, 8});
        sequences.add(new int[]{9, 10, 11, 12, 13, 14, 15, 16});
        
        int maxSeqLen = 10;
        Qwen3Dataset dataset = new Qwen3Dataset(sequences, maxSeqLen, 2, false);
        dataset.prepare(false);
        
        Qwen3Dataset.Batch batch = dataset.nextBatch();
        
        // 验证批次形状
        assertEquals(2, batch.getBatchSize());
        assertEquals(maxSeqLen, batch.getSeqLength());
        assertEquals(2, batch.getInputIds().getShape().getDimension(0));
        assertEquals(maxSeqLen, batch.getInputIds().getShape().getDimension(1));
    }
    
    @Test
    public void testInputTargetAlignment() {
        List<int[]> sequences = new ArrayList<>();
        sequences.add(new int[]{1, 2, 3, 4, 5});
        
        Qwen3Dataset dataset = new Qwen3Dataset(sequences, 10, 1, false);
        dataset.prepare(false);
        
        Qwen3Dataset.Batch batch = dataset.nextBatch();
        
        // 输入应该是前n-1个token
        // 目标应该是后n-1个token
        float input0 = batch.getInputIds().get(0, 0);
        float target0 = batch.getTargetIds().get(0, 0);
        
        // input[0] = seq[0] = 1
        // target[0] = seq[1] = 2
        assertEquals(1.0f, input0, 0.001f);
        assertEquals(2.0f, target0, 0.001f);
    }
    
    @Test
    public void testReset() {
        List<int[]> sequences = new ArrayList<>();
        for (int i = 0; i < 5; i++) {
            sequences.add(new int[]{i, i+1, i+2});
        }
        
        Qwen3Dataset dataset = new Qwen3Dataset(sequences, 10, 2, false);
        
        // 第一次遍历
        dataset.prepare(false);
        while (dataset.hasNext()) {
            dataset.nextBatch();
        }
        assertFalse(dataset.hasNext());
        
        // 重置后可以再次遍历
        dataset.reset();
        dataset.prepare(false);
        assertTrue(dataset.hasNext());
    }
    
    @Test
    public void testShuffling() {
        List<int[]> sequences = new ArrayList<>();
        for (int i = 0; i < 100; i++) {
            sequences.add(new int[]{i, i+1, i+2});
        }
        
        // 创建两个数据集，一个打乱，一个不打乱
        Qwen3Dataset shuffledDataset = new Qwen3Dataset(sequences, 10, 10, true);
        Qwen3Dataset orderedDataset = new Qwen3Dataset(sequences, 10, 10, false);
        
        shuffledDataset.prepare(true);  // 打乱
        orderedDataset.prepare(false);  // 不打乱
        
        Qwen3Dataset.Batch shuffledBatch = shuffledDataset.nextBatch();
        Qwen3Dataset.Batch orderedBatch = orderedDataset.nextBatch();
        
        // 打乱后的第一批很可能与未打乱的不同（虽然不是100%保证）
        // 这里只验证两者都能正常生成批次
        assertNotNull(shuffledBatch);
        assertNotNull(orderedBatch);
    }
    
    @Test
    public void testPosttrainMode() {
        List<int[]> sequences = new ArrayList<>();
        sequences.add(new int[]{1, 2, 3, 4, 5});
        
        List<String> prompts = new ArrayList<>();
        prompts.add("Hello");
        
        List<String> responses = new ArrayList<>();
        responses.add("World");
        
        Qwen3Dataset dataset = new Qwen3Dataset(
            sequences, prompts, responses, 10, 1, false
        );
        
        dataset.prepare(false);
        Qwen3Dataset.Batch batch = dataset.nextBatch();
        
        // 验证微调数据可用
        assertNotNull(batch.getPrompts());
        assertNotNull(batch.getResponses());
        assertEquals("Hello", batch.getPrompts()[0]);
        assertEquals("World", batch.getResponses()[0]);
    }
}
